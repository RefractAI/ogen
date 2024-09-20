import torch
import torch.nn as nn
import math

from tqdm import tqdm

class SinusoidalEmbedder(nn.Module):
    def __init__(self, dim, max_period=1000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        scaling_factor = math.log(self.max_period) / (half_dim - 1)
        scaling_factor = torch.exp(torch.arange(half_dim, device=device) * -scaling_factor)
        embeddings = timesteps[:, None].float() * scaling_factor[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        return embeddings
    
class OGen(nn.Module):
    def __init__(self, transformer, num_patches, patch_size, text_seq_len, num_timesteps=1000):
        super(OGen, self).__init__()
        
        # Initialize the Phi-2 transformer model (as Phi-3 is not publicly available)
        self.transformer = transformer
        self.hidden_size = transformer.config.hidden_size
        self.num_patches = num_patches
        self.num_timesteps = num_timesteps
        
        # Linear layer to project VAE outputs to transformer embedding size
        self.latent_dim = 4 * patch_size * patch_size
        self.image_proj = nn.Linear(self.latent_dim, self.transformer.config.hidden_size)
        self.image_unproj = nn.Linear(self.transformer.config.hidden_size, self.latent_dim)
        
        # Sine-Cosine 2D Positional Embeddings for images
        self.image_pos_embeddings = self._create_sine_cosine_positional_embeddings(self.num_patches, self.num_patches, self.hidden_size)
        
        # Positional embeddings for text
        self.text_pos_embeddings = nn.Parameter(torch.zeros(1, text_seq_len, self.transformer.config.hidden_size))
        nn.init.normal_(self.text_pos_embeddings, std=0.02)

        # Timestep embedding
        self.timestep_embedding = SinusoidalEmbedder(self.transformer.config.hidden_size)
        

        
    def _create_sine_cosine_positional_embeddings(self, height, width, embed_dim):
        """
        Creates sine-cosine 2D positional embeddings.
        """
        pe = torch.zeros(height * width, embed_dim)
        position = torch.arange(0, height * width, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, height*width, embed_dim)
        return nn.Parameter(pe, requires_grad=False)
    
    def forward(self, input_ids, input_images, timestep):
        """
        input_ids: List of strings
        input_images: Tensor of shape (batch_size, channels, height, width)
        timesteps: Integer
        """

        image_embeddings = self.image_proj(input_images)  # Shape: (batch, num_patches, hidden_size)
        image_embeddings = image_embeddings + self.image_pos_embeddings[:, :image_embeddings.size(1), :]  # Add positional embeddings
        
        # Combine text and image embeddings
        text_embeddings = self.transformer.model.embed_tokens(input_ids) + self.text_pos_embeddings[:, :input_ids.size(1), :]
        embeddings = torch.cat([text_embeddings, image_embeddings], dim=1)
        
        # Add timestep embeddings
        timestep_tensor = torch.full((embeddings.shape[0],), timestep, device=embeddings.device, dtype=torch.long)
        timestep_emb = self.timestep_embedding(timestep_tensor).unsqueeze(1)
        embeddings = embeddings + timestep_emb 
        
        # Construct attention mask
        text_length = input_ids.size(1)
        image_length = image_embeddings.size(1)
        total_length = text_length + image_length
        
        attention_mask = torch.ones((embeddings.size(0), 1, total_length, total_length), device=self.transformer.device)
        
        outputs = self.transformer(inputs_embeds=embeddings, attention_mask=attention_mask, output_hidden_states=True)
        
        return outputs

    def inference(self, input_conditions, steps=30):
        """
        Perform inference using the flow matching method.
        """
        # Sample Gaussian noise
        noise = torch.randn_like(input_conditions)
        
        # Iterate flow matching steps
        latent = noise
        for t in tqdm(range(steps), desc="Flow matching steps"):
            timestep = torch.full((latent.size(0),), t, device=latent.device, dtype=torch.long)
            velocity = self.forward(input_conditions, latent, timestep).hidden_states[-1]
            latent = latent + velocity * (1/steps)  # Update rule based on rectified flow
        
        # Decode latent to image
        with torch.no_grad():
            generated_image = self.vae.decode(latent).sample
        
        return generated_image

    def training_step(self, texts, images, timestep):
        """
        Performs a single training step.
        """
        noise = torch.randn_like(images)
        image_fraction = timestep/self.num_timesteps

        noisy_images = image_fraction * images + (1-image_fraction) * noise
        noisy_images_original = noisy_images.detach().clone()
        target_velocities = images - noise

        outputs = self.forward(texts, noisy_images, timestep)
        predicted_velocities = outputs.hidden_states[-1]

        # Separate text and image predictions
        text_length = texts.size(1)
        text_predictions = self.transformer.lm_head(predicted_velocities[:, :text_length, :])
        image_predictions = self.image_unproj(predicted_velocities[:, text_length:, :])

        # Autoregressive loss for text
        text_targets = texts[:, 1:]  # Shift right to get next token
        text_predictions = text_predictions[:, :-1, :]  # Remove last prediction

        # Modify text prediction
        text_loss = nn.CrossEntropyLoss()(text_predictions.reshape(-1, text_predictions.size(-1)), text_targets.reshape(-1))

        # MSE loss for images
        image_loss = nn.MSELoss()(image_predictions, target_velocities)

        # Combine losses
        loss = text_loss + image_loss * 5

        denoised_image = noisy_images_original + (1-image_fraction) * image_predictions

        return loss, text_loss, image_loss, denoised_image, noise, target_velocities, image_predictions, noisy_images
