from typing import List
import torch
import numpy as np
from PIL import Image   

def to_tensor(image:Image, size):
    with torch.no_grad():

        width, height = image.size
        
        # Calculate the size of the square crop
        crop_size = min(width, height)
        
        # Calculate the coordinates for cropping
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        
        # Perform the center crop
        cropped_image = image.crop((left, top, right, bottom))

        image = image.resize(size, Image.Resampling.LANCZOS)

        image_np = np.array(image).astype(np.float32) / 255.0
        
        # Normalize the image
        image_np = (image_np - 0.5) * 2.0
        
        # Convert the image to a PyTorch tensor
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        
        # Encode the image using the VAE
        
        return image_tensor

def vae_encode(image:Image, size, vae, accelerator):
    with torch.no_grad():

        image_tensor = to_tensor(image, size).to(accelerator.device).unsqueeze(0)
        
        # Encode the image using the VAE
        
        encoded_image = vae.encode(image_tensor).latent_dist.sample().squeeze() * vae.config.scaling_factor


    return encoded_image

def vae_encode_batch(images: torch.Tensor, vae, vae_batch_size: int, accelerator):
    batch_size = images.shape[0]
    assert vae_batch_size <= batch_size, "VAE batch size must be less than or equal to the total batch size"
    
    encoded_images = []
    images = images.to(accelerator.device)
    
    with torch.no_grad():
        for i in range(0, batch_size, vae_batch_size):
            sub_batch = images[i:i+vae_batch_size]
            encoded_sub_batch = vae.encode(sub_batch).latent_dist.sample() * vae.config.scaling_factor
            encoded_images.append(encoded_sub_batch)
    
    # Combine all encoded sub-batches into a single tensor
    return torch.cat(encoded_images, dim=0)

def vae_decode(latent, vae):

    with torch.no_grad():
        decoded_image = vae.decode(latent.detach()[0].unsqueeze(0) / vae.config.scaling_factor).sample

        # Convert the processed image back to a PIL Image
        decoded_image_np = decoded_image.squeeze().permute(1, 2, 0).float().cpu().numpy()
        
        # Denormalize the pixel values
        decoded_image_np = (decoded_image_np / 2.0 + 0.5) * 255.0
        
        # Clip the pixel values to the valid range [0, 255]
        decoded_image_np = np.clip(decoded_image_np, 0, 255).astype(np.uint8)
        
        # Convert the NumPy array to a PIL image
        decoded_image = Image.fromarray(decoded_image_np)

    return decoded_image