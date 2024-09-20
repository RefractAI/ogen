    # Function for inference
import os
import torch
from tqdm import tqdm
import torch.nn.functional as F

from vae import vae_decode
from PIL import Image

def inference(model, vae, optimizer, text, latents, save_path, steps=50, start_timestep=0.7):
        with torch.no_grad():
            model.eval()
            if torch.cuda.is_available():
                 optimizer.eval()

            #todo add noise before inference loop
            _, _, denoised_tokens, _, _, _, _ = model(text=text, latents=latents, num_inference_steps=steps, return_loss=False, start_timestep=start_timestep)
   
            decoded_images = vae_decode(denoised_tokens, vae)

            # Create a folder to save the images if it doesn't exist
            os.makedirs('inference_results', exist_ok=True)

            # Save the decoded image
            decoded_images.save(save_path)
            print("Inference complete.")
            model.train()
            if torch.cuda.is_available():
                 optimizer.train()

def debug_image(model, unpatchify, vae, image_patches, noise, predicted_noise, target_noise, noisy_latent, denoised_image, epoch, step_counter):

    image_patches = unpatchify(image_patches)
    noise = unpatchify(noise)
    predicted_noise = unpatchify(predicted_noise)
    target_noise = unpatchify(target_noise)
    noisy_latent = unpatchify(noisy_latent)
    denoised_image = unpatchify(denoised_image)


    decoded_image_patches = vae_decode(image_patches, vae)
    decoded_noise = vae_decode(noise, vae)
    decoded_target_noise = vae_decode(target_noise, vae)
    decoded_predicted_noise = vae_decode(predicted_noise, vae)
    decoded_noisy_latent = vae_decode(noisy_latent, vae)
    decoded_denoised_image = vae_decode(denoised_image, vae)  

    # Function to convert tensor to PIL Image
    def tensor_to_pil(tensor):
        tensor = tensor.cpu().detach()
        tensor = (tensor + 1) / 2  # Rescale from [-1, 1] to [0, 1]
        tensor = tensor.clamp(0, 1)
        if tensor.shape[0] > 3:
            tensor = tensor[:3]  # Take only the first 3 channels
        return Image.fromarray((tensor.permute(1, 2, 0).numpy() * 255).astype('uint8'))
    
    undecoded_image_patches = tensor_to_pil(image_patches[0])
    undecoded_noise = tensor_to_pil(noise[0])
    undecoded_target_noise = tensor_to_pil(target_noise[0])
    undecoded_predicted_noise = tensor_to_pil(predicted_noise[0])
    undecoded_noisy_latent = tensor_to_pil(noisy_latent[0])
    undecoded_denoised_image = tensor_to_pil(denoised_image[0])

    # Resize un-decoded images to match the size of decoded images
    target_size = (decoded_image_patches.width, decoded_image_patches.height)
    undecoded_image_patches = undecoded_image_patches.resize(target_size, Image.NEAREST)
    undecoded_noise = undecoded_noise.resize(target_size, Image.NEAREST)
    undecoded_target_noise = undecoded_target_noise.resize(target_size, Image.NEAREST)
    undecoded_predicted_noise = undecoded_predicted_noise.resize(target_size, Image.NEAREST)
    undecoded_noisy_latent = undecoded_noisy_latent.resize(target_size, Image.NEAREST)
    undecoded_denoised_image = undecoded_denoised_image.resize(target_size, Image.NEAREST)

    # Create a new combined image with both decoded and un-decoded versions
    W = decoded_image_patches.width
    combined_image = Image.new('RGB', (W * 6, W * 2))

    # Paste decoded images in the first row
    combined_image.paste(decoded_image_patches, (0, 0))
    combined_image.paste(decoded_noise, (W, 0))
    combined_image.paste(decoded_target_noise, (W * 2, 0))
    combined_image.paste(decoded_predicted_noise, (W * 3, 0))
    combined_image.paste(decoded_noisy_latent, (W * 4, 0))
    combined_image.paste(decoded_denoised_image, (W * 5, 0))

    # Paste un-decoded images in the second row
    combined_image.paste(undecoded_image_patches, (0, W))
    combined_image.paste(undecoded_noise, (W, W))
    combined_image.paste(undecoded_target_noise, (W * 2, W))
    combined_image.paste(undecoded_predicted_noise, (W * 3, W))
    combined_image.paste(undecoded_noisy_latent, (W * 4, W))
    combined_image.paste(undecoded_denoised_image, (W * 5, W))

    os.makedirs('debug_results', exist_ok=True)

    # Save the combined image
    combined_image.save(f'debug_results/debug_epoch_{epoch+1}_step_{step_counter}.png')



