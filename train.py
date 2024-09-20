import os
import random
import torch
from data import CachedDataset, TransfusionDataset, create_text_image_pairs, load_pairs_from_disk, patchify, unpatchify, resume_checkpoint, save_checkpoint, save_pairs_to_disk
from inference import debug_image
from ogen import OGen
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL
from vae import vae_decode, vae_encode_batch
import argparse
from tqdm import tqdm
from accelerate import Accelerator

def train():
    # Add command-line argument parsing
    parser = argparse.ArgumentParser(description='Train the Transfusion model')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--model_name', type=str, default='HuggingFaceTB/SmolLM-1.7B', help='Name of the model to use')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--image_size', type=int, default=256, help='Image size for training')
    parser.add_argument('--patch_size', type=int, default=2, help='Patch size for training')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Use gradient checkpointing')
    parser.add_argument('--diffusion_loss_weight', type=float, default=5, help='Weight for the diffusion loss')
    parser.add_argument('--max_length', type=int, default=128, help='Max length for the input text')
    parser.add_argument('--warmup_steps', type=int, default=100, help='Warmup steps for the scheduler')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='Transformer hidden dimension')
    parser.add_argument('--depth', type=int, default=20, help='Transformer depth')
    parser.add_argument('--heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--debug_steps', type=int, default=50, help='Number of steps to debug')
    parser.add_argument('--inference_steps', type=int, default=100, help='Number of steps to inference')
    parser.add_argument('--save_steps', type=int, default=200, help='Number of steps to save')
    parser.add_argument('--cache', action='store_true', help='Recache the dataset')
    parser.add_argument('--cache_batch_size', type=int, default=1, help='Batch size for cache loading')
    parser.add_argument('--optimizer', type=str, default="adamw_schedulefree", help='Optimizer')

    args = parser.parse_args()
    model_name = args.model_name

    accelerator = Accelerator(mixed_precision='bf16' if torch.cuda.is_available() else None)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    vae_name = "madebyollin/sdxl-vae-fp16-fix"
    vae = AutoencoderKL.from_pretrained(vae_name).to(accelerator.device)
    vae.requires_grad = False
    max_length = args.max_length
    image_size = args.image_size
    batch_size = args.batch_size
    patch_size = args.patch_size
    learning_rate = args.learning_rate
    diffusion_loss_weight = args.diffusion_loss_weight
    warmup_steps = args.warmup_steps
    N_debug = args.debug_steps
    N_inference = args.inference_steps
    N_save = args.save_steps
    num_epochs = 10
    N_loss_window = 100  # Number of steps for moving average
    cache_batch_size = args.cache_batch_size
    gradient_checkpointing = args.gradient_checkpointing

    print(f"Model name: {model_name}")
    print(f"Learning rate: {learning_rate}")
    print(f"Gradient checkpointing: {gradient_checkpointing}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {image_size}")
    print(f"Patch size: {patch_size}")
    print(f"Diffusion loss weight: {diffusion_loss_weight}")
    print(f"Debug: {N_debug} Inference: {N_inference} Save: {N_save}")
    print(f"Max length: {max_length}")
    print(f"VAE: {vae_name}")
    print(f"Cache batch size: {cache_batch_size}")

    transformer = AutoModelForCausalLM.from_pretrained(model_name)

    model = OGen(transformer=transformer,num_patches=image_size//patch_size//8, text_seq_len=max_length, patch_size=patch_size)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params}")

    if not os.path.isfile("pairs.pkl"):
        pairs = create_text_image_pairs(["source"])
        save_pairs_to_disk(pairs,"pairs.pkl")
    text_image_pairs = load_pairs_from_disk('pairs.pkl')
    print(f"Loaded {len(text_image_pairs)} text-image pairs")

    dataset = TransfusionDataset(text_image_pairs, tokenizer, model, text_seq_len=max_length, image_size=image_size)

    torch.cuda.empty_cache()
    # Cache the dataset
    cache_dir = 'dataset_cache'
    os.makedirs(cache_dir, exist_ok=True)

    if args.cache or not os.path.exists(cache_dir) or len(os.listdir(cache_dir)) == 0:
        temp_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, multiprocessing_context='fork' if torch.backends.mps.is_available() else None)

        for i, batch in enumerate(tqdm(temp_dataloader, desc="Caching dataset")):
            batch["image_latents"] = vae_encode_batch(batch["pixel_values"], vae, vae_batch_size=batch_size, accelerator=accelerator)
            del batch["pixel_values"]

            cache_file = os.path.join(cache_dir, f'batch_{batch_size}_{i}.pt')
            torch.save(batch, cache_file)
    else:
        print("Using existing cached dataset.")

    cached_dataset = CachedDataset(cache_dir, batch_size, accelerator)
    dataloader = DataLoader(cached_dataset, batch_size=cache_batch_size, shuffle=False, num_workers=cache_batch_size, multiprocessing_context='fork' if torch.backends.mps.is_available() else None)

    # Optimizer
    optim = "adamw_schedulefree"
    if optim == "adamw8bit":
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    elif optim == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    elif optim == "adamw_schedulefree":
        from schedulefree import AdamWScheduleFree
        optimizer = AdamWScheduleFree(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), foreach=torch.cuda.is_available(), warmup_steps=warmup_steps)
        optimizer.train()
    else:
        raise ValueError(f"Optimizer {optim} not found")
    
    # Add LR scheduler
    from transformers import get_linear_schedule_with_warmup

    # Calculate the total number of training steps
    total_steps = len(dataloader) * num_epochs

    # Create the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Initialize epoch and step counter
    start_epoch = 0
    step_counter = 0

    # Print optimizer class
    print(f"Optimizer: {optimizer.__class__.__name__}")

    # Load checkpoint if resume flag is set
    if args.resume:
        model, optimizer, scheduler, start_epoch, step_counter = resume_checkpoint(model, optimizer, scheduler)

    # Prepare everything with accelerator
    model, vae, optimizer, dataloader, scheduler = accelerator.prepare(model, vae, optimizer, dataloader, scheduler)

    model = model.to(accelerator.device)
    print("Model device: ", accelerator.device)

    # Sample for inference
    sample_batch = torch.load(os.path.join(cache_dir, cached_dataset.cache_files[0]))
    sample_text = sample_batch['input_ids'][0].unsqueeze(0).to(accelerator.device)
    sample_latents = sample_batch['image_latents'][0].unsqueeze(0).to(accelerator.device)

    # Decode the sample latents using the VAE
    decoded_sample_latents = vae_decode(sample_latents, accelerator.unwrap_model(vae))

    # Create a folder to save the decoded sample latents if it doesn't exist
    os.makedirs('inference_results', exist_ok=True)
    decoded_sample_latents.save(f'inference_results/sample_latents_epoch_0.png')
    
    # Add noise to the sample latents
    torch.manual_seed(42)
    noise = torch.randn_like(sample_latents)
    
    sample_latents = 0.5 * sample_latents + 0.5 * noise

    text_loss_window = []
    diffusion_loss_window = []

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for epoch in range(start_epoch, num_epochs):

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/10", leave=False)
        for i, batch in enumerate(progress_bar, 1):
            step_counter += 1

            bsz = batch['input_ids'].shape[1] * cache_batch_size
            text = batch['input_ids'].reshape(bsz, -1).clone().to(accelerator.device)
            latents = batch['image_latents'].reshape(bsz, 4, image_size // 8, image_size // 8).clone().to(accelerator.device)

            latents = patchify(latents, patch_size)

            timestep = random.randint(0, 1000)

            if step_counter % N_inference == 0 or step_counter % N_debug == 0:
                timestep = 700 
            
            with accelerator.autocast():
                
                loss, text_loss, image_loss, denoised_tokens, noise, flow, pred_flow, noised_image = model.training_step(texts=text, images=latents, timestep=timestep)

                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Update loss windows
                text_loss_window.append(text_loss.item())
                diffusion_loss_window.append(image_loss.item())
                
                if len(text_loss_window) > N_loss_window:
                    text_loss_window.pop(0)
                if len(diffusion_loss_window) > N_loss_window:
                    diffusion_loss_window.pop(0)
                
                # Calculate moving averages
                avg_text_loss = sum(text_loss_window) / len(text_loss_window)
                avg_diffusion_loss = sum(diffusion_loss_window) / len(diffusion_loss_window)
                
                progress_bar.set_postfix({
                    'dloss': f"{avg_diffusion_loss:.4f}",
                    'tloss': f"{avg_text_loss:.4f}",        
                    'lr': f"{scheduler.get_last_lr()[0]:.6f}"
                })

                if step_counter % N_debug == 0:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    # Create partial unpatchify function with arguments already applied
                    unpatchify2 = lambda x: unpatchify(x, patch_size, bsz, 4, image_size // 8, image_size // 8)
                    debug_image(unwrapped_model, unpatchify2, accelerator.unwrap_model(vae), latents, noise, pred_flow, flow, noised_image, denoised_tokens, epoch, step_counter)

                
                if step_counter % N_inference == 0:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    #inference(unwrapped_model, accelerator.unwrap_model(vae), optimizer, sample_text, sample_latents, f'inference_results/inference_epoch_{epoch+1}_step_{step_counter}.png')

                # Save the model every N steps
                if step_counter % N_save == 0:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    save_checkpoint(unwrapped_model, optimizer, scheduler, loss, epoch, step_counter)

if __name__ == '__main__':
    train() 

