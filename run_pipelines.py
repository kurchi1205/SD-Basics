import os
import torch
from tqdm import tqdm
from typing import List
from variant_loss import variant_loss
from pipelines import get_pipelines, load_styles
from utils import set_timesteps, latents_to_pil


def get_images_with_variant_loss(prompt: str, version: str, rgb_value: List, num_images: int, steps: int, height: int, width: int, guidance_scale: float, seed: int = 32, variant_loss_scale: int = 200, save_images=True):
    pipeline = get_pipelines(version=version)
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = pipeline['tokenizer']
    text_encoder = pipeline['text_encoder'].to(torch_device)
    vae = pipeline['vae'].to(torch_device)
    unet = pipeline['unet'].to(torch_device)
    scheduler = pipeline['scheduler']
    generator = torch.manual_seed(seed)   # Seed generator to create the inital latent noise

    text_input = tokenizer([prompt], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

    # And the uncond. input as before:
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * num_images, padding="max_length", max_length=max_length, return_tensors="pt"
    )

    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # Prep Scheduler
    set_timesteps(scheduler, steps)

    # Prep latents
    latents = torch.randn(
    (num_images, unet.in_channels, height // 8, width // 8),
    generator=generator,
    )
    latents = latents.to(torch_device)
    latents = latents * scheduler.init_noise_sigma

    # Loop
    for i, t in tqdm(enumerate(scheduler.timesteps), total=len(scheduler.timesteps)):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        sigma = scheduler.sigmas[i]
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

        # perform CFG
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        ### ADDITIONAL GUIDANCE ###
        if i%5 == 0:
            # Requires grad on the latents
            latents = latents.detach().requires_grad_()

            # Get the predicted x0:
            # latents_x0 = latents - sigma * noise_pred
            latents_x0 = scheduler.step(noise_pred, t, latents).pred_original_sample
            scheduler._step_index = i

            # Decode to image space
            denoised_images = vae.decode((1 / 0.18215) * latents_x0).sample / 2 + 0.5 # range (0, 1)
            # Calculate loss
            loss = variant_loss(denoised_images, rgb_value) * variant_loss_scale

            # Occasionally print it out
            if i%10==0:
                print(i, 'loss:', loss.item())

            # Get gradient
            cond_grad = torch.autograd.grad(loss, latents)[0]

            # Modify the latents based on this gradient
            latents = latents.detach() - cond_grad * sigma**2

        # Now step with scheduler
        latents = scheduler.step(noise_pred, t, latents).prev_sample


    images = latents_to_pil(latents, vae)
    if save_images:
        if not os.path.exists('Results'):
            os.makedirs('Results')
        for i, image in enumerate(images):
            image.save(f'Results/{i}_({rgb_value[0]}, {rgb_value[1]}, {rgb_value[2]}).png')
    return images


if __name__ == "__main__":
    prompt = "A cute kitten"
    version = 'v2'
    # rgb_value = [249, 55, 255]
    # rgb_value = [255, 206, 51]
    rgb_value = [255, 165, 0]

    num_images = 1
    steps = 70
    height = 512
    width = 512
    guidance_scale = 7
    images = get_images_with_variant_loss(prompt, version, rgb_value, num_images, steps, height, width, guidance_scale)