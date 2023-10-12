import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel, EulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer, logging



def get_pipelines(version: str):
    assert version in ('v1', 'v2', 'v1-sdxl'), 'version must be one of v1, v2, v1-sdxl'

    if version == 'v1':
        vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

        # Load the tokenizer and text encoder to tokenize and encode the text.
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

        # The UNet model for generating the latents.
        unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

        # The noise scheduler
        scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

    if version == 'v2':
        vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="vae")

        # Load the tokenizer and text encoder to tokenize and encode the text.
        tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="text_encoder")

        # The UNet model for generating the latents.
        unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="unet")

        # The noise scheduler
        scheduler = EulerDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="scheduler")

    pipeline = {
        "vae": vae,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "unet": unet,
        "scheduler": scheduler
    }

    return pipeline


def load_styles(style):
#     styles = ['herge', 'nebula', 'dreamcore',\
#               'orientalist-art', 'minecraft-concept-art']
    
    torch.load(f"styles/{style}_learned_embeds.bin")
