import random
import os
from pipelines import load_styles
from run_pipelines import get_images_with_variant_loss
from gen_config import config_for_gen

def test_pipelines():
    prompt = "A cute kitten"
    version = 'v1'

    for style in list(config_for_gen.keys()):
        seed = config_for_gen[style]['seed']
        rgb_value = config_for_gen[style]['color_code']
        col_name = config_for_gen[style]['col_name']

        prompt = f"{prompt} in the style of text."
        num_images = 1
        steps = 50
        height = 512
        width = 512
        guidance_scale = 7
        image = get_images_with_variant_loss(prompt, version, rgb_value, num_images, steps, height, width, guidance_scale, seed=seed, save_images=False, style=style)[0]
        if not os.path.exists(f"sd_{version}"):
            os.makedirs(f"sd_{version}")
        image.save(f"sd_{version}/res_{style}_{col_name}.png")

if __name__=="__main__":
    test_pipelines()