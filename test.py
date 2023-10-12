import random
import os
from run_pipelines import get_images_with_variant_loss

def test_pipelines():
    prompt = "A cute kitten"
    styles = []
    version = 'v2'

    for style in styles:
        rgb_value = random.choice([249, 55, 255], [249, 55, 255], [255, 206, 51], [255, 165, 0])
        prompt = f"{prompt} in the style of {style}"
        num_images = 1
        steps = 70
        height = 512
        width = 512
        guidance_scale = 7
        image = get_images_with_variant_loss(prompt, version, rgb_value, num_images, steps, height, width, guidance_scale)
        if not os.path.exists(f"sd_{version}_{style}"):
            os.makedirs(f"sd_{version}_{style}")
        image.save(f"sd_{version}_{style}/res_({rgb_value[0]}, {rgb_value[1]}, {rgb_value[2]}).png")
