import random
import os
from pipelines import load_styles
from run_pipelines import get_images_with_variant_loss

def test_pipelines():
    prompt = "A cute kitten"
    styles = ['anders-zorn', 'nebula', 'dreamcore', \
              'manga', 'minecraft-concept-art']
    version = 'v1'

    for style in styles:
        seed = random.seed(random.random())
        rgb_value = random.choice([[249, 55, 255], [249, 55, 255], [255, 206, 51], [255, 165, 0]])
        prompt = f"{prompt} in the style of text."
        num_images = 1
        steps = 50
        height = 512
        width = 512
        guidance_scale = 7
        image = get_images_with_variant_loss(prompt, version, rgb_value, num_images, steps, height, width, guidance_scale, seed=32, save_images=False, style=style)[0]
        if not os.path.exists(f"sd_{version}_{style}"):
            os.makedirs(f"sd_{version}_{style}")
        image.save(f"sd_{version}_{style}/res_({rgb_value[0]}, {rgb_value[1]}, {rgb_value[2]}).png")

if __name__=="__main__":
    test_pipelines()