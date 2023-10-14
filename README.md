## Stable Diffusion Styles with Textual Inversion and Custom Coloring with Variant Loss

### Objective
To implement `pipeline for textual inversion` in SD 1 and SD 2 <br>
To add a `variant loss` that can bias the latent to a shade

### How textual inversion works?
In textual inversion training, we supply some images and train the CLIP embedding of a `predefined token` to be similar to the image embeddings of the supplied images. 
Thus the `predefined token` now has the embedding relevant to the images. Now during inference we `define our token which we want to be similar to the embedding of the given styled images`.
We replace the embedding of that token with the embedding of `predefined token` to get images similar to the styled ones.
<br>
<br>
Basically, we are ensuring or token to `mimic` the behavior of the predefined token.

### How custom coloring works?
At each scheduler step, we can get a latent from the predicted noise. I ensure `each channel of this latent is inclined to a shade`to get the final color. These latents are normalized, 
so I have define the colors in a normalized way only. We find the `error between current channel value and what I want`, based on that we define a loss and it is minimized with some weightage.

### Results on SD 1

<img width="644" alt="Screen Shot 2023-10-14 at 8 09 23 AM" src="https://github.com/kurchi1205/SD-Basics/assets/40196782/1698e104-6d0f-4a3b-93b5-cc81d0122992">


### Steps to Reproduce
- install all the requirements
- run `python test.py` (version is set at `v1` for SD-1)
- results will be in the folder SD-1
- For any other style add your embeds in the style folder as `{style}_learned_embed.bin` and add that style in the `gen_config.py` with your color and seed.
- Support for SD 2 is coming soon !!