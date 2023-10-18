# (c) Alex Spirin 2022-2023

#inspired by https://github.com/ArtVentureX/comfyui-animatediff/blob/main/animatediff/sampler.py
def inject_motion_module_to_unet(diffusion_model, mm):

    #insert motion modules depending on surrounding layers
    for i in range(12):
        a, b = divmod(i, 3)
        if type(diffusion_model.input_blocks[i][-1]).__name__ not in ["Downsample","Conv2d"]:
            diffusion_model.input_blocks[i].append(mm.down_blocks[a].motion_modules[b-1])

        if type(diffusion_model.output_blocks[i][-1]).__name__ == "Upsample":
            diffusion_model.output_blocks[i].insert(-1, mm.up_blocks[a].motion_modules[b])
        else:
            diffusion_model.output_blocks[i].append(mm.up_blocks[a].motion_modules[b])

    diffusion_model.middle_block.insert(-1, mm.mid_block.motion_modules[0])


