# (c) Alex Spirin 2022-2023

import math
import random

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

def make_ctx_sched(
    total_length = 32,
    context_length = 16,
    overlap = 4,
    steps = 15,
    shuffle = True
):

    idxs=list(range(total_length))
    step = context_length-overlap
    step_ids = []

    for i in range(steps):
        """add 1st and last passes"""
        inner_ids = [idxs[:context_length]]
        if idxs[-context_length:] not in inner_ids:
            inner_ids.append(idxs[-context_length:])

        """change offset from 0 to step in increments of 2"""
        start_offset = max(-step, -((i)*2%step))
        for j in range(math.ceil(len(idxs)/(step))):

            start = j*step+start_offset
            end = j*step + context_length+start_offset
            
            if end>len(idxs):
                """drop last or overshooting ids as we already have last pass"""
                continue

            ids = idxs[start:end]
            if ids not in inner_ids and ids!=[]:
                """drop duplicates"""
                inner_ids.append(ids)
        if shuffle: random.shuffle(inner_ids)
        step_ids.append(inner_ids)
    return step_ids


