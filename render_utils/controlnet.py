import torch
import os
from huggingface_hub import HfApi
from pathlib import Path
from diffusers.utils import load_image
from PIL import Image
import numpy as np
from controlnet_aux import PidiNetDetector, HEDdetector
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
DEFAULT_PRPOMPT = "A sporty SUV in the 300-500k RMB price range with an aggressive exterior, powerful performance, and a tech-focused interior, luminous grille, hardcore off-road styling"

def generate_rendering(image_path, prompt=DEFAULT_PRPOMPT, seed=0):
    """
    Generate an image using ControlNet with a mask applied to remove the background.

    Args:
        image_path (str): Path to the input image.
        mask_path (str): Path to the mask image (binary mask).
        output_control_image (str): Path to save the processed control image.
        output_image (str): Path to save the final generated image.
    """
    checkpoint = "lllyasviel/control_v11p_sd15_scribble"

    # Load the input image
    control_image = load_image(image_path)
    # Apply the mask to the control image
    control_image = np.array(control_image)
    control_image = Image.fromarray(control_image)

    # Load the ControlNet model
    controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    # Generate the final image
    prompt = prompt
    generator = torch.manual_seed(seed)
    generated_image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]

    # Apply the mask to the generated image
    generated_image = np.array(generated_image)
    generated_image = Image.fromarray(generated_image)
    output_image = image_path.replace(".png", "controlnet_output.png")
    generated_image.save(output_image)
    return output_image

