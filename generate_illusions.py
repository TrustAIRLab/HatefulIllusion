import torch
from PIL import Image
import random
from diffusers import (
    DiffusionPipeline,
    AutoencoderKL,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    StableDiffusionLatentUpscalePipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler
)
import time
import os, sys
from transformers import CLIPImageProcessor
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from pathlib import Path
from utils import *
import fire

BASE_MODEL = "SG161222/Realistic_Vision_V5.1_noVAE"

# Initialize both pipelines
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
controlnet = ControlNetModel.from_pretrained("monster-labs/control_v1p_sd15_qrcode_monster", torch_dtype=torch.float16)

# Initialize the safety checker conditionally
SAFETY_CHECKER_ENABLED = os.environ.get("SAFETY_CHECKER", "0") == "1"
SAFETY_CHECKER_ENABLED = False
safety_checker = None
feature_extractor = None

if SAFETY_CHECKER_ENABLED:
    safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker").to("cuda").to(dtype=torch.float16)
    feature_extractor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

main_pipe = StableDiffusionControlNetPipeline.from_pretrained(
    BASE_MODEL,
    controlnet=controlnet,
    vae=vae,
    safety_checker=safety_checker,
    feature_extractor=feature_extractor,
    torch_dtype=torch.float16,
).to("cuda")

image_pipe = StableDiffusionControlNetImg2ImgPipeline(**main_pipe.components)

# Sampler map
SAMPLER_MAP = {
    "DPM++ Karras SDE": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras=True, algorithm_type="sde-dpmsolver++"),
    "Euler": lambda config: EulerDiscreteScheduler.from_config(config),
}

def center_crop_resize(img, output_size=(512, 512)):
    width, height = img.size

    # Calculate dimensions to crop to the center
    new_dimension = min(width, height)
    left = (width - new_dimension)/2
    top = (height - new_dimension)/2
    right = (width + new_dimension)/2
    bottom = (height + new_dimension)/2

    # Crop and resize
    img = img.crop((left, top, right, bottom))
    img = img.resize(output_size)

    return img

def common_upscale(samples, width, height, upscale_method, crop=False):
    if crop == "center":
        old_width = samples.shape[3]
        old_height = samples.shape[2]
        old_aspect = old_width / old_height
        new_aspect = width / height
        x = 0
        y = 0
        if old_aspect > new_aspect:
            x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
        elif old_aspect < new_aspect:
            y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
        s = samples[:,:,y:old_height-y,x:old_width-x]
    else:
        s = samples

    return torch.nn.functional.interpolate(s, size=(height, width), mode=upscale_method)

def upscale(samples, upscale_method, scale_by):
    #s = samples.copy()
    width = round(samples["images"].shape[3] * scale_by)
    height = round(samples["images"].shape[2] * scale_by)
    s = common_upscale(samples["images"], width, height, upscale_method, "disabled")
    return (s)
    
def convert_to_pil(base64_image):
    pil_image = Image.open(base64_image)
    return pil_image

def inference(
    control_image: Image.Image,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float = 8.0,
    controlnet_conditioning_scale: float = 1,
    control_guidance_start: float = 0,    
    control_guidance_end: float = 1,
    upscaler_strength: float = 0.5,
    seed: int = 2025,
    sampler = "DPM++ Karras SDE",
):  
    # Generate the initial image
    #init_image = init_pipe(prompt).images[0]

    # Rest of your existing code
    control_image_small = center_crop_resize(control_image)
    control_image_large = center_crop_resize(control_image, (1024, 1024))

    main_pipe.scheduler = SAMPLER_MAP[sampler](main_pipe.scheduler.config)
    my_seed = random.randint(0, 2**32 - 1) if seed == -1 else seed
    generator = torch.Generator(device="cuda").manual_seed(my_seed)
    
    out = main_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=control_image_small,
        guidance_scale=float(guidance_scale),
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),
        generator=generator,
        control_guidance_start=float(control_guidance_start),
        control_guidance_end=float(control_guidance_end),
        num_inference_steps=50,
        output_type="latent"
    )
    
    upscaled_latents = upscale(out, "nearest-exact", 2)
    
    out_image = image_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        control_image=control_image_large,        
        image=upscaled_latents,
        guidance_scale=float(guidance_scale),
        generator=generator,
        num_inference_steps=50,
        strength=upscaler_strength,
        control_guidance_start=float(control_guidance_start),
        control_guidance_end=float(control_guidance_end),
        controlnet_conditioning_scale=float(controlnet_conditioning_scale)
    )
    
    return out_image["images"]

def main(target: str="hate_slangs",
         controlnet_scale: float=1.0,
         image_root: str="data",
         save_root: str="data/generations"):
    
    hate_slangs = fetch_hate_slangs()
    hate_symbols = fetch_hate_symbols()

    save_root =  f"{save_root}/{target}"

    prompts = open(os.path.join(image_root, "prompts/descriptive_30_prompts.txt"), "r").read().splitlines()
    negative_prompt = "low quality"
    
    # setup the controlnet scale
    # The seleciton is done through simple experiments to balance the number of high-visibility and low-visibility illusions
    controlnet_scale_map = {
        "hate_slangs": 0.9,
        "hate_symbols": 1.2,
        "digits": 0.9,
    }
    
    controlnet_scale = controlnet_scale_map[target]
    
    if target == "hate_slangs":
        
        for slang in hate_slangs:

            image_fname = os.path.join(image_root, "messages/hate_slangs", f"{slang}.png")
            image = Image.open(image_fname)

            for prompt_idx, prompt in enumerate(prompts):

                images = inference(control_image=image,
                                   prompt=prompt,
                                   negative_prompt=negative_prompt,
                                   controlnet_conditioning_scale=controlnet_scale
                                   )
                
                Path(f"{save_root}/{slang}").mkdir(exist_ok=True, parents=True)
                images[0].save(f"{save_root}/{slang}/prompt_{prompt_idx}.png")

    elif target == "hate_symbols":
        image_root  = os.path.join(image_root,  "messages/hate_symbols")
        image_fnames = os.listdir(os.path.join(image_root))
        
        for idx, image_fname in enumerate(image_fnames):
            hate_symbol = image_fname.replace(".png", "")
            
            original_image_file = os.path.join(image_root, image_fname)
            image = Image.open(original_image_file).convert("RGB")

            for prompt_idx, prompt in enumerate(prompts):

                images = inference(control_image=image,
                                prompt=prompt,
                                negative_prompt=negative_prompt,
                                controlnet_conditioning_scale=controlnet_scale
                                )
                
                Path(f"{save_root}/{hate_symbol}").mkdir(exist_ok=True, parents=True)
                images[0].save(f"{save_root}/{hate_symbol}/prompt_{prompt_idx}.png")
                    
    elif target == "digits":
        image_root  = os.path.join(image_root,  "messages/digits")
        
        digits = range(10)
        
        for digit in digits:
            image = Image.open(os.path.join(image_root, f"{digit}.png"))
            
            for prompt_idx, prompt in enumerate(prompts):

                images = inference(control_image=image,
                                prompt=prompt,
                                negative_prompt=negative_prompt,
                                controlnet_conditioning_scale=controlnet_scale
                                )
                
                Path(f"{save_root}/{digit}").mkdir(exist_ok=True, parents=True)
                images[0].save(f"{save_root}/{digit}/prompt_{prompt_idx}.png")
                
    elif "object" in target:
        
        image_root = os.path.join(image_root, "messages", target)
        
        image_fnames = os.listdir(image_root)
        
        for image_fname in image_fnames:
            image = Image.open(os.path.join(image_root, image_fname))
            
            for prompt_idx, prompt in enumerate(prompts):
    
                images = inference(control_image=image,
                                prompt=prompt,
                                negative_prompt=negative_prompt,
                                controlnet_conditioning_scale=controlnet_scale
                                )
                
                Path(f"{save_root}/{image_fname.replace('.png', '')}").mkdir(exist_ok=True, parents=True)
                images[0].save(f"{save_root}/{image_fname.replace('.png', '')}/prompt_{prompt_idx}.png")
                
    elif "blank" in target:
        
        width, height = 512, 512
        background_color = (255, 255, 255)  # white background; you can change this as needed

        # Create a new blank image with the specified size and color
        blank_image = Image.new("RGB", (width, height), background_color)

        for prompt_idx, prompt in enumerate(prompts):

            images = inference(control_image=blank_image,
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            controlnet_conditioning_scale=controlnet_scale
                            )
            
            Path(f"{save_root}").mkdir(exist_ok=True, parents=True)
            images[0].save(f"{save_root}/prompt_{prompt_idx}.png")
                   
if __name__ == "__main__":
    
    fire.Fire(main)
