import torch
import torchvision
import time
import os
import threading
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from src.scheduler_perflow import PeRFlowScheduler

# Create output directory
output_dir = "generated_images"
os.makedirs(output_dir, exist_ok=True)

def save_image_async(image_tensor, filename):
    """Async function to save image without blocking generation"""
    def save():
        torchvision.utils.save_image(image_tensor, filename)
    threading.Thread(target=save, daemon=True).start()

# Load PeRFlow model
pipe = StableDiffusionPipeline.from_pretrained("hansyan/perflow-sd15-dreamshaper", torch_dtype=torch.float16)
pipe.scheduler = PeRFlowScheduler.from_config(pipe.scheduler.config, prediction_type="diff_eps", num_time_windows=4)
pipe.to("cuda", torch.float16)

prompt = "1girl with dog hair, thick frame glasses"
full_prompt = "RAW photo, 8k uhd, dslr, high quality, film grain, highly detailed, masterpiece; " + prompt
neg_prompt = "distorted, blur, smooth, low-quality, warm, haze, over-saturated, high-contrast, out of focus, dark"

# Run the generation loop
frame_count = 0
start_time = time.time()

while True:
    generator = torch.Generator("cuda").manual_seed(1024 + frame_count)  # Different seed each time
    
    samples = pipe(
        prompt=[full_prompt], 
        negative_prompt=[neg_prompt],
        height=512,
        width=512,
        num_inference_steps=4, 
        guidance_scale=7.5,
        generator=generator,
        output_type='pt',
    ).images
    
    # Save image asynchronously to avoid blocking
    filename = os.path.join(output_dir, f"perflow_image_{frame_count:06d}.png")
    save_image_async(samples[0], filename)
    
    frame_count += 1
    current_time = time.time()
    elapsed_time = current_time - start_time
    fps = frame_count / elapsed_time
    print(f"FPS: {fps:.2f} - Image generated (512x512) - Saved: {filename}")
    
    # Use Ctrl+C to stop the program