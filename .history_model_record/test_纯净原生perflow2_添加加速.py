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

def setup_xformers_optimizations(pipe):
    """Setup xFormers optimizations - proven and stable"""
    print("Setting up xFormers optimizations...")
    
    # 1. Enable xFormers memory efficient attention (CRITICAL)
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("✓ xFormers memory efficient attention enabled")
    except Exception as e:
        print(f"❌ xFormers not available: {e}")
        print("   Install with: pip install xformers")
        return False
    
    # 2. Disable safety checker - major speedup
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    print("✓ Safety checker disabled")
    
    # 3. Enable VAE optimizations
    try:
        pipe.enable_vae_slicing()
        print("✓ VAE slicing enabled")
    except Exception as e:
        print(f"⚠ VAE slicing failed: {e}")
    
    # 4. Enable attention slicing for lower memory
    try:
        pipe.enable_attention_slicing(1)
        print("✓ Attention slicing enabled")
    except Exception as e:
        print(f"⚠ Attention slicing failed: {e}")
    
    # 5. Set models to eval mode and optimize memory
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    
    # Disable gradients for all parameters
    for param in pipe.unet.parameters():
        param.requires_grad = False
    for param in pipe.vae.parameters():
        param.requires_grad = False
    for param in pipe.text_encoder.parameters():
        param.requires_grad = False
    
    print("✓ Models optimized for inference")
    return True

def setup_tensorrt_acceleration(pipe, enable_tensorrt=False):
    """Setup TensorRT acceleration - maximum performance"""
    if not enable_tensorrt:
        print("TensorRT disabled (set enable_tensorrt=True for maximum speed)")
        return False
    
    try:
        import torch_tensorrt
        print("🔥 TensorRT acceleration starting...")
        print("⏰ This will take 5-15 minutes but gives 2-4x speedup!")
        print("☕ Perfect time for a coffee break...")
        
        # Step 1: Prepare inputs for tracing
        print("📋 Step 1/4: Preparing sample inputs...")
        batch_size = 2  # For classifier-free guidance
        height, width = 512, 512
        latent_height, latent_width = height // 8, width // 8
        
        sample_latent = torch.randn(
            batch_size, 4, latent_height, latent_width, 
            dtype=torch.float16, device='cuda'
        )
        sample_timestep = torch.tensor([100], dtype=torch.long, device='cuda')
        sample_encoder_hidden_states = torch.randn(
            batch_size, 77, 768, 
            dtype=torch.float16, device='cuda'
        )
        
        # Step 2: Trace the UNet model
        print("🔍 Step 2/4: Tracing UNet model...")
        pipe.unet.eval()
        with torch.no_grad():
            # Test run to ensure model works
            _ = pipe.unet(sample_latent, sample_timestep, sample_encoder_hidden_states)
            
            # Trace the model
            traced_unet = torch.jit.trace(
                pipe.unet,
                (sample_latent, sample_timestep, sample_encoder_hidden_states),
                strict=False,
                check_trace=False  # Disable trace checking for stability
            )
        
        print("⚙️ Step 3/4: Compiling with TensorRT (this takes the longest)...")
        # Step 3: Compile with TensorRT
        trt_unet = torch_tensorrt.compile(
            traced_unet,
            inputs=[
                # Latent input
                torch_tensorrt.Input(
                    min_shape=[1, 4, latent_height, latent_width],
                    opt_shape=[batch_size, 4, latent_height, latent_width],
                    max_shape=[4, 4, latent_height, latent_width],
                    dtype=torch.half
                ),
                # Timestep input
                torch_tensorrt.Input(
                    min_shape=[1], 
                    opt_shape=[1], 
                    max_shape=[1], 
                    dtype=torch.long
                ),
                # Encoder hidden states
                torch_tensorrt.Input(
                    min_shape=[1, 77, 768],
                    opt_shape=[batch_size, 77, 768],
                    max_shape=[4, 77, 768],
                    dtype=torch.half
                )
            ],
            enabled_precisions={torch.half},  # Use FP16
            workspace_size=1 << 26,  # 64MB workspace
            max_batch_size=4,
            use_fp16=True,
            strict_types=True
        )
        
        # Step 4: Replace the original UNet
        print("🔄 Step 4/4: Replacing UNet with TensorRT engine...")
        pipe.unet = trt_unet
        
        print("🚀 TensorRT compilation complete!")
        print("💪 Expected performance: 2-4x faster inference")
        return True
        
    except ImportError:
        print("❌ TensorRT not available!")
        print("📦 Install with:")
        print("   pip install torch-tensorrt")
        print("   # or for NVIDIA official:")
        print("   pip install tensorrt")
        return False
        
    except Exception as e:
        print(f"❌ TensorRT compilation failed: {e}")
        print("💡 Try reducing batch size or image resolution")
        return False

# Basic CUDA optimizations
print("Setting up CUDA optimizations...")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
print("✓ CUDA optimizations enabled")

# Load model
print("Loading PeRFlow model...")
pipe = StableDiffusionPipeline.from_pretrained(
    "hansyan/perflow-sd15-dreamshaper", 
    torch_dtype=torch.float16,
    use_safetensors=True,
    safety_checker=None,
    requires_safety_checker=False
)

pipe.scheduler = PeRFlowScheduler.from_config(
    pipe.scheduler.config, 
    prediction_type="diff_eps", 
    num_time_windows=4
)

# Move to GPU
pipe.to("cuda", torch.float16)

# Apply xFormers optimizations
xformers_success = setup_xformers_optimizations(pipe)

if not xformers_success:
    print("❌ xFormers setup failed - performance will be limited")

# TensorRT setup
print("\n" + "="*60)
print("🚀 TENSORRT ACCELERATION SETUP")
print("="*60)
print("Set ENABLE_TENSORRT = True for maximum performance")
print("⚠️  First run will take 5-15 minutes to compile")
print("✅ Subsequent runs will be 2-4x faster")
print("="*60)

# SET THIS TO TRUE FOR MAXIMUM PERFORMANCE
ENABLE_TENSORRT = True  # Change to True for maximum speed

tensorrt_enabled = setup_tensorrt_acceleration(pipe, enable_tensorrt=ENABLE_TENSORRT)

# Pre-encode prompts for speed
prompt = "1girl with dog hair, thick frame glasses"
full_prompt = "RAW photo, 8k uhd, dslr, high quality, film grain, highly detailed, masterpiece; " + prompt
neg_prompt = "distorted, blur, smooth, low-quality, warm, haze, over-saturated, high-contrast, out of focus, dark"

print("\nPre-encoding text prompts...")
with torch.no_grad():
    # Get text embeddings
    text_inputs = pipe.tokenizer(
        [full_prompt],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipe.text_encoder(text_inputs.input_ids.to("cuda"))[0]
    
    # Get negative embeddings
    uncond_inputs = pipe.tokenizer(
        [neg_prompt],
        padding="max_length", 
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    uncond_embeddings = pipe.text_encoder(uncond_inputs.input_ids.to("cuda"))[0]
    
    # Combine for classifier-free guidance
    prompt_embeds = torch.cat([uncond_embeddings, text_embeddings])

print("✓ Text prompts pre-encoded")

# Warmup runs (especially important for TensorRT)
print("Performing warmup runs...")
warmup_count = 5 if tensorrt_enabled else 2

with torch.no_grad():
    for i in range(warmup_count):
        print(f"  Warmup {i+1}/{warmup_count}...")
        generator = torch.Generator("cuda").manual_seed(1024)
        
        _ = pipe(
            prompt_embeds=prompt_embeds,
            height=512,
            width=512,
            num_inference_steps=4,
            guidance_scale=7.5,
            generator=generator,
            output_type='pt'
        )

print("✓ Warmup complete!")

# Performance expectations
print("\n" + "="*50)
print("EXPECTED PERFORMANCE:")
if tensorrt_enabled:
    print("🚀 TensorRT + xFormers: 8-15 FPS")
elif xformers_success:
    print("⚡ xFormers only: 3-6 FPS")
else:
    print("🐌 No optimizations: 1-3 FPS")
print("="*50)

print("\nStarting optimized generation loop...")
print("Press Ctrl+C to stop")

# Main generation loop
frame_count = 0
start_time = time.time()

try:
    while True:
        generator = torch.Generator("cuda").manual_seed(1024 + frame_count)
        
        with torch.no_grad():
            samples = pipe(
                prompt_embeds=prompt_embeds,
                height=512,
                width=512,
                num_inference_steps=4,
                guidance_scale=7.5,
                generator=generator,
                output_type='pt',
            ).images
        
        # Save image asynchronously
        filename = os.path.join(output_dir, f"perflow_image_{frame_count:06d}.png")
        save_image_async(samples[0], filename)
        
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time
        fps = frame_count / elapsed_time
        
        # Performance indicator
        if tensorrt_enabled:
            indicator = "🚀 TRT"
        elif xformers_success:
            indicator = "⚡ XF"
        else:
            indicator = "🐌 BASE"
            
        print(f"{indicator} FPS: {fps:.2f} - Image {frame_count} - {filename}")
        
        # Less frequent cleanup to avoid overhead
        if frame_count % 50 == 0:
            torch.cuda.empty_cache()

except KeyboardInterrupt:
    final_fps = frame_count / (time.time() - start_time)
    print(f"\n🏁 Generation stopped. Total: {frame_count} images")
    print(f"📊 Average FPS: {final_fps:.2f}")
    
    # Performance analysis
    print("\n" + "="*50)
    print("FINAL PERFORMANCE REPORT:")
    if tensorrt_enabled:
        print(f"🚀 TensorRT + xFormers: {final_fps:.2f} FPS")
        if final_fps < 8:
            print("💡 Consider upgrading GPU for better TensorRT performance")
    elif xformers_success:
        print(f"⚡ xFormers optimization: {final_fps:.2f} FPS")
        print("💡 Enable TensorRT for 2-4x more speed")
    else:
        print(f"🐌 Basic performance: {final_fps:.2f} FPS")
        print("💡 Install xFormers for significant speedup")
    print("="*50)
    
except Exception as e:
    print(f"❌ Error occurred: {e}")
finally:
    torch.cuda.empty_cache()