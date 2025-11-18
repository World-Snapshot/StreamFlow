#!/usr/bin/env python3
"""
Test58: 简洁的高性能生成器

基于test57的所有功能，但去掉实验性代码，保持简洁：
- 🚀 TensorRT修复版（自动处理时间步兼容性）
- ⚡ 流水线批量去噪
- 🔧 简单配置，直接生成
- 📊 清晰的性能报告
"""

import torch, torchvision
from src.scheduler_perflow import PeRFlowScheduler
import time
import os

from diffusers import AutoencoderTiny, StableDiffusionPipeline
from src.streamflow.pipeline_batch_pipeline import PipelineBatchStreamFlow


def add_tensorrt_timestep_compatibility(stream):
    """
    🚀 关键修复：添加TensorRT时间步兼容性
    
    问题：TensorRT不能处理不同时间步 [1000, 750, 500, 250]
    解决：在检测到不同时间步时，分解为单独调用
    """
    if not hasattr(stream.unet, 'forward'):
        print("⚠️  UNet没有forward方法，跳过兼容性修复")
        return None
    
    original_forward = stream.unet.forward
    
    def tensorrt_compatible_forward(sample, timestep, encoder_hidden_states, **kwargs):
        """
        TensorRT兼容的forward方法
        自动处理不同时间步问题
        """
        # 检查是否有不同时间步
        if isinstance(timestep, torch.Tensor) and timestep.dim() > 0 and len(timestep) > 1:
            unique_timesteps = torch.unique(timestep)
            if len(unique_timesteps) > 1:
                # 🎯 检测到不同时间步，分解处理
                batch_size = sample.shape[0]
                results = []
                for i in range(batch_size):
                    single_sample = sample[i:i+1]
                    single_timestep = timestep[i:i+1]
                    single_encoder_states = encoder_hidden_states[i:i+1]
                    
                    single_result = original_forward(
                        single_sample, single_timestep, single_encoder_states, **kwargs
                    )
                    results.append(single_result)
                
                # 重新组装结果
                if isinstance(results[0], tuple):
                    assembled = []
                    for i in range(len(results[0])):
                        assembled.append(torch.cat([r[i] for r in results], dim=0))
                    return tuple(assembled)
                else:
                    return torch.cat(results, dim=0)
        
        # 相同时间步或单个时间步，直接调用TensorRT
        return original_forward(sample, timestep, encoder_hidden_states, **kwargs)
    
    # 应用兼容性修复
    stream.unet.forward = tensorrt_compatible_forward
    print("✅ TensorRT时间步兼容性已添加")
    return original_forward


# ================================
# 🔧 配置区域 - 所有设置都在这里
# ================================

# 基础配置（基于test43）
USE_TINY_VAE = True              # 设为True可进一步加速，但会轻微影响质量
ACCELERATION = "xformers"        # "xformers", "tensorrt", "none" - 🚀 测试修复后的tensorrt
ITERATIONS = 100                 # 生成图像数量

# 流水线配置
USE_PIPELINE_BATCH = True        # 🚀 关键新增：真正的批量去噪开关 True=流水线批量去噪，False=原始StreamFlow
CFG_TYPE = "none"               # "none", "full", "self", "initialize" - I usually use none and full
GUIDANCE_SCALE = 7.5            # CFG强度
NUM_INFERENCE_STEPS = 4         # 推理步数

# 提示词
PROMPT_BASE = "RAW photo, 8k uhd, dslr, high quality, film grain, highly detailed, masterpiece"
PROMPT_SUBJECT = "A man with brown skin, a beard, and dark eyes"
NEGATIVE_PROMPT = "distorted, blur, smooth, low-quality, warm, haze, over-saturated, high-contrast, out of focus, dark"

# 输出配置
OUTPUT_DIR = "Multigpu"
SEED = 1024

print("🚀 PeRFlow高性能multigpu生成器")
print("=" * 50)
print(f"🔧 配置:")
print(f"   VAE: {'TinyVAE' if USE_TINY_VAE else '原始VAE'}")
print(f"   加速: {ACCELERATION}")
print(f"   流水线批量: {'✅' if USE_PIPELINE_BATCH else '❌'}")
print(f"   生成数量: {ITERATIONS}")

# ================================
# 📦 模型加载
# ================================
print(f"\\n📦 加载模型...")

from accelerate import PartialState
distributed_state = PartialState()
with distributed_state.main_process_first():
    pipe = StableDiffusionPipeline.from_pretrained(
        "hansyan/perflow-sd15-dreamshaper", 
        torch_dtype=torch.float16
    )

    pipe.scheduler = PeRFlowScheduler.from_config(
        pipe.scheduler.config, 
        prediction_type="diff_eps", 
        num_time_windows=4
    )


pipe.to(distributed_state.device)

if USE_TINY_VAE:
    with distributed_state.main_process_first():
        vae = AutoencoderTiny.from_pretrained("madebyollin/taesd")
   
    vae.to(
    device=pipe.device, dtype=pipe.dtype)
    pipe.vae = vae
    del vae
# ================================
# 🚀 创建流水线
# ================================
print("🚀 创建流水线...")

stream = PipelineBatchStreamFlow(
    pipe,
    t_index_list=[0, 1, 2, 3],  # PeRFlow的4个时间步 [0, 1, 2, 3]，使用49这种时间步似乎可以提升质量：[0, 12, 24, 49]；但是0 1 2 3对于cfg为none时效果非常好
    torch_dtype=torch.float16,
    frame_buffer_size=1,  # 帧缓冲大小：1=无缓冲，2-8=多帧缓冲
    cfg_type=CFG_TYPE,  # none, full, self, initialize
    use_pipeline_batch=USE_PIPELINE_BATCH,  # 启用流水线批量去噪
    vae_decode_method="normalize",  # "normalize" 或 "dynamic" - 优化的解码方法
    do_add_noise=True,  # 添加噪声：True=标准模式，False=快速模式
)

# ================================
# ⚡ 加速设置
# ================================
if ACCELERATION == "xformers":
    pipe.enable_xformers_memory_efficient_attention()
    print("⚡ xformers加速已启用")
elif ACCELERATION == "tensorrt":
    print("🚀 启用TensorRT加速...")
    try:
        from src.streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt
        from src.streamdiffusion.pipeline import StreamDiffusion
        
        # 创建引擎目录
        engine_dir = "tensorrt_engines_test59_fixed"
        os.makedirs(engine_dir, exist_ok=True)
        
        temp_stream = StreamDiffusion(
            pipe, t_index_list=[0, 1, 2, 3], torch_dtype=torch.float16,
            frame_buffer_size=1, cfg_type=CFG_TYPE, use_denoising_batch=True,
            width=512, height=512,
        )
        
        temp_stream.prepare(PROMPT_BASE, NEGATIVE_PROMPT, num_inference_steps=NUM_INFERENCE_STEPS, guidance_scale=GUIDANCE_SCALE)
        
        print("   编译TensorRT引擎...")
        accelerated_stream = accelerate_with_tensorrt(
            temp_stream, engine_dir=engine_dir, max_batch_size=4, min_batch_size=1, use_cuda_graph=False,
        )
        
        stream.unet = accelerated_stream.unet
        
        # 🚀 应用时间步兼容性修复
        if USE_PIPELINE_BATCH:
            add_tensorrt_timestep_compatibility(stream)
            print("   时间步兼容性修复已应用")
        
        del temp_stream, accelerated_stream
        torch.cuda.empty_cache()
        print("✅ TensorRT加速已启用")
        
    except Exception as e:
        print(f"❌ TensorRT失败，回退到无加速: {e}")
        ACCELERATION = "none"

# ================================
# 🔥 预热
# ================================
# 预热阶段
print(f"\\n🔥 预热中...")

with distributed_state.split_between_processes(list(range(10))) as local_idxs:
    generator = torch.Generator(distributed_state.device).manual_seed(SEED)

    prompt_text = f"{PROMPT_BASE}; {PROMPT_SUBJECT}"
    # 准备StreamFlow
    stream.prepare(prompt_text, NEGATIVE_PROMPT, num_inference_steps=NUM_INFERENCE_STEPS, guidance_scale=GUIDANCE_SCALE)

    # 预热生成
    for i in range(10):
        _ = stream.txt2img()
        if i == 0:
            print("✅ 预热完成")

# ================================
# 🎨 图像生成
# ================================
print(f"\\n🎨 开始生成 {ITERATIONS} 张图像...")
print(f"📝 提示词: {prompt_text}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
results = []
torch.cuda.synchronize()
start_time = time.time()
with distributed_state.split_between_processes(list(range(ITERATIONS))) as local_idxs:
    
    print(f"{len(local_idxs)} image per gpu")
    print(f'using {ITERATIONS//len(local_idxs)} GPUs')

    # 生成图像
    for idx in local_idxs:
        sample = stream.txt2img()
        

        torchvision.utils.save_image(
            sample,
            os.path.join(OUTPUT_DIR, f"image_{idx:06d}.png")
        )
torch.cuda.synchronize()
elapsed = time.time() - start_time    

# ================================
# 📊 最终统计
# ================================

print(f"\\n" + "=" * 50)
print(f"📊 性能统计")
print(f"=" * 50)
print(f"总图像数:      {ITERATIONS}")
print(f"平均生成时间:  {elapsed/ITERATIONS:.3f}s")
print(f"总用时:        {elapsed:.2f}s")
print(f"FPS:        {ITERATIONS/elapsed:.2f}")
print(f"加速方法:      {ACCELERATION}")
print(f"流水线批量:    {'✅' if USE_PIPELINE_BATCH else '❌'}")
    

print(f"\\n🎉 测试完成！")
print(f"🔍 请检查图像质量和连续性")

if ACCELERATION == "tensorrt" and USE_PIPELINE_BATCH:
    print(f"\\n💡 TensorRT修复版使用指南:")
    print(f"   1. 设置 ACCELERATION = 'tensorrt'")
    print(f"   2. 首次运行会自动编译引擎")
    print(f"   3. 后续运行直接加载引擎")
    print(f"   4. 自动处理时间步兼容性")
    print(f"   5. 享受无噪音的TensorRT加速！")

print(f"\\n🎉 生成完成！")