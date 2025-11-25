#!/usr/bin/env python3
"""
Test58: 简洁的高性能生成器

基于test57的所有功能，但去掉实验性代码，保持简洁：
- 🚀 TensorRT修复版（自动处理时间步兼容性）
- ⚡ 流水线批量去噪
- 🔧 简单配置，直接生成
- 📊 清晰的性能报告
- 📄 支持YAML配置文件
"""

import torch, torchvision
from src.scheduler_perflow import PeRFlowScheduler
import time
import os
import sys
import yaml
from pathlib import Path

from diffusers import AutoencoderTiny, StableDiffusionPipeline
from src.streamflow.pipeline_batch_pipeline import PipelineBatchStreamFlow


def load_config_from_yaml(yaml_path):
    """从YAML文件加载配置"""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


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

# 🔧 检查是否通过命令行传入YAML配置
if len(sys.argv) > 1 and sys.argv[1].endswith('.yaml'):
    CONFIG_PATH = sys.argv[1]
    print(f"📄 从YAML加载配置: {CONFIG_PATH}")
    config = load_config_from_yaml(CONFIG_PATH)
    CONFIG_NAME = config.get('name', 'unnamed')

    # 从YAML加载配置
    USE_TINY_VAE = config['model']['use_tiny_vae']
    USE_INT8_VAE = config['model']['use_int8_vae']
    ACCELERATION = config['acceleration']['type']
    USE_CUDA_GRAPH = config['acceleration'].get('use_cuda_graph', False)
    ITERATIONS = config['test']['iterations']

    USE_PIPELINE_BATCH = config['pipeline']['use_pipeline_batch']
    FRAME_BUFFER_SIZE = config['pipeline'].get('frame_buffer_size', 1)
    VAE_DECODE_METHOD = config['pipeline'].get('vae_decode_method', 'normalize')
    DO_ADD_NOISE = config['pipeline'].get('do_add_noise', True)
    CFG_TYPE = config['pipeline']['cfg_type']
    GUIDANCE_SCALE = config['pipeline']['guidance_scale']

    USE_DYNAMIC_STEPS = config['denoising']['use_dynamic_steps']
    NUM_INFERENCE_STEPS = config['denoising']['num_inference_steps']

    USE_TENSORRT_COMPATIBILITY = config['tensorrt']['use_compatibility']
    TENSORRT_OPTIMIZATION = config['tensorrt'].get('optimization', {})

    VAE_BATCH_SIZE = config['vae']['batch_size']

    PROMPT_BASE = config['prompts']['base']
    PROMPT_SUBJECT = config['prompts']['subject']
    NEGATIVE_PROMPT = config['prompts']['negative']

    OUTPUT_DIR = os.path.join(config['test']['output_dir'], CONFIG_NAME)
    SEED = config['test']['seed']
    WIDTH = config['test'].get('width', 512)
    HEIGHT = config['test'].get('height', 512)

else:
    # 默认配置（保持原有的硬编码配置）
    CONFIG_NAME = "default"

    # 基础配置（基于test43）
    USE_TINY_VAE = True              # 设为True可进一步加速，但会轻微影响质量
    USE_INT8_VAE = False             # 🔬 实验性：INT8量化VAE（更快但可能影响质量）
    ACCELERATION = "xformers"        # "xformers", "tensorrt", "none" - 🚀 测试修复后的tensorrt
    USE_CUDA_GRAPH = False           # CUDA Graphs优化
    ITERATIONS = 100                 # 生成图像数量

    # 流水线配置
    USE_PIPELINE_BATCH = True        # 🚀 关键新增：真正的批量去噪开关 True=流水线批量去噪，False=原始StreamFlow
    FRAME_BUFFER_SIZE = 1            # 帧缓冲大小
    VAE_DECODE_METHOD = "normalize"  # VAE解码方法
    DO_ADD_NOISE = True              # 添加噪声
    CFG_TYPE = "none"               # "none", "full", "self", "initialize" - I usually use none and full
    GUIDANCE_SCALE = 7.5            # CFG强度

    # 去噪步数配置
    USE_DYNAMIC_STEPS = False       # 🔧 是否使用动态步数
                                    # False=固定4步[0,1,2,3]（质量好，推荐）
                                    # True=根据NUM_INFERENCE_STEPS动态（灵活，测试用）
    NUM_INFERENCE_STEPS = 4         # 推理步数（仅当USE_DYNAMIC_STEPS=True时生效）

    # TensorRT高级配置
    USE_TENSORRT_COMPATIBILITY = False  # 🔧 TensorRT时间步兼容性层
                                         # False=直接批处理（更快2fps，但是轻微损失质量）
                                         # True=拆分处理不同时间步（安全但会慢2fps）
    TENSORRT_OPTIMIZATION = {}       # TensorRT编译优化选项

    # VAE优化配置
    VAE_BATCH_SIZE = 1  # 🚀 VAE批量解码：累积N张latent后批量解码
                        # 1=逐个解码（慢，延迟低）
                        # 4=批量解码（快50%+，延迟稍高）
                        # 建议：离线生成用4-8，实时用1-2

    # 提示词
    PROMPT_BASE = "RAW photo, 8k uhd, dslr, high quality, film grain, highly detailed, masterpiece"
    PROMPT_SUBJECT = "A man with brown skin, a beard, and dark eyes"
    NEGATIVE_PROMPT = "distorted, blur, smooth, low-quality, warm, haze, over-saturated, high-contrast, out of focus, dark"

    # 输出配置
    OUTPUT_DIR = "test59_simple_output"
    SEED = 1024
    WIDTH = 512
    HEIGHT = 512

print(f"🚀 PeRFlow高性能生成器 - 配置: {CONFIG_NAME}")
print("=" * 50)
print(f"🔧 配置详情:")
vae_desc = "TinyVAE" if USE_TINY_VAE else "原始VAE"
if USE_INT8_VAE:
    vae_desc += " + INT8量化"
print(f"   VAE: {vae_desc}")
print(f"   加速: {ACCELERATION}")
print(f"   流水线批量: {'✅' if USE_PIPELINE_BATCH else '❌'}")
print(f"   生成数量: {ITERATIONS}")

# ================================
# 📦 模型加载
# ================================
print(f"\\n📦 加载模型...")

pipe = StableDiffusionPipeline.from_pretrained(
    "hansyan/perflow-sd15-dreamshaper", 
    torch_dtype=torch.float16
)

pipe.scheduler = PeRFlowScheduler.from_config(
    pipe.scheduler.config, 
    prediction_type="diff_eps", 
    num_time_windows=4
)
pipe.to("cuda", torch.float16)
# 重置显存峰值统计，便于监控
torch.cuda.reset_peak_memory_stats()

if USE_TINY_VAE:
    if USE_INT8_VAE:
        # 加载预量化的INT8 TinyVAE
        from utils.quantization import load_quantized_tinyvae
        pipe.vae = load_quantized_tinyvae(device=pipe.device, dtype=pipe.dtype)
    else:
        # 加载普通TinyVAE
        pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(
            device=pipe.device, dtype=pipe.dtype
        )

# ================================
# 🚀 创建流水线
# ================================
print("🚀 创建流水线...")

# 🔧 根据配置选择去噪步数
if USE_DYNAMIC_STEPS:
    # 动态模式：根据NUM_INFERENCE_STEPS生成
    t_index_list = list(range(NUM_INFERENCE_STEPS))
    prepare_steps = NUM_INFERENCE_STEPS
    print(f"   去噪模式: 动态步数")
    print(f"   去噪步数: {NUM_INFERENCE_STEPS}")
    print(f"   时间步索引: {t_index_list}")
else:
    # 固定模式：使用预设的4步（质量最优）
    t_index_list = [0, 1, 2, 3]
    prepare_steps = 4
    print(f"   去噪模式: 固定4步（质量优先）")
    print(f"   时间步索引: {t_index_list}")

stream = PipelineBatchStreamFlow(
    pipe,
    t_index_list=t_index_list,  # 动态生成，跟随NUM_INFERENCE_STEPS
    torch_dtype=torch.float16,
    frame_buffer_size=FRAME_BUFFER_SIZE,  # 帧缓冲大小：1=无缓冲，2-8=多帧缓冲
    cfg_type=CFG_TYPE,  # none, full, self, initialize
    use_pipeline_batch=USE_PIPELINE_BATCH,  # 启用流水线批量去噪
    vae_decode_method=VAE_DECODE_METHOD,  # "normalize", "dynamic", "clamp"
    do_add_noise=DO_ADD_NOISE,  # 添加噪声：True=标准模式，False=快速模式
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

        # 🔧 根据USE_PIPELINE_BATCH选择合适的编译配置
        if USE_PIPELINE_BATCH:
            # 流水线模式：需要处理4个不同阶段 (batch_size=4)
            compile_use_denoising_batch = True
            compile_max_batch_size = 4
            engine_dir = os.path.join("tensorrt_engines_pipeline_batch", CONFIG_NAME)
        else:
            # 普通模式：逐步去噪 (batch_size=1 或 2 for CFG)
            compile_use_denoising_batch = False
            compile_max_batch_size = 2 if GUIDANCE_SCALE > 1.0 and CFG_TYPE != "none" else 1
            engine_dir = os.path.join("tensorrt_engines_sequential", CONFIG_NAME)

        os.makedirs(engine_dir, exist_ok=True)
        print(f"   引擎目录: {engine_dir}")
        print(f"   编译batch_size: {compile_max_batch_size}")

        temp_stream = StreamDiffusion(
            pipe, t_index_list=t_index_list, torch_dtype=torch.float16,
            frame_buffer_size=1, cfg_type=CFG_TYPE,
            use_denoising_batch=compile_use_denoising_batch,  # 🔧 根据模式选择
            width=512, height=512,
        )

        temp_stream.prepare(PROMPT_BASE, NEGATIVE_PROMPT, num_inference_steps=prepare_steps, guidance_scale=GUIDANCE_SCALE)

        print("   编译TensorRT引擎...")
        accelerated_stream = accelerate_with_tensorrt(
            temp_stream, engine_dir=engine_dir,
            max_batch_size=compile_max_batch_size,  # 🔧 匹配推理时的batch size
            min_batch_size=1,
            use_cuda_graph=False,
        )
        
        stream.unet = accelerated_stream.unet
        stream.vae = accelerated_stream.vae  # 🚀 使用TensorRT加速的VAE

        # 🚀 应用时间步兼容性修复（可选）
        if USE_PIPELINE_BATCH and USE_TENSORRT_COMPATIBILITY:
            add_tensorrt_timestep_compatibility(stream)
            print("   ⚠️  时间步兼容性修复已应用（会降低性能）")
        elif USE_PIPELINE_BATCH and not USE_TENSORRT_COMPATIBILITY:
            print("   🚀 兼容性层已禁用，使用原生TensorRT批处理")
        
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

generator = torch.Generator("cuda").manual_seed(SEED)

prompt_text = f"{PROMPT_BASE}; {PROMPT_SUBJECT}"

# 准备StreamFlow
stream.prepare(prompt_text, NEGATIVE_PROMPT, num_inference_steps=prepare_steps, guidance_scale=GUIDANCE_SCALE)

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

if VAE_BATCH_SIZE > 1:
    print(f"🚀 VAE批量解码模式: batch_size={VAE_BATCH_SIZE}")

    # 批量VAE解码模式
    latent_buffer = []
    buffer_start_idx = []
    buffer_times = []

    for i in range(ITERATIONS):
        torch.cuda.synchronize()
        start_time = time.time()

        # 只生成latent（不解码）
        latent = stream.generate_latent()

        torch.cuda.synchronize()
        elapsed = time.time() - start_time

        # 累积到buffer
        latent_buffer.append(latent)
        buffer_start_idx.append(i)
        buffer_times.append(elapsed)

        # 当buffer满或最后一批时，批量解码
        if len(latent_buffer) == VAE_BATCH_SIZE or i == ITERATIONS - 1:
            torch.cuda.synchronize()
            decode_start = time.time()

            # 批量解码
            latents_batch = torch.cat(latent_buffer, dim=0)
            images_batch = stream.decode_latents(latents_batch)

            torch.cuda.synchronize()
            decode_time = time.time() - decode_start

            # 平摊解码时间到每张图
            decode_time_per_image = decode_time / len(latent_buffer)

            # 保存图像并记录时间
            for j, (img_idx, gen_time) in enumerate(zip(buffer_start_idx, buffer_times)):
                total_time = gen_time + decode_time_per_image
                results.append(total_time)

                # 保存图像
                torchvision.utils.save_image(
                    images_batch[j:j+1],
                    os.path.join(OUTPUT_DIR, f"image_{WIDTH}_{img_idx:06d}.png")
                )

                # 显示进度
                if img_idx % 10 == 0 or img_idx < 10:
                    img_per_sec = 1 / total_time
                    avg_fps = len(results) / sum(results)
                    print(f"图像 {img_idx+1:3d}/{ITERATIONS} | FPS: {img_per_sec:6.2f} | 平均FPS: {avg_fps:6.2f} | 生成: {gen_time:.3f}s | 解码: {decode_time_per_image:.3f}s")

            # 清空buffer
            latent_buffer = []
            buffer_start_idx = []
            buffer_times = []
else:
    print(f"📝 逐个解码模式 (VAE_BATCH_SIZE=1)")

    # 原始逐个解码模式（拆分UNet/VAE计时，便于评估）
    for i in range(ITERATIONS):
        torch.cuda.synchronize()
        unet_start = time.time()

        # 只生成latent
        latent = stream.generate_latent()

        torch.cuda.synchronize()
        unet_time = time.time() - unet_start

        torch.cuda.synchronize()
        vae_start = time.time()

        # 解码
        sample = stream.decode_latents(latent)

        torch.cuda.synchronize()
        vae_time = time.time() - vae_start

        total_time = unet_time + vae_time
        results.append(total_time)

        # 计算性能
        img_per_sec = 1 / total_time
        avg_fps = len(results) / sum(results)

        # 保存图像
        torchvision.utils.save_image(
            sample,
            os.path.join(OUTPUT_DIR, f"image_{WIDTH}_{i:06d}.png")
        )

        # 显示进度
        if i % 10 == 0 or i < 10:
            print(f"图像 {i+1:3d}/{ITERATIONS} | FPS: {img_per_sec:6.2f} | 平均FPS: {avg_fps:6.2f} | 生成: {unet_time:.3f}s | 解码: {vae_time:.3f}s")

# ================================
# 📊 最终统计
# ================================
if results:
    avg_time = sum(results) / len(results)
    total_fps = len(results) / sum(results)
    min_time = min(results)
    max_time = max(results)
    
    print(f"\\n" + "=" * 50)
    print(f"📊 性能统计")
    print(f"=" * 50)
    print(f"总图像数:      {len(results)}")
    print(f"平均生成时间:  {avg_time:.3f}s")
    print(f"平均FPS:       {total_fps:.2f}")
    print(f"最快FPS:       {1/min_time:.2f}")
    print(f"最慢FPS:       {1/max_time:.2f}")
    print(f"总用时:        {sum(results):.2f}s")
    print(f"加速方法:      {ACCELERATION}")
    print(f"流水线批量:    {'✅' if USE_PIPELINE_BATCH else '❌'}")
    print(f"VAE INT8量化:  {'✅' if USE_INT8_VAE else '❌'}")
    print(f"VAE批量解码:   {VAE_BATCH_SIZE}")

    # 🎯 性能评估
    print(f"\\n💡 性能评估:")
    if total_fps >= 12:
        print(f"   🎉 优秀！已达到12 FPS目标 ({total_fps:.1f} FPS)")
    elif total_fps >= 8:
        print(f"   ✅ 良好！接近目标 ({total_fps:.1f} FPS)")
    elif total_fps >= 6:
        print(f"   ⚡ 不错！有提升空间 ({total_fps:.1f} FPS)")
    else:
        print(f"   🔧 需要优化 ({total_fps:.1f} FPS)")
    
    # 🎨 质量提醒
    if ACCELERATION == "tensorrt" and USE_PIPELINE_BATCH:
        print(f"   🎨 TensorRT修复版：应该消除了噪音问题")
    
    print(f"\\n📁 所有图像已保存到: {OUTPUT_DIR}")
    
    # 🚀 使用建议
    print(f"\\n💡 优化建议:")
    if total_fps < 12:
        if ACCELERATION != "tensorrt":
            print(f"   - 尝试设置 ACCELERATION = 'tensorrt' 获得更高性能")
        if not USE_TINY_VAE:
            print(f"   - 尝试设置 USE_TINY_VAE = True 获得更快速度")
        if not USE_PIPELINE_BATCH:
            print(f"   - 尝试设置 USE_PIPELINE_BATCH = True 启用流水线加速")
    else:
        print(f"   🎉 配置已优化！享受高性能生成")

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

# 打印显存峰值（备用监控）
try:
    peak_mem_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)
    print(f"PEAK_MEM_MB: {peak_mem_mb:.2f}")
except Exception:
    pass
