import torch, torchvision
from src.scheduler_perflow import PeRFlowScheduler
import time
import os

from diffusers import AutoencoderTiny, StableDiffusionPipeline

# 使用我们优化的StreamFlow而不是原始StreamDiffusion
from src.streamflow import StreamFlow
from src.streamflow.image_utils import postprocess_image

def run_perflow_frame_buffer_test():
    """
    使用StreamFlow进行PeRFlow帧缓冲测试
    测试不同frame_buffer_size对性能和质量的影响
    """
    print("🚀 PeRFlow + StreamFlow 帧缓冲测试")
    print("=" * 60)
    
    # 配置选项
    USE_TINY_VAE = False  # 设为True可进一步加速，但会轻微影响质量
    VAE_DECODE_METHOD = "normalize"  # "normalize" 或 "dynamic"
    ACCELERATION = "xformers"  # xformers, tensorrt, none
    
    # 新增：帧缓冲和相关设置
    FRAME_BUFFER_SIZE = 1  # 帧缓冲大小：1=无缓冲，2-8=多帧缓冲
    USE_DENOISING_BATCH = True  # 批量降噪：StreamFlow中默认优化
    DO_ADD_NOISE = True  # 添加噪声：True=标准模式，False=快速模式
    
    print(f"🔧 配置:")
    print(f"   VAE类型: {'TinyVAE' if USE_TINY_VAE else '原始VAE'}")
    print(f"   解码方法: {VAE_DECODE_METHOD}")
    print(f"   加速方法: {ACCELERATION}")
    print(f"   帧缓冲大小: {FRAME_BUFFER_SIZE} ({'无缓冲' if FRAME_BUFFER_SIZE == 1 else f'{FRAME_BUFFER_SIZE}帧缓冲'})")
    print(f"   批量降噪: {'启用(StreamFlow优化)' if USE_DENOISING_BATCH else '禁用'}")
    print(f"   添加噪声: {'启用' if DO_ADD_NOISE else '禁用'}")
    
    # 帧缓冲说明
    if FRAME_BUFFER_SIZE > 1:
        print(f"   💡 帧缓冲效果: 可能提高流式生成的流畅度和一致性")
    else:
        print(f"   💡 无帧缓冲: 每帧独立生成，延迟最低")
    
    # 加载PeRFlow模型
    print(f"\n📦 加载PeRFlow模型...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "hansyan/perflow-sd15-dreamshaper", 
        torch_dtype=torch.float16
    )
    
    # 设置PeRFlow调度器
    pipe.scheduler = PeRFlowScheduler.from_config(
        pipe.scheduler.config, 
        prediction_type="diff_eps", 
        num_time_windows=4
    )
    pipe.to("cuda", torch.float16)
    
    # 可选：使用TinyVAE加速
    if USE_TINY_VAE:
        print("🔄 加载TinyVAE...")
        pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(
            device=pipe.device, dtype=pipe.dtype
        )
    
    # 创建StreamFlow（我们的优化版本）- 启用帧缓冲
    stream = StreamFlow(
        pipe,
        t_index_list=[0, 12, 24, 49],  # PeRFlow的4个时间步 [0, 1, 2, 3]，使用49这种时间步似乎可以提升质量：[0, 12, 24, 49]
        torch_dtype=torch.float16,
        frame_buffer_size=FRAME_BUFFER_SIZE,  # 关键：启用帧缓冲
        cfg_type="full",
        use_original_scheduler=True,  # 使用PeRFlow原生调度器
        vae_decode_method=VAE_DECODE_METHOD,  # 优化的解码方法
        do_add_noise=DO_ADD_NOISE,  # 是否添加噪声
    )
    
    # 启用加速
    if ACCELERATION == "xformers":
        pipe.enable_xformers_memory_efficient_attention()
        print("⚡ 启用xformers加速")
    
    # 测试提示词
    prompts_list = ["A man with brown skin, a beard, and dark eyes"]
    prompt = "RAW photo, 8k uhd, dslr, high quality, film grain, highly detailed, masterpiece; " + prompts_list[0]
    neg_prompt = "distorted, blur, smooth, low-quality, warm, haze, over-saturated, high-contrast, out of focus, dark"
    
    print(f"\n📝 提示词: {prompt}")
    
    # 预热阶段
    print(f"\n🔥 预热中...")
    generator = torch.Generator("cuda").manual_seed(1024)
    
    # 准备StreamFlow
    stream.prepare(prompt, neg_prompt, num_inference_steps=4, guidance_scale=7.5)
    
    # 预热生成
    for i in range(10):
        _ = stream.txt2img()
        if i % 3 == 0:
            print(f"   预热 {i+1}/10")
    
    print("✅ 预热完成")
    
    # 帧缓冲测试设置
    output_dir = f"test22_frame_buffer_{FRAME_BUFFER_SIZE}_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 测试不同的生成模式
    test_modes = [
        {"name": "txt2img", "iterations": 50, "description": "文本到图像生成"},
        {"name": "sequence", "iterations": 30, "description": "序列生成（模拟视频帧）"},
    ]
    
    all_results = {}
    
    for mode_config in test_modes:
        mode_name = mode_config["name"]
        iterations = mode_config["iterations"]
        description = mode_config["description"]
        
        print(f"\n🎨 开始{description}测试 ({mode_name})...")
        print(f"💾 输出目录: {output_dir}")
        print(f"🔢 生成次数: {iterations}")
        
        results = []
        quality_samples = []  # 保存一些样本用于质量检查
        
        # 生成循环
        for i in range(iterations):
            torch.cuda.synchronize()
            start_time = time.time()
            
            if mode_name == "txt2img":
                # 标准txt2img生成
                sample = stream.txt2img()
            elif mode_name == "sequence":
                # 序列生成：使用前一帧作为输入（模拟视频）
                if i == 0:
                    # 第一帧从txt2img开始
                    sample = stream.txt2img()
                else:
                    # 后续帧使用img2img（如果支持的话）
                    sample = stream.txt2img()  # 暂时还是用txt2img
            
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            results.append(elapsed)
            
            # 计算FPS
            img_per_sec = 1 / elapsed
            avg_fps = len(results) / sum(results)
            
            # 保存图像
            output_path = os.path.join(output_dir, f"{mode_name}_{i:06d}.png")
            torchvision.utils.save_image(sample, output_path)
            
            # 保存一些样本用于质量对比
            if i in [0, 5, 10, 15, 20, iterations-1]:
                quality_samples.append((i, sample.clone()))
            
            # 显示进度
            if i % 10 == 0 or i < 5:
                print(f"🖼️  {mode_name} {i+1:3d}/{iterations} | "
                      f"FPS: {img_per_sec:6.2f} | "
                      f"平均FPS: {avg_fps:6.2f} | "
                      f"用时: {elapsed:.3f}s")
        
        # 模式统计
        if results:
            avg_time = sum(results) / len(results)
            total_fps = len(results) / sum(results)
            min_time = min(results)
            max_time = max(results)
            
            print(f"\n📊 {description}性能统计")
            print("-" * 50)
            print(f"总图像数:     {len(results)}")
            print(f"平均生成时间: {avg_time:.3f}s")
            print(f"最快时间:     {min_time:.3f}s ({1/min_time:.2f} FPS)")
            print(f"最慢时间:     {max_time:.3f}s ({1/max_time:.2f} FPS)")
            print(f"平均FPS:      {total_fps:.2f}")
            print(f"总用时:       {sum(results):.2f}s")
            
            all_results[mode_name] = {
                "avg_time": avg_time,
                "total_fps": total_fps,
                "min_time": min_time,
                "max_time": max_time,
                "total_images": len(results)
            }
        
        # 生成质量样本网格
        if quality_samples:
            print(f"🎨 生成{mode_name}质量样本网格...")
            # 创建样本网格
            sample_tensors = [sample for _, sample in quality_samples]
            grid = torchvision.utils.make_grid(
                torch.cat(sample_tensors, dim=0), 
                nrow=3, 
                padding=2, 
                normalize=False
            )
            
            grid_path = os.path.join(output_dir, f"{mode_name}_quality_samples_grid.png")
            torchvision.utils.save_image(grid, grid_path)
            print(f"✅ {mode_name}质量样本网格已保存: {grid_path}")
        
        quality_samples = []  # 清空准备下一个模式
    
    # 整体性能对比
    print(f"\n🏆 帧缓冲性能总结")
    print("=" * 60)
    print(f"帧缓冲大小: {FRAME_BUFFER_SIZE}")
    print("-" * 60)
    print(f"{'模式':<15} {'平均FPS':<10} {'最快FPS':<10} {'图像数':<8}")
    print("-" * 60)
    
    for mode_name, stats in all_results.items():
        fastest_fps = 1.0 / stats["min_time"]
        print(f"{mode_name:<15} {stats['total_fps']:<10.2f} {fastest_fps:<10.2f} {stats['total_images']:<8}")
    
    # 帧缓冲效果分析
    print(f"\n💡 帧缓冲效果分析:")
    print(f"   📊 当前配置 (buffer_size={FRAME_BUFFER_SIZE}):")
    
    if FRAME_BUFFER_SIZE == 1:
        print(f"      - 无缓冲模式：每帧独立生成，延迟最低")
        print(f"      - 适合：单张图像生成、最低延迟需求")
    elif FRAME_BUFFER_SIZE <= 4:
        print(f"      - 小缓冲模式：平衡延迟和流畅度")
        print(f"      - 适合：实时应用、轻量级流式生成")
    else:
        print(f"      - 大缓冲模式：更高的流畅度，但延迟增加")
        print(f"      - 适合：高质量视频生成、批量处理")
    
    # 与不同buffer size的对比建议
    print(f"\n🔄 建议测试不同帧缓冲大小:")
    print(f"   - buffer_size=1: 最低延迟")
    print(f"   - buffer_size=2: 轻量级流式")
    print(f"   - buffer_size=4: 平衡模式（当前）")
    print(f"   - buffer_size=8: 高流畅度")
    
    print(f"\n🎉 帧缓冲测试完成！")
    print(f"📁 所有图像已保存到: {output_dir}")
    print(f"🔍 请检查不同模式的图像质量和连续性")
    
    return {
        "frame_buffer_size": FRAME_BUFFER_SIZE,
        "results": all_results,
        "output_dir": output_dir,
        "config": {
            "use_tiny_vae": USE_TINY_VAE,
            "vae_decode_method": VAE_DECODE_METHOD,
            "acceleration": ACCELERATION,
            "use_denoising_batch": USE_DENOISING_BATCH,
            "do_add_noise": DO_ADD_NOISE
        }
    }


if __name__ == "__main__":
    # 运行帧缓冲测试
    results = run_perflow_frame_buffer_test()
    
    print(f"\n🏆 帧缓冲测试总结:")
    print(f"帧缓冲大小: {results['frame_buffer_size']}")
    print(f"测试结果: {results['results']}")
    print(f"配置: {results['config']}")