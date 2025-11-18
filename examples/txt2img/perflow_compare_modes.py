import os
import sys
import time
from typing import Literal, Dict, Optional

import fire
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.wrapper_perflow import PeRFlowWrapper

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def main(
    output_dir: str = os.path.join(CURRENT_DIR, "..", "..", "images", "outputs", "perflow_compare"),
    model_id_or_path: str = "hansyan/perflow-sd15-dreamshaper",
    prompt: str = "RAW photo, 8k uhd, dslr, high quality, film grain, highly detailed, masterpiece; A beautiful landscape with mountains and lake",
    negative_prompt: str = "distorted, blur, smooth, low-quality, warm, haze, over-saturated, high-contrast, out of focus, dark",
    width: int = 512,
    height: int = 512,
    seed: int = 1024,
):
    """
    对比不同VAE解码方法的效果
    
    Parameters
    ----------
    output_dir : str
        输出目录
    model_id_or_path : str
        PeRFlow模型路径
    prompt : str
        提示词
    negative_prompt : str
        负向提示词
    width : int
        图像宽度
    height : int  
        图像高度
    seed : int
        随机种子
    """
    
    print("🔍 PeRFlow VAE解码方法对比测试")
    print(f"📝 提示词: {prompt}")
    print(f"📏 尺寸: {width}x{height}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 测试的解码方法
    decode_methods = ["normalize", "dynamic", "clamp"]
    vae_types = [False, True]  # False: 原始VAE, True: TinyVAE
    
    results = {}
    
    for use_tiny_vae in vae_types:
        vae_name = "tiny_vae" if use_tiny_vae else "original_vae"
        
        for decode_method in decode_methods:
            config_name = f"{vae_name}_{decode_method}"
            print(f"\n🔧 测试配置: {config_name}")
            
            # 创建包装器
            wrapper = PeRFlowWrapper(
                model_id_or_path=model_id_or_path,
                t_index_list=[0, 1, 2, 3],
                mode="txt2img",
                output_type="pil",
                vae_decode_method=decode_method,
                device="cuda",
                dtype=torch.float16,
                width=width,
                height=height,
                warmup=3,  # 减少预热以加快测试
                acceleration="xformers",
                use_tiny_vae=use_tiny_vae,
                seed=seed,
                num_inference_steps=4,
                guidance_scale=7.5,
            )
            
            # 准备推理
            wrapper.prepare(
                prompt=prompt,
                negative_prompt=negative_prompt,
            )
            
            # 生成并计时
            start_time = time.time()
            image = wrapper.txt2img()
            generation_time = time.time() - start_time
            
            # 保存图像
            output_path = os.path.join(output_dir, f"{config_name}.png")
            image.save(output_path)
            
            # 记录结果
            results[config_name] = {
                "time": generation_time,
                "vae_type": "TinyVAE" if use_tiny_vae else "Original VAE",
                "decode_method": decode_method,
                "output_path": output_path
            }
            
            print(f"✅ 完成 - 用时: {generation_time:.3f}s")
            print(f"   图像保存: {output_path}")
            
            # 清理GPU内存
            del wrapper
            torch.cuda.empty_cache()
    
    # 生成报告
    print(f"\n📊 测试结果总结:")
    print("=" * 80)
    print(f"{'配置':<25} {'VAE类型':<15} {'解码方法':<12} {'用时(s)':<10} {'FPS':<8}")
    print("-" * 80)
    
    for config_name, result in results.items():
        fps = 1.0 / result["time"]
        print(f"{config_name:<25} {result['vae_type']:<15} {result['decode_method']:<12} {result['time']:<10.3f} {fps:<8.2f}")
    
    print(f"\n📁 所有图像已保存到: {output_dir}")
    print(f"\n💡 建议:")
    print(f"   - normalize模式: 最佳质量平衡，推荐日常使用")
    print(f"   - dynamic模式: 最大动态范围，但可能有色偏")
    print(f"   - clamp模式: 速度最快但会偏暗，不推荐")
    print(f"   - TinyVAE: 显著加速但轻微质量损失")


if __name__ == "__main__":
    fire.Fire(main)