import os
import sys
from typing import Literal, Dict, Optional

import fire

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.wrapper_perflow import PeRFlowWrapper

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def main(
    output: str = os.path.join(CURRENT_DIR, "..", "..", "images", "outputs", "perflow_output.png"),
    model_id_or_path: str = "hansyan/perflow-sd15-dreamshaper",
    lora_dict: Optional[Dict[str, float]] = None,
    prompt: str = "RAW photo, 8k uhd, dslr, high quality, film grain, highly detailed, masterpiece; A man with brown skin, a beard, and dark eyes",
    negative_prompt: str = "distorted, blur, smooth, low-quality, warm, haze, over-saturated, high-contrast, out of focus, dark",
    width: int = 512,
    height: int = 512,
    acceleration: Literal["none", "xformers", "tensorrt"] = "xformers",
    vae_decode_method: Literal["normalize", "dynamic", "clamp"] = "normalize",
    use_tiny_vae: bool = False,
    num_inference_steps: int = 4,
    guidance_scale: float = 7.5,
    seed: int = 1024,
    num_images: int = 1,
):
    """
    PeRFlow高质量图像生成

    Parameters
    ----------
    output : str, optional
        输出图像文件路径
    model_id_or_path : str
        PeRFlow模型路径，默认使用dreamshaper版本
    lora_dict : Optional[Dict[str, float]], optional
        LoRA字典，键为LoRA名称，值为缩放因子
        例如: {'LoRA_1': 0.5, 'LoRA_2': 0.7}
    prompt : str
        正向提示词
    negative_prompt : str
        负向提示词
    width : int, optional
        图像宽度，默认512
    height : int, optional
        图像高度，默认512
    acceleration : Literal["none", "xformers", "tensorrt"]
        加速方法，推荐xformers
    vae_decode_method : Literal["normalize", "dynamic", "clamp"]
        VAE解码方法：
        - "normalize": 标准归一化，推荐用于保持质量
        - "dynamic": 动态归一化，最大动态范围但可能有色偏
        - "clamp": 直接截断，会偏暗
    use_tiny_vae : bool, optional
        是否使用TinyVAE加速，False保证最佳质量
    num_inference_steps : int, optional
        推理步数，PeRFlow推荐4步
    guidance_scale : float, optional
        CFG引导缩放，默认7.5
    seed : int, optional
        随机种子，默认1024
    num_images : int, optional
        生成图像数量，默认1张
    """
    
    print("🎨 PeRFlow高质量图像生成")
    print(f"📝 提示词: {prompt}")
    print(f"🔧 VAE解码方法: {vae_decode_method}")
    print(f"📏 尺寸: {width}x{height}")
    print(f"🎯 生成数量: {num_images}")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output), exist_ok=True)
    
    # 初始化PeRFlow包装器
    wrapper = PeRFlowWrapper(
        model_id_or_path=model_id_or_path,
        t_index_list=[0, 1, 2, 3],  # PeRFlow标准4步
        lora_dict=lora_dict,
        mode="txt2img",
        output_type="pil",
        vae_decode_method=vae_decode_method,
        device="cuda",
        dtype=torch.float16,
        width=width,
        height=height,
        warmup=5,
        acceleration=acceleration,
        use_tiny_vae=use_tiny_vae,
        seed=seed,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    
    # 准备推理
    wrapper.prepare(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    
    # 生成图像
    import time
    start_time = time.time()
    
    if num_images == 1:
        # 单张图像
        image = wrapper.txt2img()
        image.save(output)
        print(f"✅ 图像已保存: {output}")
    else:
        # 批量生成
        images = wrapper.batch_generate(num_images=num_images, show_progress=True)
        
        # 保存多张图像
        base_name = os.path.splitext(output)[0]
        ext = os.path.splitext(output)[1]
        
        for i, image in enumerate(images):
            if num_images == 1:
                save_path = output
            else:
                save_path = f"{base_name}_{i+1:03d}{ext}"
            image.save(save_path)
            print(f"✅ 图像已保存: {save_path}")
    
    generation_time = time.time() - start_time
    
    # 性能统计
    stats = wrapper.get_performance_stats()
    print(f"\n📊 性能统计:")
    print(f"   总用时: {generation_time:.2f}s")
    print(f"   单张平均: {generation_time/num_images:.2f}s")
    print(f"   平均FPS: {num_images/generation_time:.2f}")
    print(f"   推理时间EMA: {stats['inference_time_ema']:.3f}s")


if __name__ == "__main__":
    import torch
    fire.Fire(main)