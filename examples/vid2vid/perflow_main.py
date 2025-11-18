import os
import sys
import time
from pathlib import Path
from typing import Literal, Dict, Optional

import cv2
import fire
import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.wrapper_perflow import PeRFlowWrapper

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def process_video(
    input_video: str,
    output_video: str = os.path.join(CURRENT_DIR, "..", "..", "images", "outputs", "perflow_video_output.mp4"),
    model_id_or_path: str = "hansyan/perflow-sd15-dreamshaper", 
    prompt: str = "anime style, vibrant colors, detailed",
    negative_prompt: str = "blurry, low quality, distorted",
    vae_decode_method: Literal["normalize", "dynamic", "clamp"] = "normalize",
    use_tiny_vae: bool = True,  # 视频处理推荐使用TinyVAE加速
    acceleration: Literal["none", "xformers", "tensorrt"] = "xformers",
    guidance_scale: float = 7.5,
    strength: float = 0.8,  # img2img强度
    seed: int = 42,
    fps: Optional[int] = None,  # 输出fps，None则使用输入视频的fps
    max_frames: Optional[int] = None,  # 最大处理帧数，用于测试
    skip_frames: int = 1,  # 跳帧处理，1表示处理每一帧
    output_width: int = 512,
    output_height: int = 512,
):
    """
    PeRFlow视频到视频处理
    
    Parameters
    ----------
    input_video : str
        输入视频路径
    output_video : str
        输出视频路径
    model_id_or_path : str
        PeRFlow模型路径
    prompt : str
        风格化提示词
    negative_prompt : str
        负向提示词
    vae_decode_method : str
        VAE解码方法，推荐"normalize"
    use_tiny_vae : bool
        视频处理推荐True以提高速度
    acceleration : str
        加速方法
    guidance_scale : float
        CFG引导强度
    strength : float
        img2img变换强度，0.0-1.0
    seed : int
        随机种子
    fps : int
        输出视频帧率，None使用输入视频帧率
    max_frames : int
        最大处理帧数，用于测试
    skip_frames : int
        跳帧处理间隔
    output_width : int
        输出宽度
    output_height : int
        输出高度
    """
    
    print("🎬 PeRFlow视频处理开始")
    print(f"📹 输入视频: {input_video}")
    print(f"💾 输出视频: {output_video}")
    print(f"🎨 风格提示: {prompt}")
    print(f"🔧 VAE解码: {vae_decode_method}")
    print(f"⚡ 使用TinyVAE: {use_tiny_vae}")
    
    # 检查输入视频
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"输入视频不存在: {input_video}")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    
    # 读取输入视频信息
    cap = cv2.VideoCapture(input_video)
    input_fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📊 输入视频信息:")
    print(f"   分辨率: {input_width}x{input_height}")
    print(f"   帧率: {input_fps} FPS")
    print(f"   总帧数: {total_frames}")
    
    if fps is None:
        fps = input_fps
    
    if max_frames:
        process_frames = min(max_frames, total_frames)
    else:
        process_frames = total_frames
    
    actual_process_frames = process_frames // skip_frames
    print(f"   将处理: {actual_process_frames} 帧 (跳帧间隔: {skip_frames})")
    
    # 初始化PeRFlow包装器
    wrapper = PeRFlowWrapper(
        model_id_or_path=model_id_or_path,
        t_index_list=[0, 1, 2, 3],
        mode="img2img",  # 视频处理使用img2img模式
        output_type="pil",
        vae_decode_method=vae_decode_method,
        device="cuda",
        dtype=torch.float16,
        width=output_width,
        height=output_height,
        warmup=3,
        acceleration=acceleration,
        use_tiny_vae=use_tiny_vae,
        seed=seed,
        num_inference_steps=4,
        guidance_scale=guidance_scale,
    )
    
    # 准备推理
    wrapper.prepare(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
    )
    
    # 设置视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (output_width, output_height))
    
    # 处理视频帧
    frame_count = 0
    processed_count = 0
    start_time = time.time()
    
    print(f"\n🚀 开始处理视频帧...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count >= process_frames:
                break
            
            # 跳帧处理
            if frame_count % skip_frames == 0:
                # 预处理帧
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_pil = frame_pil.resize((output_width, output_height), Image.Resampling.LANCZOS)
                
                # PeRFlow处理
                frame_start = time.time()
                processed_pil = wrapper.img2img(frame_pil)
                frame_time = time.time() - frame_start
                
                # 转换回OpenCV格式
                processed_np = np.array(processed_pil)
                processed_bgr = cv2.cvtColor(processed_np, cv2.COLOR_RGB2BGR)
                
                # 写入视频
                out.write(processed_bgr)
                
                processed_count += 1
                
                # 显示进度
                if processed_count % 10 == 0 or processed_count <= 5:
                    elapsed = time.time() - start_time
                    avg_fps = processed_count / elapsed if elapsed > 0 else 0
                    eta = (actual_process_frames - processed_count) / avg_fps if avg_fps > 0 else 0
                    
                    print(f"📸 帧 {processed_count:4d}/{actual_process_frames} "
                          f"| 处理用时: {frame_time:.3f}s "
                          f"| 平均FPS: {avg_fps:.2f} "
                          f"| ETA: {eta:.1f}s")
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\n⚠️  用户中断处理")
    
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    total_time = time.time() - start_time
    
    # 性能统计
    print(f"\n✅ 视频处理完成!")
    print(f"📊 性能统计:")
    print(f"   处理帧数: {processed_count}")
    print(f"   总用时: {total_time:.2f}s")
    print(f"   平均每帧: {total_time/processed_count:.3f}s")
    print(f"   处理FPS: {processed_count/total_time:.2f}")
    print(f"💾 输出视频: {output_video}")


def process_webcam(
    output_video: str = os.path.join(CURRENT_DIR, "..", "..", "images", "outputs", "perflow_webcam_output.mp4"),
    model_id_or_path: str = "hansyan/perflow-sd15-dreamshaper",
    prompt: str = "anime style, vibrant colors, detailed",
    negative_prompt: str = "blurry, low quality, distorted", 
    vae_decode_method: Literal["normalize", "dynamic", "clamp"] = "normalize",
    use_tiny_vae: bool = True,
    duration: int = 30,  # 录制时长（秒）
    display: bool = True,  # 是否显示实时预览
):
    """
    实时摄像头处理
    """
    print("📹 PeRFlow实时摄像头处理")
    print(f"⏱️  录制时长: {duration}秒")
    print(f"🎨 风格: {prompt}")
    print("按 'q' 提前退出")
    
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头")
    
    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    fps = 30
    
    # 初始化PeRFlow
    wrapper = PeRFlowWrapper(
        model_id_or_path=model_id_or_path,
        t_index_list=[0, 1, 2, 3],
        mode="img2img",
        output_type="pil",
        vae_decode_method=vae_decode_method,
        use_tiny_vae=use_tiny_vae,
        width=512,
        height=512,
        warmup=5,
    )
    
    wrapper.prepare(prompt=prompt, negative_prompt=negative_prompt)
    
    # 视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (512, 512))
    
    start_time = time.time()
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 检查时间限制
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break
            
            # 处理帧
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb).resize((512, 512))
            
            processed_pil = wrapper.img2img(frame_pil)
            processed_np = np.array(processed_pil)
            processed_bgr = cv2.cvtColor(processed_np, cv2.COLOR_RGB2BGR)
            
            # 保存到视频
            out.write(processed_bgr)
            
            # 显示预览
            if display:
                # 并排显示原始和处理后的帧
                original_resized = cv2.resize(frame, (512, 512))
                combined = np.hstack([original_resized, processed_bgr])
                cv2.imshow('PeRFlow Real-time (Original | Processed)', combined)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            
            if frame_count % 30 == 0:
                avg_fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"📸 处理了 {frame_count} 帧 | 平均FPS: {avg_fps:.2f}")
    
    except KeyboardInterrupt:
        print("\n⚠️  用户中断")
    
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    total_time = time.time() - start_time
    print(f"\n✅ 实时处理完成!")
    print(f"📊 统计: {frame_count} 帧，{total_time:.1f}s，平均 {frame_count/total_time:.2f} FPS")
    print(f"💾 输出: {output_video}")


if __name__ == "__main__":
    fire.Fire({
        "video": process_video,
        "webcam": process_webcam,
    })