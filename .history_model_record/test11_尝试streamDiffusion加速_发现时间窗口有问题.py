import os
import sys
import time
import threading
from multiprocessing import Process, Queue
from typing import Dict, List, Literal, Optional
import queue

import torch
from PIL import Image
from streamdiffusion.image_utils import postprocess_image

# 添加utils路径
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.wrapper import StreamDiffusionWrapper

# ========================
# 🔧 配置参数
# ========================

# 基础配置
MODEL_PATH = "hansyan/perflow-sd15-dreamshaper"  # PeRFlow模型
BATCH_SIZE = 1  # 批量大小
ACCELERATION = "xformers"  # none, xformers, tensorrt

# PeRFlow专用配置
USE_PERFLOW = True
PERFLOW_STEPS = 4  # PeRFlow最佳步数

# 其他配置
USE_TINY_VAE = True
WIDTH = 512
HEIGHT = 512
WARMUP = 10

# ========================
# 🚀 高性能图像保存系统
# ========================

class HighPerformanceImageSaver:
    def __init__(self, output_dir: str, max_queue_size: int = 100):
        self.output_dir = output_dir
        self.save_queue = Queue(maxsize=max_queue_size)
        self.save_process = None
        self.total_saved = 0
        self.is_running = False
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 图像保存目录: {output_dir}")
    
    def start(self):
        """启动高性能保存进程"""
        self.is_running = True
        self.save_process = Process(target=self._save_worker, daemon=True)
        self.save_process.start()
        print("💾 高性能图像保存器已启动")
    
    def _save_worker(self):
        """专用保存工作进程"""
        saved_count = 0
        last_report_time = time.time()
        
        while True:
            try:
                # 非阻塞获取保存任务
                try:
                    item = self.save_queue.get(timeout=0.5)
                except queue.Empty:
                    if not self.is_running and self.save_queue.empty():
                        break
                    continue
                
                if item is None:  # 退出信号
                    break
                
                images, filename_base, batch_idx = item
                
                # 处理图像保存
                if isinstance(images, list):
                    # 多张图像
                    for i, img in enumerate(images):
                        filename = f"{filename_base}_batch{batch_idx:06d}_img{i:02d}.png"
                        filepath = os.path.join(self.output_dir, filename)
                        img.save(filepath, optimize=True)  # 优化保存
                        saved_count += 1
                else:
                    # 单张图像
                    filename = f"{filename_base}_batch{batch_idx:06d}.png"
                    filepath = os.path.join(self.output_dir, filename)
                    images.save(filepath, optimize=True)
                    saved_count += 1
                
                # 每30秒报告一次保存状态
                current_time = time.time()
                if current_time - last_report_time > 30:
                    print(f"💾 已保存 {saved_count} 张图像")
                    last_report_time = current_time
                    
            except Exception as e:
                print(f"❌ 保存图像时出错: {e}")
                continue
        
        print(f"💾 保存进程结束，总计保存: {saved_count} 张图像")
    
    def save_async(self, images, batch_idx: int):
        """异步保存图像，不阻塞主进程"""
        try:
            # 使用时间戳作为文件名基础
            timestamp = int(time.time() * 1000)  # 毫秒级时间戳
            filename_base = f"perflow_t3_{timestamp}"
            
            # 非阻塞添加到队列
            self.save_queue.put((images, filename_base, batch_idx), block=False)
            return True
        except queue.Full:
            # 队列满时跳过，不影响生成性能
            if batch_idx % 50 == 0:  # 每50次提醒一次
                print(f"⚠️  保存队列已满，跳过批次 {batch_idx}")
            return False
    
    def get_queue_status(self):
        """获取队列状态"""
        return self.save_queue.qsize()
    
    def stop(self):
        """优雅停止保存进程"""
        print("🛑 正在停止图像保存器...")
        self.is_running = False
        
        if self.save_process and self.save_process.is_alive():
            # 等待队列清空（最多30秒）
            wait_start = time.time()
            while self.save_queue.qsize() > 0 and (time.time() - wait_start) < 30:
                remaining = self.save_queue.qsize()
                print(f"⏳ 等待保存完成，剩余: {remaining} 张")
                time.sleep(2)
            
            # 发送停止信号
            try:
                self.save_queue.put(None, timeout=1)
            except queue.Full:
                pass
            
            # 等待进程结束
            self.save_process.join(timeout=10)
            if self.save_process.is_alive():
                print("⚠️  强制终止保存进程")
                self.save_process.terminate()
            
        print("✅ 图像保存器已停止")

# ========================
# 主生成函数
# ========================

def run_official_batch_generation(
    prompt: str = "RAW photo, masterpiece, 1girl with brown hair, glasses, detailed face",
    negative_prompt: str = "blurry, low quality, distorted, bad anatomy",
    iterations: int = 100,
):
    """
    使用官方推荐的方式进行批量生成
    """
    
    # 创建输出目录
    output_dir = "perflow_t3_output"
    
    print("=== StreamDiffusion + PeRFlow (时间步3) ===")
    print(f"模型: {MODEL_PATH}")
    print(f"批量大小: {BATCH_SIZE}")
    print(f"加速方法: {ACCELERATION}")
    print(f"时间步: [3] (仅最后一步)")
    print(f"PeRFlow模式: {USE_PERFLOW}")
    
    # ========================
    # 启动高性能图像保存器
    # ========================
    
    image_saver = HighPerformanceImageSaver(output_dir)
    image_saver.start()
    
    # ========================
    # 创建StreamDiffusionWrapper
    # ========================
    
    try:
        print("🔧 初始化StreamDiffusion...")
        
        stream = StreamDiffusionWrapper(
            model_id_or_path=MODEL_PATH,
            t_index_list=[3, 2, 1, 0], 
            lora_dict=None,
            mode="txt2img",
            frame_buffer_size=BATCH_SIZE,  # 批量大小
            width=WIDTH,
            height=HEIGHT,
            warmup=WARMUP,
            acceleration=ACCELERATION,
            use_lcm_lora=False,  # PeRFlow不需要LCM
            use_tiny_vae=USE_TINY_VAE,
            enable_similar_image_filter=False,
            use_denoising_batch=True,  # 启用去噪批处理
            cfg_type="none",  # txt2img模式限制
            seed=42,
        )
        
        print("✅ StreamDiffusionWrapper创建成功")
        print(f"📊 实际batch_size: {stream.batch_size}")
        
    except Exception as e:
        print(f"❌ StreamDiffusionWrapper创建失败: {e}")
        image_saver.stop()
        return
    
    # ========================
    # 准备生成
    # ========================
    
    try:
        stream.prepare(
            prompt=prompt,
            num_inference_steps=PERFLOW_STEPS,  # 4步
        )
        print("✅ 生成准备完成")
        
    except Exception as e:
        print(f"❌ 生成准备失败: {e}")
        image_saver.stop()
        return
    
    # ========================
    # 批量生成循环
    # ========================
    
    print(f"🔥 开始生成 ({iterations} 次迭代)...")
    print("💡 提示: 使用t_index_list=[3]可能影响图像质量")
    
    results = []
    successful_iterations = 0
    
    try:
        for i in range(iterations):
            iteration_start_time = time.time()
            
            try:
                # 执行生成
                x_outputs = stream.stream.txt2img()
                
                # 立即转换为PIL（在主线程中完成，避免序列化问题）
                images = postprocess_image(x_outputs.cpu(), output_type="pil")
                
                # 异步保存图像（不阻塞生成）
                save_success = image_saver.save_async(images, i)
                
                # 计算性能
                elapsed = time.time() - iteration_start_time
                results.append(elapsed)
                successful_iterations += 1
                
                # 计算FPS
                num_images = len(images) if isinstance(images, list) else 1
                fps = num_images / elapsed
                avg_fps = successful_iterations / sum(results)
                
                # 显示进度
                if i % 10 == 0 or i < 10:  # 前10次每次显示，之后每10次显示
                    queue_size = image_saver.get_queue_status()
                    save_status = "✅" if save_success else "⚠️ "
                    print(f"{save_status} 迭代 {i+1:3d}/{iterations} | "
                          f"图像: {num_images} | "
                          f"FPS: {fps:6.2f} | "
                          f"平均FPS: {avg_fps:6.2f} | "
                          f"队列: {queue_size:3d} | "
                          f"用时: {elapsed:.3f}s")
                
                # 每50次迭代显示详细状态
                if (i + 1) % 50 == 0:
                    queue_size = image_saver.get_queue_status()
                    total_fps = successful_iterations / sum(results)
                    print(f"📊 进度报告: {i+1}/{iterations} 完成 | "
                          f"总体FPS: {total_fps:.2f} | "
                          f"保存队列: {queue_size}")
                
            except Exception as iteration_error:
                print(f"❌ 迭代 {i+1} 失败: {iteration_error}")
                continue
                
    except KeyboardInterrupt:
        print("\n🛑 用户中断生成...")
    
    except Exception as e:
        print(f"\n❌ 生成过程出错: {e}")
    
    # ========================
    # 停止保存器并统计
    # ========================
    
    image_saver.stop()
    
    # ========================
    # 性能统计
    # ========================
    
    if results:
        avg_time = sum(results) / len(results)
        total_fps = successful_iterations / sum(results)
        
        print(f"\n📊 === 性能统计 ===")
        print(f"成功迭代: {successful_iterations}/{iterations}")
        print(f"平均迭代时间: {avg_time:.3f}s")
        print(f"平均FPS: {total_fps:.2f}")
        print(f"配置: PeRFlow(t=3) + {ACCELERATION} + 批量{BATCH_SIZE}")
        print(f"图像保存到: {output_dir}")
        
    else:
        print("❌ 没有成功完成的迭代")

# ========================
# 主函数
# ========================

if __name__ == "__main__":
    print("PeRFlow + StreamDiffusion 批量生成工具")
    print("=" * 40)
    
    # 直接运行StreamDiffusion + PeRFlow模式
    run_official_batch_generation()