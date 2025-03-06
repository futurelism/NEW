"""
CLIP 模型性能分析脚本 - 内存优化版本
"""
import argparse
import gc
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import CLIPModel, CLIPProcessor

from vidur.logger import init_logger
from vidur.profiling.common.cuda_timer import CudaTimer
from vidur.profiling.common.timer_stats_store import TimerStatsStore

logger = init_logger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="CLIP模型性能分析")
    parser.add_argument("--models", nargs="+", required=True, help="要分析的模型列表")
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 2, 4, 8],
                        help="批处理大小列表")
    parser.add_argument("--output_dir", type=str, default="profiling_outputs/clip",
                        help="输出目录")
    parser.add_argument("--cpu_only", action="store_true", help="仅使用CPU进行分析")
    return parser.parse_args()

def profile_clip_model(
        model_name: str,
        batch_sizes: List[int],
        output_dir: str,
        cpu_only: bool = False
) -> None:
    """
    对CLIP模型进行性能分析

    Args:
        model_name: CLIP模型名称
        batch_sizes: 要测试的批次大小列表
        output_dir: 输出目录
        cpu_only: 是否仅使用CPU
    """
    logger.info(f"开始分析模型 {model_name}")

    # 设置设备
    device = torch.device("cpu" if cpu_only else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        # 清理 CUDA 缓存
        torch.cuda.empty_cache()

    # 加载模型和处理器 - 使用 half 精度减少内存使用
    try:
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)

        # 移至设备
        model.to(device)
        if device.type == "cuda":
            model = model.half()  # 使用半精度

        model.eval()
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        return

    # 获取模型配置
    config = model.config
    image_size = config.vision_config.image_size
    projection_dim = config.projection_dim
    max_position_embeddings = getattr(config.text_config, "max_position_embeddings", 77)

    # 准备性能分析结果存储
    results = []

    for batch_size in batch_sizes:
        logger.info(f"分析批次大小: {batch_size}")

        try:
            # 清理内存
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # 减小图像尺寸以节省内存
            test_image_size = min(image_size, 224)

            # 创建随机输入
            input_dtype = torch.float16 if device.type == "cuda" else torch.float32
            dummy_images = torch.randn(batch_size, 3, test_image_size, test_image_size, dtype=input_dtype).to(device)
            dummy_texts = ["This is a simple test text"] * batch_size

            # 处理文本输入
            text_inputs = processor(text=dummy_texts, return_tensors="pt", padding=True, truncation=True)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

            # 预热 - 减少预热次数
            with torch.no_grad():
                for _ in range(3):
                    _ = model.get_image_features(dummy_images)
                    _ = model.get_text_features(**text_inputs)
                    if device.type == "cuda":
                        torch.cuda.synchronize()

            # 测量图像编码器时间
            timer_stats = TimerStatsStore()
            for _ in range(5):  # 减少迭代次数
                with torch.no_grad(), CudaTimer("image_encoder", timer_stats):
                    _ = model.get_image_features(dummy_images)
                    if device.type == "cuda":
                        torch.cuda.synchronize()

            image_encoder_time = timer_stats.get_timer_stats("image_encoder").median

            # 测量文本编码器时间
            timer_stats.clear()
            for _ in range(5):  # 减少迭代次数
                with torch.no_grad(), CudaTimer("text_encoder", timer_stats):
                    _ = model.get_text_features(**text_inputs)
                    if device.type == "cuda":
                        torch.cuda.synchronize()

            text_encoder_time = timer_stats.get_timer_stats("text_encoder").median

            # 测量投影和相似度计算时间
            timer_stats.clear()
            for _ in range(5):  # 减少迭代次数
                with torch.no_grad(), CudaTimer("projection_similarity", timer_stats):
                    image_features = model.get_image_features(dummy_images)
                    text_features = model.get_text_features(**text_inputs)
                    # 归一化特征
                    image_features = image_features / (image_features.norm(dim=1, keepdim=True) + 1e-8)
                    text_features = text_features / (text_features.norm(dim=1, keepdim=True) + 1e-8)
                    # 计算相似度
                    logits_per_image = image_features @ text_features.t()
                    if device.type == "cuda":
                        torch.cuda.synchronize()

            projection_similarity_time = timer_stats.get_timer_stats("projection_similarity").median

            # 收集结果
            results.append({
                "model_name": model_name,
                "batch_size": batch_size,
                "image_resolution": image_size,
                "text_length": max_position_embeddings,
                "projection_dim": projection_dim,
                "time_stats.image_encoder.median": image_encoder_time,
                "time_stats.text_encoder.median": text_encoder_time,
                "time_stats.projection_similarity.median": projection_similarity_time
            })

        except Exception as e:
            logger.error(f"批次大小 {batch_size} 分析失败: {str(e)}")
            # 尝试利用已有数据进行估计
            if len(results) > 0:
                # 使用已有结果线性估计
                last_result = results[-1]
                factor = batch_size / last_result["batch_size"]
                results.append({
                    "model_name": model_name,
                    "batch_size": batch_size,
                    "image_resolution": image_size,
                    "text_length": max_position_embeddings,
                    "projection_dim": projection_dim,
                    "time_stats.image_encoder.median": last_result["time_stats.image_encoder.median"] * factor,
                    "time_stats.text_encoder.median": last_result["time_stats.text_encoder.median"] * factor,
                    "time_stats.projection_similarity.median": last_result["time_stats.projection_similarity.median"] * factor,
                    "estimated": True
                })
            continue

        # 清理资源
        del dummy_images, text_inputs
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # 释放模型占用的内存
    del model, processor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # 保存结果
    df = pd.DataFrame(results)
    model_dir = model_name.replace("/", "_")
    output_path = os.path.join(output_dir, model_dir)
    os.makedirs(output_path, exist_ok=True)
    df.to_csv(os.path.join(output_path, "clip.csv"), index=False)

    logger.info(f"模型 {model_name} 分析完成，结果已保存到 {output_path}/clip.csv")

def main():
    """主函数"""
    args = parse_args()

    # 如果指定的是时间戳目录，则创建
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # 分析每个模型
    for model_name in args.models:
        try:
            profile_clip_model(
                model_name=model_name,
                batch_sizes=args.batch_sizes,
                output_dir=output_dir,
                cpu_only=args.cpu_only
            )
        except Exception as e:
            logger.error(f"分析模型 {model_name} 时出错: {str(e)}")

if __name__ == "__main__":
    main()
