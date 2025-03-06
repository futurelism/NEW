import argparse
import os
import time
import torch
import numpy as np
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

def profile_clip_model(model_name, batch_sizes, image_resolutions, text_lengths, output_dir):
    """分析CLIP模型在不同配置下的性能"""
    results = []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for batch_size in batch_sizes:
        for image_resolution in image_resolutions:
            for text_length in text_lengths:
                # 加载模型
                model = CLIPModel.from_pretrained(model_name).to(device)
                processor = CLIPProcessor.from_pretrained(model_name)

                # 创建模拟输入
                dummy_images = [np.random.randint(0, 255, (image_resolution, image_resolution, 3),
                                                  dtype=np.uint8) for _ in range(batch_size)]
                dummy_images = [Image.fromarray(img) for img in dummy_images]

                dummy_texts = ["A" * text_length for _ in range(batch_size)]

                # 预处理输入
                inputs = processor(text=dummy_texts, images=dummy_images, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # 分别测量不同组件的时间
                # 1. 图像编码器
                start_time = time.time()
                with torch.no_grad():
                    image_features = model.get_image_features(**{k: v for k, v in inputs.items()
                                                                 if k in ['pixel_values']})
                image_encoder_time = (time.time() - start_time) * 1000  # 转换为毫秒

                # 2. 文本编码器
                start_time = time.time()
                with torch.no_grad():
                    text_features = model.get_text_features(**{k: v for k, v in inputs.items()
                                                               if k in ['input_ids', 'attention_mask']})
                text_encoder_time = (time.time() - start_time) * 1000  # 转换为毫秒

                # 3. 投影和相似度计算
                start_time = time.time()
                with torch.no_grad():
                    logits_per_image = model.logits_per_image(image_features, text_features)
                projection_similarity_time = (time.time() - start_time) * 1000  # 转换为毫秒

                # 记录结果
                results.append({
                    'model': model_name,
                    'batch_size': batch_size,
                    'image_resolution': image_resolution,
                    'text_length': text_length,
                    'time_stats.image_encoder.median': image_encoder_time,
                    'time_stats.text_encoder.median': text_encoder_time,
                    'time_stats.projection_similarity.median': projection_similarity_time,
                    'projection_dim': model.projection_dim,
                })

                # 释放内存
                del model
                del processor
                torch.cuda.empty_cache()

    # 保存结果
    results_df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(f"{output_dir}/clip_profiling_results.csv", index=False)

    return results_df

def main():
    parser = argparse.ArgumentParser(description='Profile CLIP models')
    parser.add_argument('--models', nargs='+', default=["openai/clip-vit-base-patch32"],
                        help='CLIP model names to profile')
    parser.add_argument('--batch_sizes', nargs='+', type=int, default=[1, 2, 4, 8, 16, 32, 64],
                        help='Batch sizes to profile')
    parser.add_argument('--image_resolutions', nargs='+', type=int, default=[224, 336],
                        help='Image resolutions to profile')
    parser.add_argument('--text_lengths', nargs='+', type=int, default=[16, 32, 64, 77],
                        help='Text lengths to profile')
    parser.add_argument('--output_dir', default='./profiling_outputs/clip',
                        help='Output directory for profiling results')

    args = parser.parse_args()

    for model_name in args.models:
        print(f"Profiling model: {model_name}")
        profile_clip_model(
            model_name=model_name,
            batch_sizes=args.batch_sizes,
            image_resolutions=args.image_resolutions,
            text_lengths=args.text_lengths,
            output_dir=f"{args.output_dir}/{model_name}"
        )

if __name__ == "__main__":
    main()
