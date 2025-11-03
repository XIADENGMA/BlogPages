#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
使用 Web 浏览器可视化 LeRobotDataset 中任意 episode 的所有帧数据。
Visualize data of **all** frames of any episode of a dataset of type LeRobotDataset in a web browser.

注意 Note:
    - Episode 的最后一帧不一定对应最终状态 / The last frame doesn't always correspond to a final state
    - 图像可能存在压缩伪影 / Images may show compression artifacts from mp4 encoding

访问 Access:
    浏览器打开 / Open in browser: http://localhost:PORT
"""

import argparse
import gc
import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import rerun as rr
import torch
import torch.utils.data
import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, DONE, OBS_STATE, REWARD


class EpisodeSampler(torch.utils.data.Sampler):
    """用于采样单个 episode 的所有帧 / Sampler for all frames of a single episode."""

    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]
        to_idx = dataset.meta.episodes["dataset_to_index"][episode_index]
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self) -> Iterator:
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    """
    将 PyTorch CHW float32 图像转换为 NumPy HWC uint8 格式。
    Convert PyTorch CHW float32 image to NumPy HWC uint8 format.
    """
    assert chw_float32_torch.dtype == torch.float32
    assert chw_float32_torch.ndim == 3
    c, h, w = chw_float32_torch.shape
    assert c < h and c < w, (
        f"期望通道优先格式，但得到 / expect channel first images, but got {chw_float32_torch.shape}"
    )
    hwc_uint8_numpy = (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()
    return hwc_uint8_numpy


def visualize_episode(
    dataset: LeRobotDataset,
    episode_index: int,
    batch_size: int = 32,
    num_workers: int = 0,
) -> None:
    """
    在 Rerun 中可视化单个 episode 的所有帧。
    Visualize all frames of a single episode in Rerun.

    Args:
        dataset: LeRobot 数据集 / LeRobot dataset
        episode_index: Episode 索引 / Episode index
        batch_size: 批处理大小 / Batch size for dataloader
        num_workers: 数据加载进程数 / Number of worker processes
    """
    logging.info(
        f"📊 加载 Episode {episode_index} 的数据加载器 / Loading dataloader for Episode {episode_index}"
    )

    episode_sampler = EpisodeSampler(dataset, episode_index)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        sampler=episode_sampler,
    )

    total_frames = len(episode_sampler)
    logging.info(
        f"📈 Episode {episode_index} 共有 {total_frames} 帧 / Episode {episode_index} has {total_frames} frames"
    )

    # 记录数据到 Rerun / Log data to Rerun
    for batch in tqdm.tqdm(dataloader, total=len(dataloader), desc=f"Episode {episode_index}"):
        # 遍历批次中的每一帧 / iterate over the batch
        for i in range(len(batch["index"])):
            rr.set_time_sequence("frame_index", batch["frame_index"][i].item())
            rr.set_time_seconds("timestamp", batch["timestamp"][i].item())

            # 显示相机图像 / display camera images
            for key in dataset.meta.camera_keys:
                rr.log(f"cameras/{key}", rr.Image(to_hwc_uint8_numpy(batch[key][i])))

            # 显示动作空间的每个维度 / display each dimension of action space
            if ACTION in batch:
                for dim_idx, val in enumerate(batch[ACTION][i]):
                    rr.log(f"{ACTION}/dim_{dim_idx}", rr.Scalar(val.item()))

            # 显示观测状态空间的每个维度 / display each dimension of observed state space
            if OBS_STATE in batch:
                for dim_idx, val in enumerate(batch[OBS_STATE][i]):
                    rr.log(f"state/dim_{dim_idx}", rr.Scalar(val.item()))

            # 显示完成标志 / display done flag
            if DONE in batch:
                rr.log(DONE, rr.Scalar(batch[DONE][i].item()))

            # 显示奖励 / display reward
            if REWARD in batch:
                rr.log(REWARD, rr.Scalar(batch[REWARD][i].item()))

            # 显示成功标志 / display success flag
            if "next.success" in batch:
                rr.log("success", rr.Scalar(batch["next.success"][i].item()))

    logging.info(f"✅ Episode {episode_index} 可视化完成 / Episode {episode_index} visualization complete")


def visualize_dataset_web(
    dataset: LeRobotDataset,
    episode_indices: list[int],
    batch_size: int = 32,
    num_workers: int = 0,
    port: int = 9090,
    open_browser: bool = True,
    memory_limit: str = "25%",
) -> None:
    """
    使用 Web 界面可视化数据集。
    Visualize dataset using web interface.

    Args:
        dataset: LeRobot 数据集 / LeRobot dataset
        episode_indices: 要可视化的 episode 索引列表 / List of episode indices to visualize
        batch_size: 批处理大小 / Batch size for dataloader
        num_workers: 数据加载进程数 / Number of worker processes
        port: Web 服务器端口 / Web server port
        open_browser: 是否自动打开浏览器 / Whether to automatically open browser
        memory_limit: Rerun 内存限制 / Memory limit for Rerun
    """
    repo_id = dataset.repo_id

    # 初始化 Rerun Web 界面 / Initialize Rerun web interface
    logging.info("🌐 启动 Rerun Web 界面 / Starting Rerun Web interface")
    logging.info(f"📍 访问地址 / Access URL: http://localhost:{port}")
    logging.info(f"💾 内存限制 / Memory limit: {memory_limit}")

    rr.init(f"{repo_id}_web_viz", spawn=False)

    # 手动触发垃圾回收，避免阻塞 / Manually call garbage collector to avoid blocking
    gc.collect()

    # 启动 Web 服务器 / Start web server
    rr.serve_web(
        open_browser=open_browser,
        web_port=port,
        server_memory_limit=memory_limit,
    )

    # 可视化每个 episode / Visualize each episode
    for episode_idx in episode_indices:
        if episode_idx >= len(dataset.meta.episodes):
            logging.warning(
                f"⚠️  Episode {episode_idx} 不存在，跳过 / Episode {episode_idx} does not exist, skipping"
            )
            continue

        # 为每个 episode 创建记录路径 / Create recording path for each episode
        rr.log(f"episode_{episode_idx}/info", rr.TextLog(f"Episode {episode_idx}"), static=True)

        # 设置时间序列标记当前 episode / Set time sequence for current episode
        rr.set_time_sequence("episode", episode_idx)

        visualize_episode(
            dataset=dataset,
            episode_index=episode_idx,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    logging.info("✨ 所有 episode 可视化完成 / All episodes visualization complete")
    logging.info("🌐 Web 服务器持续运行中，按 Ctrl+C 退出 / Web server running, press Ctrl+C to exit")

    # 保持服务器运行 / Keep server running
    try:
        import time

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("👋 收到 Ctrl-C，正在退出 / Ctrl-C received, exiting")


def main():
    parser = argparse.ArgumentParser(
        description="使用 Web 浏览器可视化 LeRobot 数据集 / Visualize LeRobot dataset in web browser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="数据集仓库 ID / Dataset repository ID (e.g. `lerobot/pusht` or `xiadengma/record-test-so101`)",
    )

    # Episode 选择参数 / Episode selection arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--episode-index",
        type=int,
        help="要可视化的单个 episode 索引 / Single episode index to visualize",
    )
    group.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        help="要可视化的多个 episode 索引 / Multiple episode indices to visualize (e.g. 0 1 2 3)",
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="本地数据集根目录 / Root directory for local dataset (e.g. `--root ./data/datasets/xiadengma/record-test-so101`)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="DataLoader 批处理大小 / Batch size for DataLoader (default: 32)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader 进程数 / Number of DataLoader worker processes (default: 4)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9090,
        help="Web 服务器端口 / Web server port (default: 9090)",
    )
    parser.add_argument(
        "--open-browser",
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        default=True,
        help="是否自动打开浏览器 / Whether to automatically open browser (default: True)",
    )
    parser.add_argument(
        "--memory-limit",
        type=str,
        default="25%",
        help="Rerun 内存限制 / Memory limit for Rerun (default: 25%%)",
    )
    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help="时间戳容差（秒）/ Tolerance in seconds for timestamps (default: 1e-4)",
    )

    args = parser.parse_args()

    # 配置日志 / Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 确定要可视化的 episode 列表 / Determine episode list
    if args.episode_index is not None:
        episode_indices = [args.episode_index]
    else:
        episode_indices = args.episodes

    logging.info("=" * 80)
    logging.info("🤖 LeRobot 数据集 Web 可视化工具 / LeRobot Dataset Web Visualizer")
    logging.info("=" * 80)
    logging.info(f"📦 数据集 / Dataset: {args.repo_id}")
    logging.info(f"📂 根目录 / Root: {args.root if args.root else 'HuggingFace Cache'}")
    logging.info(f"📊 Episodes: {episode_indices}")
    logging.info("=" * 80)

    # 加载数据集 / Load dataset
    logging.info("🔄 正在加载数据集 / Loading dataset...")
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        episodes=episode_indices,
        root=args.root,
        tolerance_s=args.tolerance_s,
    )

    logging.info("✅ 数据集加载成功 / Dataset loaded successfully")
    logging.info(f"📈 数据集总帧数 / Total frames: {len(dataset)}")
    logging.info(f"📹 相机数量 / Number of cameras: {len(dataset.meta.camera_keys)}")
    logging.info(f"🎥 相机列表 / Camera keys: {dataset.meta.camera_keys}")

    # 启动 Web 可视化 / Start web visualization
    visualize_dataset_web(
        dataset=dataset,
        episode_indices=episode_indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        port=args.port,
        open_browser=args.open_browser,
        memory_limit=args.memory_limit,
    )


if __name__ == "__main__":
    main()
