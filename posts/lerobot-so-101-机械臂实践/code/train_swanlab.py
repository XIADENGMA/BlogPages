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
SwanLab 训练配置扩展 - 扩展 TrainPipelineConfig 以支持 SwanLab

这个模块为 LeRobot 训练管道提供 SwanLab 日志记录支持。
它继承了原始的 TrainPipelineConfig，并添加了 tracker 和 swanlab 配置字段。
"""

from dataclasses import dataclass, field

from lerobot.configs.train import TrainPipelineConfig
from lerobot.extra.default_swanlab import SwanLabConfig


@dataclass
class TrainPipelineSwanLabConfig(TrainPipelineConfig):
    """支持 SwanLab 的训练管道配置

    继承自 TrainPipelineConfig，添加了以下字段：
        tracker: 日志跟踪器选择，可选值: 'wandb', 'swanlab', 'both', 'none'
        swanlab: SwanLab 配置对象

    使用示例：
        ```python
        config = TrainPipelineSwanLabConfig(
            dataset=DatasetConfig(repo_id="my/dataset"),
            tracker="swanlab",
            swanlab=SwanLabConfig(project="my-project", mode="cloud"),
        )
        ```
    """

    # Tracker selection: 'wandb', 'swanlab', 'both', or 'none'
    # 跟踪器选择: 'wandb', 'swanlab', 'both', 或 'none'
    tracker: str = "wandb"

    # SwanLab configuration
    # SwanLab 配置
    swanlab: SwanLabConfig = field(default_factory=SwanLabConfig)
