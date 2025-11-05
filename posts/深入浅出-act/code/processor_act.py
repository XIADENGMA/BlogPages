#!/usr/bin/env python
# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 以上是版权与开源许可声明，表明本代码遵循 Apache 2.0 许可证

# 小结：
# - 本文件的核心函数 make_act_pre_post_processors 会根据 ACTConfig 和可选的数据集统计，构造两个流水线对象：
#   1) 前处理流水线（PRE）：重命名 -> 加 batch 维 -> 移到设备 -> 归一化
#   2) 后处理流水线（POST）：反归一化 -> 移回 CPU
# - 这样做的好处：将数据工程与模型推理解耦，保证输入输出的形状、设备与数值尺度都符合模型与下游使用方的预期
# - features 与 norm_map 的一致性非常重要：保证前处理与后处理的变换可逆且匹配


# limitations under the License.
from typing import (
    Any,
)  # 从 typing 导入 Any，表示“任意类型”，常用于类型提示中表示通用容器

import torch  # 导入 PyTorch，用于张量（Tensor）及设备（device）管理
from lerobot.policies.act.configuration_act import ACTConfig  # 导入 ACT 策略的配置类

# 从 lerobot.processor 导入一系列“处理步骤（ProcessorStep）”与管道（Pipeline）相关类
from lerobot.processor import (
    AddBatchDimensionProcessorStep,  # 处理步骤：为输入添加 batch 维度（例如从 [C,H,W] 变为 [B,C,H,W]）
    DeviceProcessorStep,  # 处理步骤：将数据移动到指定设备（如 "cuda:0" 或 "cpu"）
    NormalizerProcessorStep,  # 处理步骤：对特征做归一化（根据统计量，如均值/方差）
    PolicyAction,  # 策略输出动作的数据结构（类型别名/封装）
    PolicyProcessorPipeline,  # 通用的策略处理“流水线”定义，包含一系列有序步骤
    RenameObservationsProcessorStep,  # 处理步骤：重命名观测字典中的键（key），以适配模型预期的输入名
    UnnormalizerProcessorStep,  # 处理步骤：把归一化后的输出反归一化回原始尺度
)
from lerobot.processor.converters import (
    policy_action_to_transition,  # 转换函数：将策略动作结构转为“transition”结构（过渡/样本格式）
    transition_to_policy_action,  # 转换函数：与上相反，将 transition 转回策略动作结构
)
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,  # 常量：后处理流水线的默认命名
    POLICY_PREPROCESSOR_DEFAULT_NAME,  # 常量：前处理流水线的默认命名
)


def make_act_pre_post_processors(
    config: ACTConfig,  # ACT 策略配置对象，内含设备、特征配置、归一化映射关系等信息
    dataset_stats: dict[str, dict[str, torch.Tensor]]
    | None = None,  # 数据集统计信息（如 mean/std），按特征名组织；可为 None
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Creates the pre- and post-processing pipelines for the ACT policy.

    The pre-processing pipeline handles normalization, batching, and device placement for the model inputs.
    The post-processing pipeline handles unnormalization and moves the model outputs back to the CPU.

    Args:
        config (ACTConfig): The ACT policy configuration object.
        dataset_stats (dict[str, dict[str, torch.Tensor]] | None): A dictionary containing dataset
            statistics (e.g., mean and std) used for normalization. Defaults to None.

    Returns:
        tuple[PolicyProcessorPipeline[dict[str, Any], dict[str, Any]], PolicyProcessorPipeline[PolicyAction, PolicyAction]]: A tuple containing the
        pre-processor pipeline and the post-processor pipeline.
    """
    # 上面的英文文档说明：
    # - 本函数构造并返回“前处理（pre）”与“后处理（post）”两个流水线，用于 ACT 策略的输入和输出数据处理。
    # - 前处理：将原始观测做重命名、补齐 batch 维度、移动到指定设备、并按数据集统计进行归一化，使之适配模型输入。
    # - 后处理：对模型输出做反归一化（还原到原尺度），并移动回 CPU（便于后续使用或与非 GPU 代码交互）。
    # - dataset_stats：通常包含每个特征的 mean/std，用于 Normalizer/Unnormalizer；为 None 时可能使用默认策略或跳过部分操作。
    # - 返回值是一个元组：(前处理流水线, 后处理流水线)

    # 定义前处理流水线中包含的“步骤”列表（按顺序执行）
    input_steps = [
        RenameObservationsProcessorStep(
            rename_map={}
        ),  # 重命名观测键的步骤：这里给了空映射，表示当前不需要改名
        AddBatchDimensionProcessorStep(),  # 添加 batch 维：当输入是单样本时，变为批大小为 1 的张量，便于模型统一处理
        DeviceProcessorStep(
            device=config.device
        ),  # 设备迁移：将（可能包含张量的）输入移动到 config.device（如 "cuda" 或 "cpu"）
        NormalizerProcessorStep(  # 归一化步骤：对输入/输出特征集合按 norm_map 和 stats 做标准化/归一化
            features={
                **config.input_features,
                **config.output_features,
            },  # 指定需要归一化的特征集合：将输入与输出特征合并
            norm_map=config.normalization_mapping,  # 指定特征名到“归一化配置/方式”的映射（例如使用哪组统计量）
            stats=dataset_stats,  # 数据集统计（如 mean/std），用于归一化参数
            device=config.device,  # 将统计量和运算放在相同设备上，避免跨设备拷贝/错误
        ),
    ]
    # 以上前处理的意图：
    # 1) RenameObservationsProcessorStep：有些数据集的键名与模型期望不一致，通过重命名统一接口（此处为空映射，意味着保持原样）
    # 2) AddBatchDimensionProcessorStep：即便是单条数据也要添加 batch 维度，满足大多数深度学习模型形状要求
    # 3) DeviceProcessorStep：统一把数据移动到 config 指定的设备（GPU/CPU），确保后续张量运算在同一设备上
    # 4) NormalizerProcessorStep：将输入（甚至包含模型要预测的目标特征）进行标准化，使训练/推理更稳定

    # 定义后处理流水线步骤：将模型输出从标准化空间映射回原空间，并迁移到 CPU
    output_steps = [
        UnnormalizerProcessorStep(  # 反归一化：把模型输出（先前按统计量标准化过）还原到原始数值范围
            features=config.output_features,  # 仅对输出相关的特征进行反归一化（不会动输入特征）
            norm_map=config.normalization_mapping,  # 使用与前处理一致的归一化映射表，保证前后处理对齐
            stats=dataset_stats,  # 使用相同的数据集统计参数进行反变换
        ),
        DeviceProcessorStep(
            device="cpu"
        ),  # 将最终结果统一移回 CPU：便于日志记录、与非 GPU 组件交互或序列化
    ]

    # 返回前/后处理两个 PolicyProcessorPipeline 实例：
    # - 对于前处理流水线：输入与输出都是字典（键到任意类型），因为输入通常是多模态观测的字典结构
    # - 对于后处理流水线：输入与输出是 PolicyAction（策略动作）结构/对象
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,  # 指定流水线包含的步骤序列
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,  # 使用默认的“前处理”名称，便于日志或调试
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,  # 指定后处理步骤序列
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,  # 使用默认的“后处理”名称
            to_transition=policy_action_to_transition,  # 指定如何把 PolicyAction 转换为 transition（内部可能用于统一接口）
            to_output=transition_to_policy_action,  # 指定如何把 transition 转回 PolicyAction（与上相反的方向）
        ),
    )
