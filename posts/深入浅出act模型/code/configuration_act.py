#!/usr/bin/env python
# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 以上是版权与开源许可声明，表明本代码遵循 Apache 2.0 许可证

from dataclasses import (
    dataclass,
    field,
)  # dataclass 用于简化配置类的定义，field 可定义默认值/工厂

from lerobot.configs.policies import (
    PreTrainedConfig,
)  # 项目内基类：预训练策略的通用配置抽象
from lerobot.configs.types import (
    NormalizationMode,
)  # 枚举：输入/输出的归一化模式（如 MEAN_STD、MIN_MAX）
from lerobot.optim.optimizers import AdamWConfig  # 优化器配置对象（AdamW 的超参集合）


# 使用注册器将该配置类注册为 "act" 类型，便于通过字符串查找/构造对应配置。
@PreTrainedConfig.register_subclass("act")
@dataclass  # dataclass 会自动生成 __init__/__repr__/__eq__ 等，从字段定义中推导构造参数
class ACTConfig(PreTrainedConfig):
    """Configuration class for the Action Chunking Transformers policy.

    Defaults are configured for training on bimanual Aloha tasks like "insertion" or "transfer".

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_shapes` and 'output_shapes`.

    Notes on the inputs and outputs:
        - Either:
            - At least one key starting with "observation.image is required as an input.
              AND/OR
            - The key "observation.environment_state" is required as input.
        - If there are multiple keys beginning with "observation.images." they are treated as multiple camera
          views. Right now we only support all images having the same shape.
        - May optionally work without an "observation.state" key for the proprioceptive robot state.
        - "action" is required as an output key.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        chunk_size: The size of the action prediction "chunks" in units of environment steps.
        n_action_steps: The number of action steps to run in the environment for one invocation of the policy.
            This should be no greater than the chunk size. For example, if the chunk size size 100, you may
            set this to 50. This would mean that the model predicts 100 steps worth of actions, runs 50 in the
            environment, and throws the other 50 out.
        input_shapes: A dictionary defining the shapes of the input data for the policy. The key represents
            the input data name, and the value is a list indicating the dimensions of the corresponding data.
            For example, "observation.image" refers to an input from a camera with dimensions [3, 96, 96],
            indicating it has three color channels and 96x96 resolution. Importantly, `input_shapes` doesn't
            include batch dimension or temporal dimension.
        output_shapes: A dictionary defining the shapes of the output data for the policy. The key represents
            the output data name, and the value is a list indicating the dimensions of the corresponding data.
            For example, "action" refers to an output shape of [14], indicating 14-dimensional actions.
            Importantly, `output_shapes` doesn't include batch dimension or temporal dimension.
        input_normalization_modes: A dictionary with key representing the modality (e.g. "observation.state"),
            and the value specifies the normalization mode to apply. The two available modes are "mean_std"
            which subtracts the mean and divides by the standard deviation and "min_max" which rescale in a
            [-1, 1] range.
        output_normalization_modes: Similar dictionary as `normalize_input_modes`, but to unnormalize to the
            original scale. Note that this is also used for normalizing the training targets.
        vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
        pretrained_backbone_weights: Pretrained weights from torchvision to initialize the backbone.
            `None` means no pretrained weights.
        replace_final_stride_with_dilation: Whether to replace the ResNet's final 2x2 stride with a dilated
            convolution.
        pre_norm: Whether to use "pre-norm" in the transformer blocks.
        dim_model: The transformer blocks' main hidden dimension.
        n_heads: The number of heads to use in the transformer blocks' multi-head attention.
        dim_feedforward: The dimension to expand the transformer's hidden dimension to in the feed-forward
            layers.
        feedforward_activation: The activation to use in the transformer block's feed-forward layers.
        n_encoder_layers: The number of transformer layers to use for the transformer encoder.
        n_decoder_layers: The number of transformer layers to use for the transformer decoder.
        use_vae: Whether to use a variational objective during training. This introduces another transformer
            which is used as the VAE's encoder (not to be confused with the transformer encoder - see
            documentation in the policy class).
        latent_dim: The VAE's latent dimension.
        n_vae_encoder_layers: The number of transformer layers to use for the VAE's encoder.
        temporal_ensemble_coeff: Coefficient for the exponential weighting scheme to apply for temporal
            ensembling. Defaults to None which means temporal ensembling is not used. `n_action_steps` must be
            1 when using this feature, as inference needs to happen at every step to form an ensemble. For
            more information on how ensembling works, please see `ACTTemporalEnsembler`.
        dropout: Dropout to use in the transformer layers (see code for details).
        kl_weight: The weight to use for the KL-divergence component of the loss if the variational objective
            is enabled. Loss is then calculated as: `reconstruction_loss + kl_weight * kld_loss`.
    """

    # 以上三引号是类的文档字符串（docstring），用于解释该配置类的用途和各参数含义。
    # 文档内提到的 `input_shapes` / `output_shapes` 等键，属于父类/策略使用的约定。

    # Input / output structure.
    # 观测/动作的时序结构配置
    n_obs_steps: int = 1  # 传入策略的观测步数（时间维度），目前实现只支持 1（当前步）
    chunk_size: int = 100  # 一次预测的“动作块”的长度（以环境步数计）
    n_action_steps: int = (
        100  # 每次调用策略实际执行到环境中的动作步数，不能超过 chunk_size
    )

    # 归一化模式的默认映射：按模态选择 NormalizationMode
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,  # 图像模态：减均值/除方差
            "STATE": NormalizationMode.MEAN_STD,  # 状态模态：减均值/除方差
            "ACTION": NormalizationMode.MEAN_STD,  # 动作模态：减均值/除方差（训练目标也会用到）
        }
    )

    # Architecture.
    # Vision backbone.
    vision_backbone: str = (
        "resnet18"  # 图像编码使用的 ResNet 主干名称（需为 torchvision 的 resnet 变体）
    )
    pretrained_backbone_weights: str | None = (
        "ResNet18_Weights.IMAGENET1K_V1"  # 主干的预训练权重标识；None 表示不加载
    )
    replace_final_stride_with_dilation: int = False  # 是否用空洞卷积替换 ResNet 最后一个 2x2 stride（类型标注为 int，但实际布尔使用）
    # Transformer layers.
    pre_norm: bool = False  # Transformer 是否使用 pre-norm 结构（LayerNorm 在子层之前）
    dim_model: int = 512  # Transformer 主通道隐藏维度 d_model
    n_heads: int = 8  # 多头注意力的头数
    dim_feedforward: int = 3200  # 前馈网络的扩展维度（通常为 d_model 的若干倍）
    feedforward_activation: str = "relu"  # 前馈网络的激活函数类型
    n_encoder_layers: int = 4  # Transformer 编码器层数
    # Note: Although the original ACT implementation has 7 for `n_decoder_layers`, there is a bug in the code
    # that means only the first layer is used. Here we match the original implementation by setting this to 1.
    # See this issue https://github.com/tonyzhaozh/act/issues/25#issue-2258740521.
    n_decoder_layers: int = (
        1  # Transformer 解码器层数（按原实现的实际效果设置为 1，以对齐行为）
    )
    # VAE.
    use_vae: bool = (
        True  # 训练时是否使用 VAE 目标（引入额外 Transformer 作为 VAE 编码器）
    )
    latent_dim: int = 32  # VAE 潜变量维度
    n_vae_encoder_layers: int = 4  # VAE 编码器的 Transformer 层数

    # Inference.
    # Note: the value used in ACT when temporal ensembling is enabled is 0.01.
    temporal_ensemble_coeff: float | None = (
        None  # 时间集成（temporal ensembling）的指数加权系数；None 表示关闭
    )

    # Training and loss computation.
    dropout: float = 0.1  # Transformer 层内的 dropout 比例（防止过拟合）
    kl_weight: float = (
        10.0  # 使用 VAE 时，KL 散度项的损失权重（总损失 = 重构损失 + kl_weight * KL）
    )

    # Training preset
    # 训练预设：优化器相关超参数
    optimizer_lr: float = 1e-5  # 主体学习率
    optimizer_weight_decay: float = 1e-4  # 权重衰减（L2 正则）
    optimizer_lr_backbone: float = (
        1e-5  # 视觉主干的学习率（可能与主体不同，用于微调/冻结策略）
    )

    def __post_init__(self):
        # dataclass 的钩子：在 __init__ 之后自动调用。
        # 这里首先调用父类的 __post_init__ 来完成通用初始化（如解析输入/输出特征等）。
        super().__post_init__()

        """Input validation (not exhaustive)."""
        # ——以下是对配置进行基本校验的逻辑（非穷尽）——

        # 校验视觉主干名称：必须是 ResNet 家族，否则抛出 ValueError
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        # 若启用时间集成（temporal_ensemble_coeff 非 None），则 n_action_steps 必须为 1
        # 原因：时间集成需要每一步都查询策略以形成集成
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. This is "
                "because the policy needs to be queried every step to compute the ensembled action."
            )
        # n_action_steps 不能超过 chunk_size（一次调用预测的最大可用步数上限）
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        # 目前实现不支持多观测步（时间窗口 > 1）
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        # 返回一个 AdamW 优化器的配置预设，供上层训练器构造实际优化器实例
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        # 返回 None 表示不使用学习率调度器（或由外部训练脚本自行指定）
        return None

    def validate_features(self) -> None:
        # 检查特征输入是否满足最小要求：
        # 必须至少提供一种图像特征（来自摄像头）或环境状态特征。
        # 这些属性（image_features、env_state_feature）通常在父类 __post_init__ 中解析并赋值。
        if not self.image_features and not self.env_state_feature:
            raise ValueError(
                "You must provide at least one image or the environment state among the inputs."
            )

    @property
    def observation_delta_indices(self) -> None:
        # 观测的“增量索引”定义（若用于计算时间差分等）。这里返回 None，表示不定义/不使用。
        return None

    @property
    def action_delta_indices(self) -> list:
        # 动作的“增量索引”定义：这里返回 [0, 1, ..., chunk_size-1]
        # 常见用途：指示哪些时间步上的动作需要被预测/计算，或用于构建目标序列的索引。
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        # 奖励的“增量索引”定义：此处不使用奖励差分（返回 None）
        return None
