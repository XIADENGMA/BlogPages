#!/usr/bin/env python
# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
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
# 以上是版权与开源许可声明，表明本代码遵循 Apache 2.0 许可证

"""Action Chunking Transformer Policy

As per Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (https://huggingface.co/papers/2304.13705).
The majority of changes here involve removing unused code, unifying naming, and adding helpful comments.
"""

# 说明：
# 本文件实现 ACT（Action Chunking Transformer）策略与其底层神经网络。ACT 旨在在机器人强化学习/模仿学习中，
# 一次性预测一段连续的动作序列（“动作块”/chunk），以减少每步都要前向一次模型的开销，并改善时序一致性。
# 代码支持两种模式：
#   1) 纯 Transformer 预测动作序列；
#   2) 可选 VAE（变分自编码器）训练目标：使用一个 VAE encoder 产生隐变量，再用 Transformer（此时相当于 VAE 的解码器）预测动作序列。
# 另外支持可选的“时间集成（Temporal Ensembling）”在推理时对多步动作进行指数加权平均，从而平滑并提升鲁棒性。
# 模块结构总览：
# - ACTPolicy：策略封装（选择动作、训练前向、优化参数分组、时间集成/动作队列）
# - ACTTemporalEnsembler：在线指数加权时间集成器
# - ACT：核心模型（视觉骨干网络 + Transformer 编码器/解码器 + 各种投影/位置编码 + 动作回归头）
# - ACTEncoder / ACTDecoder / 对应 Layer：标准 Transformer 层（支持 pre-norm/post-norm）
# - 位置编码（1D/2D 正弦位置编码）
# - 实用函数：get_activation_fn、create_sinusoidal_pos_embedding
#
# 术语与张量形状约定：
# - B: batch size
# - S: 序列长度（这里多指 chunk_size，即要预测的动作步数）
# - D: 隐藏维度（dim_model）
# - L: 潜变量维度（latent_dim）
# - action_dim: 动作维度
# - 图片特征：从视觉骨干网络输出的 feature map (B, C, H, W)，随后会被重排为序列。
# - Transformer 采用 PyTorch 标准接口，序列维度在最前（Seq, Batch, Channel）。
#
# 限制：仅添加注释，不修改任何原始代码逻辑或接口。

import math
from collections import deque
from collections.abc import Callable
from itertools import chain

import einops  # 张量重排工具库，便于通道/维度变换
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from lerobot.policies.act.configuration_act import (
    ACTConfig,
)  # 配置对象，集中管理所有超参数
from lerobot.policies.pretrained import PreTrainedPolicy  # 通用策略基类
from lerobot.utils.constants import (
    ACTION,
    OBS_ENV_STATE,
    OBS_IMAGES,
    OBS_STATE,
)  # 约定的数据字典键名
from torch import Tensor, nn
from torchvision.models._utils import (
    IntermediateLayerGetter,
)  # 从骨干网络中提取中间层输出
from torchvision.ops.misc import (
    FrozenBatchNorm2d,
)  # 冻结的 BN，常用于迁移学习避免数值漂移


class ACTPolicy(PreTrainedPolicy):
    """
    Action Chunking Transformer Policy as per Learning Fine-Grained Bimanual Manipulation with Low-Cost
    Hardware (paper: https://huggingface.co/papers/2304.13705, code: https://github.com/tonyzhaozh/act)
    """

    config_class = ACTConfig
    name = "act"

    def __init__(
        self,
        config: ACTConfig,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """
        super().__init__(config)
        config.validate_features()  # 校验特征配置是否一致/可用（例如是否提供所需的键）
        self.config = config

        self.model = ACT(config)  # 核心模型：视觉 + Transformer

        if config.temporal_ensemble_coeff is not None:
            # 如果配置了时间集成，就用指数加权的在线方法平滑动作序列
            self.temporal_ensembler = ACTTemporalEnsembler(
                config.temporal_ensemble_coeff, config.chunk_size
            )

        self.reset()  # 初始化动作队列或时间集成器状态

    def get_optim_params(self) -> dict:
        # TODO(aliberts, rcadene): As of now, lr_backbone == lr
        # Should we remove this and just `return self.parameters()`?
        # 将参数分组以设置不同学习率（例如对视觉骨干设置较小 LR）
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """This should be called whenever the environment is reset."""
        # 环境重置时需清空时间集成器或动作队列，以免使用旧状态
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            # 无时间集成时，使用一个定长队列缓存已预测的动作块，逐步弹出
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        # 选择动作时确保 eval 模式（禁用 Dropout/BN 统计更新）
        self.eval()  # keeping the policy in eval mode as it could be set to train mode while queue is consumed

        if self.config.temporal_ensemble_coeff is not None:
            # 使用时间集成：每次对最新的动作块进行在线融合，并返回当前应执行的单步动作
            actions = self.predict_action_chunk(batch)
            action = self.temporal_ensembler.update(actions)
            return action

        # 无时间集成：维护一个动作队列（n_action_steps 个），当队列为空时，再预测新的动作块填充
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]

            # 模型输出形状为 (B, n_action_steps, action_dim)，而队列按时间步推进，等价 (n_action_steps, B, *)
            # 因此需要转置再按时间步扩展到队列
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        if self.config.image_features:
            # 若配置了图像特征，将用户提供的多路图像拼到统一键 OBS_IMAGES 下（浅拷贝避免改动原 batch）
            batch = dict(
                batch
            )  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions = self.model(batch)[0]  # 仅取预测的动作（忽略 VAE 参数返回）
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""
        # 训练/验证前向：输出预测动作与损失
        if self.config.image_features:
            batch = dict(
                batch
            )  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        # L1 行为克隆损失（对 padding 掩码为 True 的时间步不计入损失）
        l1_loss = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none")
            * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae:
            # 当使用 VAE 目标时，额外计算 KL 散度（对潜变量逐维求和，再对 batch 求均值）
            # log_sigma_x2 是 2*log(sigma)，保持与原实现一致
            mean_kld = (
                (
                    -0.5
                    * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())
                )
                .sum(-1)
                .mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        return loss, loss_dict


class ACTTemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        """Temporal ensembling as described in Algorithm 2 of https://huggingface.co/papers/2304.13705.

        The weights are calculated as wᵢ = exp(-temporal_ensemble_coeff * i) where w₀ is the oldest action.
        They are then normalized to sum to 1 by dividing by Σwᵢ. Here's some intuition around how the
        coefficient works:
            - Setting it to 0 uniformly weighs all actions.
            - Setting it positive gives more weight to older actions.
            - Setting it negative gives more weight to newer actions.
        NOTE: The default value for `temporal_ensemble_coeff` used by the original ACT work is 0.01. This
        results in older actions being weighed more highly than newer actions (the experiments documented in
        https://github.com/huggingface/lerobot/pull/319 hint at why highly weighing new actions might be
        detrimental: doing so aggressively may diminish the benefits of action chunking).

        Here we use an online method for computing the average rather than caching a history of actions in
        order to compute the average offline. For a simple 1D sequence it looks something like:

        ```
        import torch

        seq = torch.linspace(8, 8.5, 100)
        print(seq)

        m = 0.01
        exp_weights = torch.exp(-m * torch.arange(len(seq)))
        print(exp_weights)

        # Calculate offline
        avg = (exp_weights * seq).sum() / exp_weights.sum()
        print("offline", avg)

        # Calculate online
        for i, item in enumerate(seq):
            if i == 0:
                avg = item
                continue
            avg *= exp_weights[:i].sum()
            avg += item * exp_weights[i]
            avg /= exp_weights[: i + 1].sum()
        print("online", avg)
        ```
        """
        # 中文补充：时间集成器对一个动作块内的每个时间步位置 i（0 为最旧）分配权重 w_i = exp(-m * i)。
        # 在线更新避免缓存历史所有动作，大幅节省内存/计算，适合推理时逐步滑动窗口融合。
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(
            -temporal_ensemble_coeff * torch.arange(chunk_size)
        )
        # 累积和用于在线“归一化”更新
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        """Resets the online computation variables."""
        # 清零内部缓存：当前融合的动作序列与对应的计数（每个时间步融合了多少次）
        self.ensembled_actions = None
        # (chunk_size,) count of how many actions are in the ensemble for each time step in the sequence.
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        """
        Takes a (batch, chunk_size, action_dim) sequence of actions, update the temporal ensemble for all
        time steps, and pop/return the next batch of actions in the sequence.
        """
        # 将权重张量放到与输入相同的 device 上（CPU/GPU 兼容）
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(
            device=actions.device
        )
        if self.ensembled_actions is None:
            # 第一次调用：直接把预测的动作块克隆为当前融合序列
            self.ensembled_actions = actions.clone()
            # 记录每个时间步目前的“融合次数”=1（形状对齐为 (S,1)，便于广播）
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1),
                dtype=torch.long,
                device=self.ensembled_actions.device,
            )
        else:
            # 对已有的融合序列（除了最后一个时间步）进行在线更新：
            # old_avg * sum(w[:i]) + new * w[i] 再除以 sum(w[:i+1])，形如“带权平均”的递推公式
            self.ensembled_actions *= self.ensemble_weights_cumsum[
                self.ensembled_actions_count - 1
            ]
            self.ensembled_actions += (
                actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            )
            self.ensembled_actions /= self.ensemble_weights_cumsum[
                self.ensembled_actions_count
            ]
            # 融合计数自增，封顶为 chunk_size
            self.ensembled_actions_count = torch.clamp(
                self.ensembled_actions_count + 1, max=self.chunk_size
            )
            # 将“最新一步”的原始动作直接拼到末尾（该位置没有历史平均）
            self.ensembled_actions = torch.cat(
                [self.ensembled_actions, actions[:, -1:]], dim=1
            )
            # 对应的计数也拼接 1
            self.ensembled_actions_count = torch.cat(
                [
                    self.ensembled_actions_count,
                    torch.ones_like(self.ensembled_actions_count[-1:]),
                ]
            )
        # 消费/弹出融合序列的第一个动作（当前要执行的动作），并滑动窗口
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action


class ACT(nn.Module):
    """Action Chunking Transformer: The underlying neural network for ACTPolicy.

    Note: In this code we use the terms `vae_encoder`, 'encoder', `decoder`. The meanings are as follows.
        - The `vae_encoder` is, as per the literature around variational auto-encoders (VAE), the part of the
          model that encodes the target data (a sequence of actions), and the condition (the robot
          joint-space).
        - A transformer with an `encoder` (not the VAE encoder) and `decoder` (not the VAE decoder) with
          cross-attention is used as the VAE decoder. For these terms, we drop the `vae_` prefix because we
          have an option to train this model without the variational objective (in which case we drop the
          `vae_encoder` altogether, and nothing about this model has anything to do with a VAE).

                                 Transformer
                                 Used alone for inference
                                 (acts as VAE decoder
                                  during training)
                                ┌───────────────────────┐
                                │             Outputs   │
                                │                ▲      │
                                │     ┌─────►┌───────┐  │
                   ┌──────┐     │     │      │Transf.│  │
                   │      │     │     ├─────►│decoder│  │
              ┌────┴────┐ │     │     │      │       │  │
              │         │ │     │ ┌───┴───┬─►│       │  │
              │ VAE     │ │     │ │       │  └───────┘  │
              │ encoder │ │     │ │Transf.│             │
              │         │ │     │ │encoder│             │
              └───▲─────┘ │     │ │       │             │
                  │       │     │ └▲──▲─▲─┘             │
                  │       │     │  │  │ │               │
                inputs    └─────┼──┘  │ image emb.      │
                                │    state emb.         │
                                └───────────────────────┘
    """

    def __init__(self, config: ACTConfig):
        # BERT 风格的 VAE 编码器输入： [CLS, 机器人当前关节状态（可选）, 动作序列]。
        # CLS token 经过投影后输出潜变量分布参数（mean 与 log_sigma_x2）。
        super().__init__()
        self.config = config

        if self.config.use_vae:
            # VAE 编码器（仅在使用 VAE 目标且训练阶段时启用）
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            # 机器人关节状态投影到 Transformer 隐藏维度
            if self.config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], config.dim_model
                )
            # 动作（目标关节位姿/速度等）投影到隐藏维度
            self.vae_encoder_action_input_proj = nn.Linear(
                self.config.action_feature.shape[0],
                config.dim_model,
            )
            # 将 VAE 编码器的 CLS 输出映射为潜变量分布参数（均值和对数方差 * 2）
            self.vae_encoder_latent_output_proj = nn.Linear(
                config.dim_model, config.latent_dim * 2
            )
            # 固定正弦位置编码（1D），长度 = 1(CLS) + S(动作步) + [1(关节状态，可选)]
            num_input_token_encoder = 1 + config.chunk_size
            if self.config.robot_state_feature:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(
                    num_input_token_encoder, config.dim_model
                ).unsqueeze(0),
            )

        # 视觉骨干网络（例如 ResNet），用于提取图像特征
        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[
                    False,
                    False,
                    config.replace_final_stride_with_dilation,
                ],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            # 使用 IntermediateLayerGetter 从指定层（这里为 layer4）获取特征图
            # 输出为字典 {"feature_map": output}
            self.backbone = IntermediateLayerGetter(
                backbone_model, return_layers={"layer4": "feature_map"}
            )

        # Transformer：在使用 VAE 时相当于解码器（cross-attend 到条件输入）
        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        # Transformer 编码器的输入投影与位置编码：
        # token 顺序：[latent, (robot_state), (env_state), (image_feature_map_pixels...)]
        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                self.config.env_state_feature.shape[0], config.dim_model
            )
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        if self.config.image_features:
            # 将骨干网络的通道维投影到 dim_model（1x1 卷积作为线性投影）
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )
        # 1D token（latent/robot_state/env_state）的可学习位置嵌入
        n_1d_tokens = 1  # latent
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.config.image_features:
            # 2D 正弦位置编码（对 feature map 的每个像素提供位置信息）
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(
                config.dim_model // 2
            )

        # Transformer 解码器：为 S 个要预测的时间步提供可学习查询（DETR 风格）
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # 最终线性头：将 decoder 输出映射为动作维度
        self.action_head = nn.Linear(
            config.dim_model, self.config.action_feature.shape[0]
        )

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        # 对 Transformer 层进行 Xavier 均匀初始化，提升训练稳定性
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """A forward pass through the Action Chunking Transformer (with optional VAE encoder).

        `batch` should have the following structure:
        {
            [robot_state_feature] (optional): (B, state_dim) batch of robot states.

            [image_features]: (B, n_cameras, C, H, W) batch of images.
                AND/OR
            [env_state_feature]: (B, env_dim) batch of environment states.

            [action_feature] (optional, only if training with VAE): (B, chunk_size, action dim) batch of actions.
        }

        Returns:
            (B, chunk_size, action_dim) batch of action sequences
            Tuple containing the latent PDF's parameters (mean, log(σ²)) both as (B, L) tensors where L is the
            latent dimension.
        """
        # 当使用 VAE + 训练模式时，要求 batch 中必须包含监督的动作序列 ACTION
        if self.config.use_vae and self.training:
            assert ACTION in batch, (
                "actions must be provided when using the variational objective in training mode."
            )

        # 估计 batch 大小：优先从图像，若无图像则从环境状态推断
        batch_size = (
            batch[OBS_IMAGES][0].shape[0]
            if OBS_IMAGES in batch
            else batch[OBS_ENV_STATE].shape[0]
        )

        # 1) 准备潜变量（latent）
        if self.config.use_vae and ACTION in batch and self.training:
            # 训练 + VAE：通过 VAE encoder 从 [CLS, 关节状态(可选), 动作序列] 推断潜变量分布参数
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )  # (B, 1, D)
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(
                    batch[OBS_STATE]
                )
                robot_state_embed = robot_state_embed.unsqueeze(1)  # (B, 1, D)
            action_embed = self.vae_encoder_action_input_proj(
                batch[ACTION]
            )  # (B, S, D)

            if self.config.robot_state_feature:
                vae_encoder_input = [
                    cls_embed,
                    robot_state_embed,
                    action_embed,
                ]  # (B, S+2, D)
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            # 固定位置编码（与原实现保持一致，使用 clone().detach()）
            pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, S+2, D)

            # key_padding_mask：前面 CLS 和关节状态不是 padding，后面根据 action_is_pad 指示
            # False 表示不是 pad；形状 (B, S+1 或 S+2)
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch[OBS_STATE].device,
            )
            key_padding_mask = torch.cat(
                [cls_joint_is_pad, batch["action_is_pad"]], axis=1
            )  # (bs, seq+1 or 2)

            # 送入 VAE 编码器，取 CLS 位置的输出（包含全序列信息），再映射得到 (mu, log_sigma_x2)
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]  # select the class token, with shape (B, D)
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            # 注意：这里返回的是 2*log(sigma)，与原实现一致
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]

            # 重参数化采样 latent： z = mu + sigma * eps
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            # 推理或未使用 VAE：latent 设为全零（代表“无信息”的先验）
            mu = log_sigma_x2 = None
            # TODO(rcadene, alexander-soare): remove call to `.to` to speedup forward ; precompute and use buffer
            latent_sample = torch.zeros(
                [batch_size, self.config.latent_dim], dtype=torch.float32
            ).to(batch[OBS_STATE].device)

        # 2) 准备 Transformer 编码器输入 token 与位置编码
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        # 1D token 的可学习位置嵌入（先堆为 list，后面与图像 token 一起 stack）
        encoder_in_pos_embed = list(
            self.encoder_1d_feature_pos_embed.weight.unsqueeze(1)
        )
        # 机器人关节状态 token
        if self.config.robot_state_feature:
            encoder_in_tokens.append(
                self.encoder_robot_state_input_proj(batch[OBS_STATE])
            )
        # 环境状态 token
        if self.config.env_state_feature:
            encoder_in_tokens.append(
                self.encoder_env_state_input_proj(batch[OBS_ENV_STATE])
            )

        if self.config.image_features:
            # 多相机图像：各自经过骨干提特征，再通过 1x1 conv 投影至 dim_model
            # 注意：对 MPS 设备做过数值稳定性注意（保持与原实现的注释一致）
            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)[
                    "feature_map"
                ]  # (B, C_backbone, H, W)
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(
                    dtype=cam_features.dtype
                )
                cam_features = self.encoder_img_feat_input_proj(
                    cam_features
                )  # -> (B, D, H, W)

                # 重排为 (Seq, B, D)，其中 Seq = H*W
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                # 直接 extend（列表形式）以避免先积累后再 concat 的额外开销
                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        # 将所有 token 按序列维 stack 成张量：(ES, B, D)
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        # 3) 经过 Transformer 编码器/解码器
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        # 解码器输入初始化为全零（S, B, D），再加上可学习的 decoder_pos_embed 作为查询
        # TODO(rcadene, alexander-soare): remove call to `device` ; precompute and use buffer
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        # (S, B, D) -> (B, S, D)
        decoder_out = decoder_out.transpose(0, 1)

        # 动作回归头：(B, S, D) -> (B, S, action_dim)
        actions = self.action_head(decoder_out)

        # 返回动作与（可选）VAE 参数（训练使用）
        return actions, (mu, log_sigma_x2)


class ACTEncoder(nn.Module):
    """Convenience module for running multiple encoder layers, maybe followed by normalization."""

    # 一个封装的 Transformer Encoder 堆叠模块，支持 pre-norm 配置，便于复用（包括作为 VAE encoder）

    def __init__(self, config: ACTConfig, is_vae_encoder: bool = False):
        super().__init__()
        self.is_vae_encoder = is_vae_encoder
        num_layers = (
            config.n_vae_encoder_layers
            if self.is_vae_encoder
            else config.n_encoder_layers
        )
        self.layers = nn.ModuleList(
            [ACTEncoderLayer(config) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(
        self,
        x: Tensor,
        pos_embed: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        # 逐层前向；支持外部传入位置编码与 padding 掩码
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x


class ACTEncoderLayer(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            config.dim_model, config.n_heads, dropout=config.dropout
        )

        # 前馈网络 FFN：Linear -> 激活 -> Dropout -> Linear
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        # 残差层归一化（支持 pre-norm 或 post-norm）
        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(
        self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None
    ) -> Tensor:
        # 自注意力子层
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
        x = x[0]  # note: [0] to select just the output, not the attention weights
        x = skip + self.dropout1(x)  # 残差

        # 前馈子层
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x


class ACTDecoder(nn.Module):
    def __init__(self, config: ACTConfig):
        """Convenience module for running multiple decoder layers followed by normalization."""
        super().__init__()
        self.layers = nn.ModuleList(
            [ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)]
        )
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        # 逐层 Decoder，包含自注意力与跨注意力（对 encoder_out 进行 cross-attention）
        for layer in self.layers:
            x = layer(
                x,
                encoder_out,
                decoder_pos_embed=decoder_pos_embed,
                encoder_pos_embed=encoder_pos_embed,
            )
        if self.norm is not None:
            x = self.norm(x)
        return x


class ACTDecoderLayer(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            config.dim_model, config.n_heads, dropout=config.dropout
        )
        self.multihead_attn = nn.MultiheadAttention(
            config.dim_model, config.n_heads, dropout=config.dropout
        )

        # FFN
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        # 三个归一化/Dropout 对应三处残差连接
        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        # 若提供了位置编码，则与输入相加（Transformer 常见用法）
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: (Decoder Sequence, Batch, Channel) tensor of input tokens.
            encoder_out: (Encoder Sequence, B, C) output features from the last layer of the encoder we are
                cross-attending with.
            encoder_pos_embed: (ES, 1, C) positional embedding for keys (from the encoder).
            decoder_pos_embed: (DS, 1, C) positional embedding for the queries (from the decoder).
        Returns:
            (DS, B, C) tensor of decoder output features.
        """
        # 1) 自注意力（Decoder 内部 token 之间交互）
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[
            0
        ]  # select just the output, not the attention weights
        x = skip + self.dropout1(x)

        # 2) 跨注意力（对 Encoder 输出进行查询）
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]  # select just the output, not the attention weights
        x = skip + self.dropout2(x)

        # 3) FFN
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal positional embeddings as in Attention is All You Need.

    Args:
        num_positions: Number of token positions required.
    Returns: (num_positions, dimension) position embeddings (the first dimension is the batch dimension).

    """
    # 标准的 1D 正弦/余弦位置编码实现，频率按几何级数变化（温度=10000），偶数维使用正弦，奇数维使用余弦。

    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / dimension)
            for hid_j in range(dimension)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(num_positions)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.from_numpy(sinusoid_table).float()


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings similar to what's presented in Attention Is All You Need.

    The variation is that the position indices are normalized in [0, 2π] (not quite: the lower bound is 1/H
    for the vertical direction, and 1/W for the horizontal direction.
    """

    # 为 2D 特征图（H,W）生成二维正弦位置编码。与常见实现不同，位置索引被缩放到 [0, 2π] 区间（近似），
    # 然后同样以几何级数频率生成正/余弦分量，最后在通道维上拼接（y 分量在前，x 分量在后）。

    def __init__(self, dimension: int):
        """
        Args:
            dimension: The desired dimension of the embeddings.
        """
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        # 频率几何级数的“温度”（与 1D 的 10000 一致）
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: A (B, C, H, W) batch of 2D feature map to generate the embeddings for.
        Returns:
            A (1, C, H, W) batch of corresponding sinusoidal positional embeddings.
        """
        # 仅需 H、W 形状，因此构造一个形状 (1, H, W) 的“非 mask”
        not_mask = torch.ones_like(x[0, :1])  # (1, H, W)
        # y/x 方向的累计和相当于 1..H 与 1..W（原实现从 1 开始，而不是 0）
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        # 归一化到 [0, 2π]（加入 eps 避免分母为 0）
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        # 频率几何序列（偶数/奇数通道分别对应 sin/cos）
        inverse_frequency = self._temperature ** (
            2
            * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2)
            / self.dimension
        )

        # 扩展最后一维以按通道除以频率：(1, H, W, 1)
        x_range = x_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)
        y_range = y_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)

        # 交错堆叠 sin/cos，并在通道维上展平：(1, H, W, C//2)
        pos_embed_x = torch.stack(
            (x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1
        ).flatten(3)
        pos_embed_y = torch.stack(
            (y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1
        ).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(
            0, 3, 1, 2
        )  # (1, C, H, W)

        return pos_embed


def get_activation_fn(activation: str) -> Callable:
    """Return an activation function given a string."""
    # 将字符串名称映射到对应的激活函数实现
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
