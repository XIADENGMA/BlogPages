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

"""Web-based visualization utilities using Rerun web viewer."""

import numbers
import os
from typing import Any

import numpy as np
import rerun as rr

from lerobot.utils.constants import OBS_PREFIX, OBS_STR


def init_rerun_web(
    session_name: str = "lerobot_control_loop",
    port: int = 9090,
    open_browser: bool = True,
    memory_limit: str = "10%",
) -> None:
    """
    初始化 Rerun SDK 用于 Web 浏览器可视化控制循环。

    Initializes the Rerun SDK for visualizing the control loop in a web browser.

    参数 Args:
        session_name: Rerun 会话名称 / Name of the Rerun session
        port: Web 服务器端口 / Web server port (default: 9090)
        open_browser: 是否自动打开浏览器 / Whether to automatically open browser (default: True)
        memory_limit: 内存限制 / Memory limit for Rerun (default: "10%")

    使用示例 Example:
        ```python
        from lerobot.extra.web_visualization_utils import init_rerun_web

        # 在浏览器中启动可视化 / Start visualization in browser
        init_rerun_web(session_name="my_teleoperation", port=9090)
        ```

    访问 Access:
        在浏览器中打开 / Open in browser: http://localhost:9090
    """
    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size

    rr.init(session_name)

    print("🌐 启动 Rerun Web Viewer / Starting Rerun Web Viewer")
    print(f"📍 访问地址 / Access URL: http://localhost:{port}")
    print(f"💾 内存限制 / Memory limit: {memory_limit}")

    rr.serve(
        open_browser=open_browser,
        web_port=port,
        server_memory_limit=memory_limit,
    )


def init_rerun_connect(
    addr: str = "127.0.0.1:9876",
    session_name: str = "lerobot_control_loop",
) -> None:
    """
    连接到远程 Rerun Viewer (适用于远程服务器)。

    Connect to a remote Rerun Viewer (useful for remote servers).

    参数 Args:
        addr: 远程 Rerun Viewer 地址 / Remote Rerun Viewer address (default: "127.0.0.1:9876")
        session_name: Rerun 会话名称 / Name of the Rerun session

    使用示例 Example:
        ```python
        from lerobot.extra.web_visualization_utils import init_rerun_connect

        # 连接到远程 Rerun Viewer / Connect to remote Rerun Viewer
        # 首先在另一个终端运行: rerun --port 9876
        # First run in another terminal: rerun --port 9876
        init_rerun_connect(addr="127.0.0.1:9876")
        ```
    """
    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size

    rr.init(session_name)

    print("🔗 连接到远程 Rerun Viewer / Connecting to remote Rerun Viewer")
    print(f"📍 地址 / Address: {addr}")

    rr.connect_tcp(addr=addr)


def _is_scalar(x):
    return isinstance(x, (float | numbers.Real | np.integer | np.floating)) or (
        isinstance(x, np.ndarray) and x.ndim == 0
    )


def log_rerun_data(
    observation: dict[str, Any] | None = None,
    action: dict[str, Any] | None = None,
) -> None:
    """
    将观测和动作数据记录到 Rerun 用于实时可视化。

    Logs observation and action data to Rerun for real-time visualization.

    This function iterates through the provided observation and action dictionaries and sends their contents
    to the Rerun viewer. It handles different data types appropriately:
    - Scalar values (floats, ints) are logged as `rr.Scalar`.
    - 3D NumPy arrays that resemble images (e.g., with 1, 3, or 4 channels first) are transposed
      from CHW to HWC format and logged as `rr.Image`.
    - 1D NumPy arrays are logged as a series of individual scalars, with each element indexed.
    - Other multi-dimensional arrays are flattened and logged as individual scalars.

    Keys are automatically namespaced with "observation." or "action." if not already present.

    参数 Args:
        observation: 包含观测数据的可选字典 / An optional dictionary containing observation data to log.
        action: 包含动作数据的可选字典 / An optional dictionary containing action data to log.
    """
    if observation:
        for k, v in observation.items():
            if v is None:
                continue
            key = k if str(k).startswith(OBS_PREFIX) else f"{OBS_STR}.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalar(float(v)))
            elif isinstance(v, np.ndarray):
                arr = v
                # Convert CHW -> HWC when needed
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.ndim == 1:
                    for i, vi in enumerate(arr):
                        rr.log(f"{key}_{i}", rr.Scalar(float(vi)))
                else:
                    rr.log(key, rr.Image(arr), static=True)

    if action:
        for k, v in action.items():
            if v is None:
                continue
            key = k if str(k).startswith("action.") else f"action.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalar(float(v)))
            elif isinstance(v, np.ndarray):
                if v.ndim == 1:
                    for i, vi in enumerate(v):
                        rr.log(f"{key}_{i}", rr.Scalar(float(vi)))
                else:
                    # Fall back to flattening higher-dimensional arrays
                    flat = v.flatten()
                    for i, vi in enumerate(flat):
                        rr.log(f"{key}_{i}", rr.Scalar(float(vi)))
