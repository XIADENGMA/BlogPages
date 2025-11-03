#!/usr/bin/env python3
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
通过遥操作控制机器人的脚本 (使用 Web 可视化)。

Simple script to control a robot from teleoperation (with Web visualization).
"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat
import rerun as rr
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.configs import parser
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import (
    Robot,
    RobotConfig,
    bi_so100_follower,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.teleoperators import (
    Teleoperator,
    TeleoperatorConfig,
    bi_so100_leader,
    gamepad,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
)
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging, move_cursor_up
from .web_visualization_utils import init_rerun_web, log_rerun_data


@dataclass
class TeleoperateWebConfig:
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # Limit the maximum frames per second.
    # 限制最大帧率
    fps: int = 60
    teleop_time_s: float | None = None
    # Display all cameras on screen
    # 在屏幕上显示所有摄像头
    display_data: bool = False
    # Web viewer port
    # Web 查看器端口
    web_port: int = 9090
    # Auto open browser
    # 自动打开浏览器
    open_browser: bool = True


def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    display_data: bool = False,
    duration: float | None = None,
):
    """
    This function continuously reads actions from a teleoperation device, processes them through optional
    pipelines, sends them to a robot, and optionally displays the robot's state. The loop runs at a
    specified frequency until a set duration is reached or it is manually interrupted.

    Args:
        teleop: The teleoperator device instance providing control actions.
        robot: The robot instance being controlled.
        fps: The target frequency for the control loop in frames per second.
        display_data: If True, fetches robot observations and displays them in the console and Rerun.
        duration: The maximum duration of the teleoperation loop in seconds. If None, the loop runs indefinitely.
        teleop_action_processor: An optional pipeline to process raw actions from the teleoperator.
        robot_action_processor: An optional pipeline to process actions before they are sent to the robot.
        robot_observation_processor: An optional pipeline to process raw observations from the robot.
    """

    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()

    while True:
        loop_start = time.perf_counter()

        # Get robot observation
        # 获取机器人观测
        obs = robot.get_observation()

        # Get teleop action
        # 获取遥操作动作
        raw_action = teleop.get_action()

        # Process teleop action through pipeline
         # 通过处理流水线处理遥操作动作
        teleop_action = teleop_action_processor((raw_action, obs))

        # Process action for robot through pipeline
         # 通过处理流水线生成发送给机器人的动作
        robot_action_to_send = robot_action_processor((teleop_action, obs))

        # Send processed action to robot
        # 发送处理后的动作给机器人
        _ = robot.send_action(robot_action_to_send)

        if display_data:
            # Process robot observation through pipeline
            # 通过处理流水线处理机器人观测
            obs_transition = robot_observation_processor(obs)

            log_rerun_data(
                observation=obs_transition,
                action=teleop_action,
            )

            print("\n" + "-" * (display_len + 10))  # 分隔线
            print(f"{'NAME':<{display_len}} | {'NORM':>7}")  # 标题：名称与归一化值
            # Display the final robot action that was sent
            # 显示已发送给机器人的最终动作
            for motor, value in robot_action_to_send.items():
                print(f"{motor:<{display_len}} | {value:>7.2f}")  # 每个电机的动作值（保留 2 位小数）
            move_cursor_up(len(robot_action_to_send) + 5)  # 上移光标以覆盖刷新

        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)  # 忙等待以维持目标帧率（若剩余时间为正）
        loop_s = time.perf_counter() - loop_start
        print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")  # 打印单次循环耗时与频率

        if duration is not None and time.perf_counter() - start >= duration:
            return


@parser.wrap()
def teleoperate_web(cfg: TeleoperateWebConfig):
    init_logging()  # 初始化日志
    logging.info(pformat(asdict(cfg)))  # 记录当前配置

    if cfg.display_data:
        # 使用 Web 可视化
        init_rerun_web(
            session_name="teleoperation_web",
            port=cfg.web_port,
            open_browser=cfg.open_browser,
        )

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    teleop.connect()  # 连接遥操作设备
    robot.connect()  # 连接机器人

    try:
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=cfg.fps,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )
    except KeyboardInterrupt:
        pass  # 捕获 Ctrl+C 中断，正常退出
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()  # 关闭 Rerun 会话
        teleop.disconnect()  # 断开遥操作设备
        robot.disconnect()  # 断开机器人


def main():
    teleoperate_web()


if __name__ == "__main__":
    main()
