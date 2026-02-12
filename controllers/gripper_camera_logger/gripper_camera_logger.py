"""Panda camera logger controller.
Enables the wrist camera and optionally saves frames to disk.
"""
from __future__ import annotations

import os
from datetime import datetime

from controller import Robot


def main() -> None:
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    camera = robot.getDevice("wrist_camera")
    camera.enable(timestep)

    keyboard = robot.getKeyboard()
    keyboard.enable(timestep)

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    frame = 0
    capturing = False
    while robot.step(timestep) != -1:
        key = keyboard.getKey()
        if key in (ord("S"), ord("s")):
            capturing = True
        elif key in (ord("P"), ord("p")):
            capturing = False

        # Save every 10th frame to reduce disk usage.
        if capturing and frame % 10 == 0:
            filename = f"frame_{frame:06d}.png"
            path = os.path.join(output_dir, filename)
            camera.saveImage(path, 100)
        frame += 1


if __name__ == "__main__":
    main()
