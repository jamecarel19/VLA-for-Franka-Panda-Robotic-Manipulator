"""Workspace camera logger controller.
Enables the workspace camera and optionally saves frames to disk.
"""
from __future__ import annotations

import os

from controller import Robot


def main() -> None:
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    camera = robot.getDevice("workspace_camera")
    camera.enable(timestep)

    keyboard = robot.getKeyboard()
    keyboard.enable(timestep)

    controller_dir = os.getcwd()

    output_root = os.path.join(controller_dir, "output")
    os.makedirs(output_root, exist_ok=True)

    state_path = os.path.join(controller_dir, "..", "capture_state.txt")
    last_frame = None

    width = camera.getWidth()
    height = camera.getHeight()

    def save_ppm(path: str) -> None:
        image = camera.getImage()
        if image is None:
            print(f"workspace capture failed: {path}")
            return
        with open(path, "wb") as handle:
            handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
            for y in range(height):
                for x in range(width):
                    r = camera.imageGetRed(image, width, x, y)
                    g = camera.imageGetGreen(image, width, x, y)
                    b = camera.imageGetBlue(image, width, x, y)
                    handle.write(bytes((r, g, b)))

    frame = 0
    capturing = False
    while robot.step(timestep) != -1:
        key = keyboard.getKey()
        if key in (ord("S"), ord("s")):
            capturing = True
        elif key in (ord("P"), ord("p")):
            capturing = False
        elif key in (ord("C"), ord("c")):
            capturing = not capturing

        # Sync with gripper capture state if available.
        if os.path.exists(state_path):
            try:
                with open(state_path, "r", encoding="utf-8") as handle:
                    lines = {line.split("=", 1)[0]: line.split("=", 1)[1].strip() for line in handle if "=" in line}
                capturing = lines.get("capturing", "0") == "1"
                state_frame = int(lines.get("frame", "-1"))
            except Exception:
                capturing = False
                state_frame = -1
        else:
            state_frame = -1

        if capturing and state_frame >= 0 and state_frame != last_frame:
            filename = f"frame_{state_frame:06d}.ppm"
            path = os.path.join(output_root, filename)
            save_ppm(path)
            last_frame = state_frame

        frame += 1


if __name__ == "__main__":
    main()
