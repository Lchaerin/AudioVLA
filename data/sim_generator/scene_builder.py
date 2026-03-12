"""
Simulated scene builder for Audio-VLA data generation.

Generates synthetic scenes with:
  - N objects placed on a table
  - Each object assigned a sound class and audio clip
  - Camera intrinsic/extrinsic parameters
  - Ground-truth target specification (which object the language command refers to)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

import torch


SELD_CLASSES = [
    "Female speech",
    "Male speech",
    "Clapping",
    "Telephone",
    "Laughter",
    "Domestic sounds",
    "Walk, footsteps",
    "Door, open or close",
    "Music",
    "Musical instrument",
    "Water tap, faucet",
    "Bell",
    "Knock",
]

COMMAND_TEMPLATES = [
    "Pick up the object making the {cls} sound.",
    "Grab the item that is {cls}.",
    "Retrieve the object producing {cls}.",
    "Find and pick up the thing that sounds like {cls}.",
]


@dataclass
class SceneObject:
    object_id: int
    class_idx: int
    class_name: str
    position_m: tuple[float, float, float]   # (x, y, z) in metres, camera frame
    az_rad: float
    el_rad: float


@dataclass
class Scene:
    objects: list[SceneObject]
    target_object_id: int
    language_command: str
    camera_intrinsic: torch.Tensor   # (3, 3)
    camera_extrinsic: Optional[torch.Tensor] = None  # (4, 4)

    @property
    def target_object(self) -> SceneObject:
        return next(o for o in self.objects if o.object_id == self.target_object_id)


class SceneBuilder:
    """
    Randomly build scenes for simulation data generation.

    Objects are placed in front of the camera at varying azimuths/elevations.
    """

    def __init__(
        self,
        n_objects_range: tuple[int, int] = (2, 6),
        az_range_deg:    tuple[float, float] = (-60.0, 60.0),
        el_range_deg:    tuple[float, float] = (-20.0, 20.0),
        depth_range_m:   tuple[float, float] = (0.5, 1.5),
        img_size: int = 512,
        focal_length: float = 512.0,
    ):
        self.n_objects_range = n_objects_range
        self.az_range = tuple(math.radians(d) for d in az_range_deg)
        self.el_range = tuple(math.radians(d) for d in el_range_deg)
        self.depth_range = depth_range_m
        self.img_size = img_size
        self.focal_length = focal_length

        # Build camera intrinsic (pinhole)
        cx = cy = img_size / 2.0
        self.K = torch.tensor([
            [focal_length,         0.0, cx],
            [        0.0, focal_length, cy],
            [        0.0,         0.0,  1.0],
        ], dtype=torch.float32)

    def build(self) -> Scene:
        n = random.randint(*self.n_objects_range)
        # Avoid placing two objects at the same azimuth
        azimuths = sorted(random.uniform(*self.az_range) for _ in range(n))

        objects = []
        for i, az in enumerate(azimuths):
            el    = random.uniform(*self.el_range)
            depth = random.uniform(*self.depth_range)

            cls_idx  = random.randint(0, len(SELD_CLASSES) - 1)
            cls_name = SELD_CLASSES[cls_idx]

            # 3D position in camera frame
            x = depth * math.sin(az) * math.cos(el)
            y = -depth * math.sin(el)
            z = depth * math.cos(az) * math.cos(el)

            objects.append(SceneObject(
                object_id=i,
                class_idx=cls_idx,
                class_name=cls_name,
                position_m=(x, y, z),
                az_rad=az,
                el_rad=el,
            ))

        # Choose target
        target = random.choice(objects)
        template = random.choice(COMMAND_TEMPLATES)
        command  = template.format(cls=target.class_name.lower())

        return Scene(
            objects=objects,
            target_object_id=target.object_id,
            language_command=command,
            camera_intrinsic=self.K.clone(),
        )
