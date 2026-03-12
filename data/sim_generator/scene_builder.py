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
from dataclasses import dataclass
from typing import Optional

import torch


# FSD50K (Freesound Dataset 50K) AudioSet 기반 200개 클래스
# 출처: Fonseca et al., 2022 — https://zenodo.org/record/4060432
SELD_CLASSES = [
    "Accelerating, revving, vroom",
    "Accordion",
    "Acoustic guitar",
    "Aircraft",
    "Air horn, truck horn",
    "Ambulance (siren)",
    "Animal",
    "Applause",
    "Bark",
    "Bass drum",
    "Bass guitar",
    "Bathtub (filling or washing)",
    "Bell",
    "Bicycle",
    "Bicycle bell",
    "Bird",
    "Bird vocalization, bird call, bird song",
    "Boat, Water vehicle",
    "Boiling",
    "Boom",
    "Bouncing",
    "Breathing",
    "Burping, eructation",
    "Bus",
    "Buzz",
    "Camera",
    "Car",
    "Car alarm",
    "Car passing by",
    "Cat",
    "Chatter",
    "Cheering",
    "Chewing, mastication",
    "Child speech, kid speaking",
    "Chirp, tweet",
    "Chuckle, chortle",
    "Church bell",
    "Civil defense siren",
    "Clapping",
    "Clock",
    "Coin (dropping)",
    "Computer keyboard",
    "Conversation",
    "Cough",
    "Cowbell",
    "Crack",
    "Crackle",
    "Crash cymbal",
    "Cricket",
    "Crow",
    "Crowd",
    "Crumpling, crinkling",
    "Crushing",
    "Crying, sobbing",
    "Cupboard open or close",
    "Cutlery, silverware",
    "Cymbal",
    "Dishes, pots, and pans",
    "Dog",
    "Door",
    "Doorbell",
    "Drawer open or close",
    "Drill",
    "Drip",
    "Drum",
    "Drum kit",
    "Electric guitar",
    "Engine",
    "Engine starting",
    "Explosion",
    "Fart",
    "Female singing",
    "Female speech, woman speaking",
    "Fill (with liquid)",
    "Finger snapping",
    "Fire",
    "Fireworks",
    "Fixed-wing aircraft, airplane",
    "Flute",
    "Frog",
    "Frying (food)",
    "Gasp",
    "Giggle",
    "Glass",
    "Glockenspiel",
    "Gong",
    "Growling",
    "Guitar",
    "Gunshot, gunfire",
    "Gurgling",
    "Hammer",
    "Hands",
    "Heart sounds, heartbeat",
    "Helicopter",
    "Hi-hat",
    "Hiss",
    "Human voice",
    "Humming",
    "Insect",
    "Keyboard (musical)",
    "Keys jangling",
    "Knock",
    "Laughter",
    "Lawn mower",
    "Male singing",
    "Male speech, man speaking",
    "Marimba, xylophone",
    "Mechanical fan",
    "Meow",
    "Microwave oven",
    "Motorcycle",
    "Mouse",
    "Music",
    "Musical instrument",
    "Ocean",
    "Organ",
    "Owl",
    "Piano",
    "Pig",
    "Plucked string instrument",
    "Pour",
    "Power tool",
    "Printer",
    "Rain",
    "Ratchet, pawl",
    "Rattle",
    "Rattle (instrument)",
    "Respiratory sounds",
    "Ringtone",
    "Run",
    "Rustling leaves",
    "Saxophone",
    "Sawing",
    "Scissors",
    "Scratching (performance technique)",
    "Screaming",
    "Shatter",
    "Sigh",
    "Singing",
    "Sink (filling or washing)",
    "Siren",
    "Skateboard",
    "Slam",
    "Slap, smack",
    "Sneeze",
    "Snicker",
    "Snoring",
    "Squeak",
    "Stream",
    "Strum",
    "Subway, metro, underground",
    "Swimming",
    "Tabla",
    "Tap",
    "Tearing",
    "Telephone",
    "Telephone bell ringing",
    "Television",
    "Tick",
    "Tick-tock",
    "Toilet flush",
    "Traffic noise, roadway noise",
    "Train",
    "Trumpet",
    "Typewriter",
    "Typing",
    "Ukulele",
    "Vacuum cleaner",
    "Vehicle horn, car horn, honking",
    "Violin, fiddle",
    "Walk, footsteps",
    "Water",
    "Water tap, faucet",
    "Waves, surf",
    "Whispering",
    "Whistle",
    "Wind",
    "Wind instrument, woodwind instrument",
    "Wood",
    "Writing",
    "Yell",
    "Zipper (clothing)",
    "Snare drum",
    "Cello",
    "Trombone",
    "Clarinet",
    "Harmonica",
    "Banjo",
    "Mandolin",
    "Choir",
    "Rapping",
    "Beatboxing",
    "Yodeling",
    "Chant",
    "Thump, thud",
    "Splashing",
    "Wail, moan",
    "Bass",
    "Steel guitar, slide guitar",
    "Mallet percussion",
]  # 총 200개

assert len(SELD_CLASSES) == 200, f"SELD_CLASSES 길이 오류: {len(SELD_CLASSES)}"

# 클래스 이름 → 인덱스 역방향 조회 (빠른 lookup용)
SELD_CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(SELD_CLASSES)}

COMMAND_TEMPLATES = [
    "Pick up the object making the {cls} sound.",
    "Grab the item that is producing {cls}.",
    "Retrieve the object that sounds like {cls}.",
    "Find and pick up the thing making {cls}.",
    "Get the object that is emitting {cls}.",
]


@dataclass
class SceneObject:
    object_id: int
    class_idx: int
    class_name: str
    position_m: tuple[float, float, float]   # (x, y, z) metres, camera frame
    az_rad: float
    el_rad: float


@dataclass
class Scene:
    objects: list[SceneObject]
    target_object_id: int
    language_command: str
    camera_intrinsic: torch.Tensor    # (3, 3)
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

        cx = cy = img_size / 2.0
        self.K = torch.tensor([
            [focal_length,         0.0, cx],
            [        0.0, focal_length, cy],
            [        0.0,         0.0,  1.0],
        ], dtype=torch.float32)

    def build(self, available_class_indices: list[int] | None = None) -> Scene:
        """
        Args:
            available_class_indices: 씬에 등장할 수 있는 클래스 인덱스 목록.
                None이면 전체 200개 클래스 중 랜덤 선택.
                오디오 파일이 있는 클래스만 전달하면 화이트 노이즈 없이 실제 소리만 사용.
        """
        pool = available_class_indices if available_class_indices else list(range(len(SELD_CLASSES)))
        n = random.randint(*self.n_objects_range)
        azimuths = sorted(random.uniform(*self.az_range) for _ in range(n))

        objects = []
        for i, az in enumerate(azimuths):
            el    = random.uniform(*self.el_range)
            depth = random.uniform(*self.depth_range)

            cls_idx  = random.choice(pool)
            cls_name = SELD_CLASSES[cls_idx]

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

        target   = random.choice(objects)
        template = random.choice(COMMAND_TEMPLATES)
        command  = template.format(cls=target.class_name.lower())

        return Scene(
            objects=objects,
            target_object_id=target.object_id,
            language_command=command,
            camera_intrinsic=self.K.clone(),
        )
