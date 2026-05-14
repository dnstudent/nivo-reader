from dataclasses import dataclass

from cv2.typing import Rect
from fancy_dataclass import JSONDataclass


@dataclass
class ClipSizes(JSONDataclass):
    top: int
    bottom: int
    left: int
    right: int


@dataclass
class BoundingBox(JSONDataclass):
    x: int
    y: int
    width: int
    height: int

    @classmethod
    def from_rect(cls, rect: Rect) -> "BoundingBox":
        return cls(x=rect[0], y=rect[1], width=rect[2], height=rect[3])

    def __add__(self, other: "BoundingBox") -> "BoundingBox":
        x = min(self.x, other.x)
        y = min(self.y, other.y)
        return BoundingBox(
            x=x,
            y=y,
            width=max(self.x + self.width, other.x + other.width) - x,
            height=max(self.y + self.height, other.y + other.height) - y,
        )

    def cut_clips(self, clips: ClipSizes) -> "BoundingBox":
        return BoundingBox(
            x=self.x + clips.left,
            y=self.y + clips.top,
            width=self.width - clips.left - clips.right,
            height=self.height - clips.top - clips.bottom,
        )

    def top_slice(self, top_cut_len: int) -> "BoundingBox":
        return BoundingBox(
            x=self.x,
            y=self.y,
            width=self.width,
            height=top_cut_len,
        )

    def left_slice(self, left_cut_len: int) -> "BoundingBox":
        return BoundingBox(
            x=self.x,
            y=self.y,
            width=left_cut_len,
            height=self.height,
        )

    @classmethod
    def merge(cls, boxes: list["BoundingBox"]) -> "BoundingBox | None":
        if len(boxes) == 0:
            return None
        ls = [box.x for box in boxes]
        us = [box.y for box in boxes]
        rs = [box.x + box.width for box in boxes]
        ds = [box.y + box.height for box in boxes]
        bounds = [min(ls), min(us), max(rs), max(ds)]
        return cls(
            x=bounds[0],
            y=bounds[1],
            width=bounds[2] - bounds[0],
            height=bounds[3] - bounds[1],
        )


@dataclass
class RectShape(JSONDataclass):
    width: int
    height: int
