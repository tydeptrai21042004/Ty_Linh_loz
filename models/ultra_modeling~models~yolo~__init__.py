"""YOLO detection and oriented bounding box tasks."""
from . import detect, obb
from .model import YOLO
__all__ = ("detect", "obb", "YOLO")
