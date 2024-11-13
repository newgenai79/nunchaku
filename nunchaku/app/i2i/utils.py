import cv2
import numpy as np
from PIL import Image
import argparse


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--precision", type=str, default="int4", choices=["int4", "bf16"], help="Which precisions to use"
    )
    parser.add_argument("--use-qencoder", action="store_true", help="Whether to use 4-bit text encoder")
    parser.add_argument("--no-safety-checker", action="store_true", help="Disable safety checker")
    parser.add_argument("--count-use", action="store_true", help="Whether to count the number of uses")
    args = parser.parse_args()
    return args
