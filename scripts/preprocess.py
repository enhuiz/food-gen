import tqdm
import numpy as np
import cv2
import os
from pathlib import Path
from functools import partial

import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def main():
    root = Path("data/raw")

    def slice_iter():
        stride = 167
        size = 163
        for path in sorted(root.glob("*.jpg")):
            image = cv2.imread(str(path))
            image = np.pad(image, ((0, 1), (0, 0), (0, 0)))
            for i in range(0, image.shape[0], stride):
                for j in range(0, image.shape[1], stride):
                    yield image[i : i + stride, j : j + stride][:size, :size]

    outdir = Path("data/processed/0")
    outdir.mkdir(parents=True, exist_ok=True)
    for i, slice in enumerate(tqdm.tqdm(slice_iter())):
        cv2.imwrite(str(Path(outdir, f"{i:06d}.png")), slice)


if __name__ == "__main__":
    main()
