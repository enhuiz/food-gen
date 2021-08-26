import tqdm
import cv2
import numpy as np
import torchzq
from pathlib import Path
from torch.utils.data import ConcatDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


class Runner(torchzq.Runner):
    def __init__(
        self,
        root: Path = Path("data/processed"),
        ds_repeat: int = 100,
        base_size: int = 144,
        crop_size: int = 128,
        demo_every: int = 1000,
        wandb_project: str = "food-gen",
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not root.exists():
            self.preprocess_dataset()

    def create_dataset(self, mode):
        args = self.args

        dataset = ImageFolder(
            root=args.root,
            transform=transforms.Compose(
                [
                    transforms.Resize(args.base_size),
                    transforms.RandomCrop(args.crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize(0.5, 1),
                ]
            ),
        )

        return ConcatDataset([dataset] * self.args.ds_repeat)

    def prepare_batch(self, batch, mode):
        x, _ = batch
        return x.to(self.args.device)

    def preprocess_dataset(self):
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

    @torchzq.command
    def train(self, max_epochs: int = 10, **kwargs):
        super().train(max_epochs=max_epochs, **kwargs)


def main():
    torchzq.start(Runner)
