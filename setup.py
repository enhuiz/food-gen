import subprocess
from pathlib import Path
from datetime import datetime
from setuptools import setup, find_packages


def shell(*args):
    out = subprocess.check_output(args)
    return out.decode("ascii").strip()


def write_version(version_core, pre_release=True):
    if pre_release:
        time = shell("git", "log", "-1", "--format=%cd", "--date=iso")
        time = datetime.strptime(time, "%Y-%m-%d %H:%M:%S %z")
        time = time.strftime("%Y%m%d%H%M%S")
        dirty = shell("git", "status", "--porcelain")
        version = f"{version_core}-dev{time}"
        if dirty:
            version += ".dirty"
    else:
        version = version_core

    with open(Path("food_gen", "version.py"), "w") as f:
        f.write('__version__ = "{}"\n'.format(version))

    return version


with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="food-gen",
    python_requires=">=3.6.0",
    version=write_version("0.0.1"),
    description="WIP",
    author="Niu Zhe",
    author_email="niuzhe.nz@outlook.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "torchzq >= 1.0.9.dev1",
        "torch",
        "einops",
        "opencv-python",
    ],
    entry_points={
        "console_scripts": [
            "food-gen-gan=food_gen.runners.gan:main",
            "food-gen-vae=food_gen.runners.vae:main",
        ],
    },
)
