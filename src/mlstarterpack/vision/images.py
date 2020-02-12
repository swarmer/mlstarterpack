import os
from pathlib import Path
from typing import *

from PIL import Image


IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')


def get_resized_dims(
    height: int,
    width: int,
    target_smaller_side: int,
) -> Tuple[int, int]:
    if width < height:
        if width == target_smaller_side:
            return height, width

        new_width = target_smaller_side
        new_height = int(height / width * target_smaller_side)
    else:
        if height == target_smaller_side:
            return height, width

        new_width = int(width / height * target_smaller_side)
        new_height = target_smaller_side

    return new_height, new_width


def load_img(image_path: Union[str, Path]) -> Image.Image:
    # I believe pillow images are HWC
    image_path = Path(image_path)

    if image_path.suffix.lower() in IMAGE_EXTENSIONS:
        return Image.open(image_path)
    else:
        raise ValueError(f'Unknown image type: {image_path.suffix}')


def find_images(where: Path) -> Iterable[Path]:
    for dirpath, _dirnames, filenames in os.walk(str(where)):
        for filename in filenames:
            file_path = Path(dirpath) / filename
            if not file_path.suffix or file_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            yield file_path
