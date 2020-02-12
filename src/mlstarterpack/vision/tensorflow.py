try:
    import tensorflow as tf
except ImportError as exc:
    raise ImportError('This module requires tensorflow extra feature') from exc

from .image_layout import ImageLayout
from .images import get_resized_dims


def extract_square(image, square_size: int, layout: ImageLayout):
    """Extract a square of specified size by resizing and cropping an image.
    :param image: a tensor with a single image
    :param square_size: the resulting square dimension
    :param layout of the image tensor
    """
    assert layout is ImageLayout.HWC

    height = tf.shape(image)[0]
    width = tf.shape(image)[1]

    new_height, new_width = get_resized_dims(height, width, square_size)
    if new_height != height or new_width != width:
        image = tf.image.resize(image, (new_height, new_width))

    center_crop = tf.image.resize_with_crop_or_pad(
        image,
        target_width=square_size,
        target_height=square_size,
    )
    return center_crop
