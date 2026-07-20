import numpy as np

from improcv.types import ImageFloat32


def test_image_float32_accepts_any_shape_float32_array() -> None:
    arr_2d: ImageFloat32 = np.zeros((10, 10), dtype=np.float32)
    arr_3d: ImageFloat32 = np.zeros((10, 10, 3), dtype=np.float32)

    assert arr_2d.dtype == np.float32
    assert arr_3d.dtype == np.float32
