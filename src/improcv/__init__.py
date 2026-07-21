"""improcv: modern image-processing and computer-vision utilities for NumPy and OpenCV."""

try:
    import cv2 as _cv2  # noqa: F401
except ModuleNotFoundError as _exc:
    # Only a genuinely absent cv2 gets the friendly "install one of these
    # extras" message. A present-but-broken installation (ABI mismatch, a
    # missing shared library, a corrupted build) raises a plain
    # ImportError instead of ModuleNotFoundError, or a ModuleNotFoundError
    # for some *other* module cv2 itself failed to import — in either
    # case, masking that with "you need to install OpenCV" would hide the
    # real problem, so it's left to propagate unmodified.
    if _exc.name != "cv2":
        raise
    raise ImportError(
        "improcv requires an OpenCV installation, which is not installed automatically "
        "(to avoid conflicting with an OpenCV variant you may already have). Install "
        'exactly one of: `pip install "improcv[cv]"` (opencv-python), '
        '`pip install "improcv[cv-headless]"` (opencv-python-headless), '
        '`pip install "improcv[cv-contrib]"` (opencv-contrib-python), or '
        '`pip install "improcv[cv-contrib-headless]"` (opencv-contrib-python-headless) — '
        "or install one of the four `opencv-*` packages yourself."
    ) from _exc

from improcv.analysis import (
    MeanStdDevResult,
    MinMaxResult,
    Moments,
    TemplateMatchMethod,
    histogram,
    match_template,
    mean_stddev,
    min_max_loc,
    moments,
)
from improcv.color import bgr_to_rgb, ensure_gray, rgb_to_bgr, to_hsv, to_lab, to_ycrcb
from improcv.contours import (
    ApproxMethod,
    BoundingBox,
    Contour,
    Hierarchy,
    RetrievalMode,
    RotatedRect,
    SortOrder,
    approx_poly_dp,
    bounding_boxes,
    convex_hull,
    find_contours,
    min_area_rect,
    sort_contours,
)
from improcv.edges import auto_canny, harris_corner, laplacian_edge, sobel_edge
from improcv.filters import (
    bilateral_filter,
    clahe,
    gamma_correction,
    gaussian_blur,
    histogram_equalization,
    median_blur,
)
from improcv.morphology import (
    blackhat,
    dilate,
    erode,
    morph_close,
    morph_gradient,
    morph_open,
    threshold,
    tophat,
)
from improcv.pixels import (
    adjust_brightness,
    adjust_contrast,
    alpha_blend,
    apply_lut,
    bitwise_and,
    bitwise_or,
    in_range,
    invert,
)
from improcv.regions import (
    Centroids,
    ComponentStats,
    Connectivity,
    DistanceMaskSize,
    DistanceType,
    FloodFillResult,
    Labels,
    connected_components,
    connected_components_with_stats,
    distance_transform,
    flood_fill,
)
from improcv.transforms import (
    center_crop,
    crop,
    flip,
    pad,
    resize,
    rotate,
    rotate_bound,
    translate,
    warp_affine,
    warp_perspective,
)
from improcv.types import Image, ImageFloat32, ImageU8, Mask, TransformMatrix

__all__ = [
    "ApproxMethod",
    "BoundingBox",
    "Centroids",
    "ComponentStats",
    "Connectivity",
    "Contour",
    "DistanceMaskSize",
    "DistanceType",
    "FloodFillResult",
    "Hierarchy",
    "Image",
    "ImageFloat32",
    "ImageU8",
    "Labels",
    "Mask",
    "MeanStdDevResult",
    "MinMaxResult",
    "Moments",
    "RetrievalMode",
    "RotatedRect",
    "SortOrder",
    "TemplateMatchMethod",
    "TransformMatrix",
    "adjust_brightness",
    "adjust_contrast",
    "alpha_blend",
    "apply_lut",
    "approx_poly_dp",
    "auto_canny",
    "bgr_to_rgb",
    "bilateral_filter",
    "bitwise_and",
    "bitwise_or",
    "blackhat",
    "bounding_boxes",
    "center_crop",
    "clahe",
    "connected_components",
    "connected_components_with_stats",
    "convex_hull",
    "crop",
    "dilate",
    "distance_transform",
    "ensure_gray",
    "erode",
    "find_contours",
    "flip",
    "flood_fill",
    "gamma_correction",
    "gaussian_blur",
    "harris_corner",
    "histogram",
    "histogram_equalization",
    "in_range",
    "invert",
    "laplacian_edge",
    "match_template",
    "mean_stddev",
    "median_blur",
    "min_area_rect",
    "min_max_loc",
    "moments",
    "morph_close",
    "morph_gradient",
    "morph_open",
    "pad",
    "resize",
    "rgb_to_bgr",
    "rotate",
    "rotate_bound",
    "sobel_edge",
    "sort_contours",
    "threshold",
    "to_hsv",
    "to_lab",
    "to_ycrcb",
    "tophat",
    "translate",
    "warp_affine",
    "warp_perspective",
]
