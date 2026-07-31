"""improcv: modern image-processing and computer-vision utilities for NumPy and OpenCV."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("improcv")
except PackageNotFoundError:
    # Only happens when running from a source checkout that was never
    # installed (editable or otherwise) -- package metadata doesn't exist
    # yet to read a version from.
    __version__ = "0.0.0.dev0"

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
from improcv.augmentation import (
    AffineParameters,
    AugmentedImageMask,
    CropParameters,
    FlipParameters,
    PerspectiveParameters,
    apply_affine,
    apply_crop,
    apply_flip,
    apply_perspective,
    sample_affine,
    sample_crop,
    sample_flip,
    sample_perspective,
)
from improcv.barcode import Barcode, decode_barcodes
from improcv.color import bgr_to_rgb, ensure_bgr, ensure_gray, rgb_to_bgr, to_hsv, to_lab, to_ycrcb
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
from improcv.denoising import nl_means_denoise, nl_means_denoise_colored
from improcv.detectors import (
    FastType,
    MSERRegion,
    detect_blob_keypoints,
    detect_fast_keypoints,
    detect_mser_regions,
)
from improcv.discovery import ImageMaskPair, discover_image_mask_pairs, discover_images
from improcv.dnn import (
    create_dnn_batch_blob,
    create_dnn_blob,
    load_onnx_network,
    load_onnx_network_from_bytes,
)
from improcv.drawing import draw_bounding_boxes, draw_contours, montage
from improcv.edges import auto_canny, harris_corner, laplacian_edge, sobel_edge
from improcv.evaluation import (
    ClassificationMetrics,
    ConfusionMatrixResult,
    PrecisionRecallCurve,
    RocCurve,
    auc,
    average_precision_score,
    classification_metrics,
    classification_metrics_from_confusion_matrix,
    confusion_matrix,
    multiclass_average_precision_score,
    multiclass_roc_auc_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from improcv.features import (
    DescriptorNorm,
    FeatureMethod,
    Features,
    HomographyResult,
    detect_and_compute,
    find_homography,
    match_features,
    match_features_ratio,
)
from improcv.filters import (
    bilateral_filter,
    clahe,
    gamma_correction,
    gaussian_blur,
    histogram_equalization,
    median_blur,
)
from improcv.hashing import PerceptualHash, PerceptualHashAlgorithm, average_hash, phash
from improcv.hdr import (
    calibrate_camera_response_debevec,
    calibrate_camera_response_robertson,
    fuse_exposures,
    merge_hdr_debevec,
    merge_hdr_robertson,
    tone_map,
    tone_map_drago,
    tone_map_mantiuk,
    tone_map_reinhard,
)
from improcv.hough import (
    Circle,
    HoughCircleMethod,
    Line,
    LineSegment,
    hough_circles,
    hough_line_segments,
    hough_lines,
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
from improcv.photo import (
    PencilSketchResult,
    SeamlessCloneMode,
    detail_enhance,
    pencil_sketch,
    seamless_clone,
    stylize,
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
from improcv.qrcode import QRCode, decode_qr_code, decode_qr_codes
from improcv.quality import gmsd, mse, psnr, ssim
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
from improcv.restoration import InpaintMethod, inpaint
from improcv.segmentation import grabcut_rect, watershed
from improcv.stitching import StitchMode, stitch_images
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
    "AffineParameters",
    "ApproxMethod",
    "AugmentedImageMask",
    "Barcode",
    "BoundingBox",
    "Centroids",
    "Circle",
    "ClassificationMetrics",
    "ComponentStats",
    "ConfusionMatrixResult",
    "Connectivity",
    "Contour",
    "CropParameters",
    "DescriptorNorm",
    "DistanceMaskSize",
    "DistanceType",
    "FastType",
    "FeatureMethod",
    "Features",
    "FlipParameters",
    "FloodFillResult",
    "Hierarchy",
    "HomographyResult",
    "HoughCircleMethod",
    "Image",
    "ImageFloat32",
    "ImageMaskPair",
    "ImageU8",
    "InpaintMethod",
    "Labels",
    "Line",
    "LineSegment",
    "Mask",
    "MSERRegion",
    "MeanStdDevResult",
    "MinMaxResult",
    "Moments",
    "PencilSketchResult",
    "PerceptualHash",
    "PerceptualHashAlgorithm",
    "PerspectiveParameters",
    "PrecisionRecallCurve",
    "QRCode",
    "RetrievalMode",
    "RocCurve",
    "RotatedRect",
    "SeamlessCloneMode",
    "SortOrder",
    "StitchMode",
    "TemplateMatchMethod",
    "TransformMatrix",
    "__version__",
    "adjust_brightness",
    "adjust_contrast",
    "alpha_blend",
    "apply_affine",
    "apply_crop",
    "apply_flip",
    "apply_lut",
    "apply_perspective",
    "approx_poly_dp",
    "auc",
    "auto_canny",
    "average_hash",
    "average_precision_score",
    "bgr_to_rgb",
    "bilateral_filter",
    "bitwise_and",
    "bitwise_or",
    "blackhat",
    "bounding_boxes",
    "calibrate_camera_response_debevec",
    "calibrate_camera_response_robertson",
    "center_crop",
    "clahe",
    "classification_metrics",
    "classification_metrics_from_confusion_matrix",
    "confusion_matrix",
    "connected_components",
    "connected_components_with_stats",
    "convex_hull",
    "create_dnn_batch_blob",
    "create_dnn_blob",
    "crop",
    "decode_barcodes",
    "decode_qr_code",
    "decode_qr_codes",
    "detail_enhance",
    "detect_and_compute",
    "detect_blob_keypoints",
    "detect_fast_keypoints",
    "detect_mser_regions",
    "dilate",
    "discover_image_mask_pairs",
    "discover_images",
    "distance_transform",
    "draw_bounding_boxes",
    "draw_contours",
    "ensure_bgr",
    "ensure_gray",
    "erode",
    "find_contours",
    "find_homography",
    "flip",
    "flood_fill",
    "fuse_exposures",
    "gamma_correction",
    "gaussian_blur",
    "gmsd",
    "grabcut_rect",
    "harris_corner",
    "histogram",
    "histogram_equalization",
    "hough_circles",
    "hough_line_segments",
    "hough_lines",
    "in_range",
    "inpaint",
    "invert",
    "laplacian_edge",
    "load_onnx_network",
    "load_onnx_network_from_bytes",
    "match_features",
    "match_features_ratio",
    "match_template",
    "mean_stddev",
    "median_blur",
    "merge_hdr_debevec",
    "merge_hdr_robertson",
    "min_area_rect",
    "min_max_loc",
    "moments",
    "montage",
    "morph_close",
    "morph_gradient",
    "morph_open",
    "mse",
    "multiclass_average_precision_score",
    "multiclass_roc_auc_score",
    "nl_means_denoise",
    "nl_means_denoise_colored",
    "pad",
    "pencil_sketch",
    "phash",
    "precision_recall_curve",
    "psnr",
    "resize",
    "rgb_to_bgr",
    "roc_auc_score",
    "roc_curve",
    "rotate",
    "rotate_bound",
    "sample_affine",
    "sample_crop",
    "sample_flip",
    "sample_perspective",
    "seamless_clone",
    "sobel_edge",
    "sort_contours",
    "ssim",
    "stitch_images",
    "stylize",
    "threshold",
    "to_hsv",
    "to_lab",
    "to_ycrcb",
    "tone_map",
    "tone_map_drago",
    "tone_map_mantiuk",
    "tone_map_reinhard",
    "tophat",
    "translate",
    "warp_affine",
    "warp_perspective",
    "watershed",
]
