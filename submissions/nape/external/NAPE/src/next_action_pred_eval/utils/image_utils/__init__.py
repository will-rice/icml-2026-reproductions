"""
Image utilities for next-action prediction evaluation.

Sub-modules
-----------
gif
    Pure PIL/Pillow-based GIF creation and frame annotation (no xlwings dependency).
screenshots
    xlwings-based Excel screenshot capture with optional row/column headers.
    Gracefully degrades when xlwings is not installed.
trajectories
    High-level trajectory GIF generation that orchestrates ``gif`` and
    ``screenshots`` to produce annotated animations of experiment runs.
"""

# --- gif.py (always available -- PIL only) --------------------------------
from next_action_pred_eval.utils.image_utils.gif import (
    create_gif_from_images,
    draw_frame_header,
    draw_trajectory_headers,
    extract_operation_names,
)

# --- screenshots.py (xlwings-guarded) ------------------------------------
from next_action_pred_eval.utils.image_utils.screenshots import (
    WhichSheets,
    configure_app_for_automation,
    create_configured_app,
    ensure_app_alive,
    get_extended_used_range,
    is_app_alive,
    save_images_add_headings,
    save_images_no_headings,
    with_app_recovery,
)

# --- trajectories.py (xlwings-guarded) ------------------------------------
from next_action_pred_eval.utils.image_utils.trajectories import (
    create_all_trajectory_gifs,
    create_attributed_trajectory_gif,
    create_detailed_attribution_trajectory_gif,
)

__all__ = [
    # gif
    "create_gif_from_images",
    "draw_frame_header",
    "draw_trajectory_headers",
    "extract_operation_names",
    # screenshots
    "WhichSheets",
    "configure_app_for_automation",
    "create_configured_app",
    "ensure_app_alive",
    "get_extended_used_range",
    "is_app_alive",
    "save_images_add_headings",
    "save_images_no_headings",
    "with_app_recovery",
    # trajectories
    "create_all_trajectory_gifs",
    "create_attributed_trajectory_gif",
    "create_detailed_attribution_trajectory_gif",
]
