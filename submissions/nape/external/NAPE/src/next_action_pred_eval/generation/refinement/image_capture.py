from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def ensure_sheet_image(
    workbook_path: Path,
    sheet_name: str,
    existing_path: Optional[Path],
    output_dir: Path,
    max_dimension: Optional[int],
    allow_capture: bool,
) -> Optional[Path]:
    """Return an image path for the sheet, capturing if needed.

    Uses xlwings for image capture if available. If xlwings is not installed,
    logs a warning and returns None (image capture is optional).
    """
    if existing_path and existing_path.exists():
        return existing_path
    if not allow_capture:
        logger.info("Image capture disabled and no pre-generated path supplied.")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from next_action_pred_eval.utils.image_utils.screenshots import save_images_add_headings

        sheet_ranges = save_images_add_headings(
            book=str(workbook_path),
            output=output_dir,
            sheets=[sheet_name],
            max_index=max_dimension,
            app=None,
        )
        if sheet_name in sheet_ranges:
            image_path = output_dir / f"{sheet_name}.png"
            if image_path.exists():
                return image_path
    except ImportError:
        logger.warning(
            "Image capture requires xlwings. "
            "Install xlwings or provide a pre-generated sheet image path."
        )
    except Exception as exc:  # pragma: no cover - depends on Excel runtime
        logger.warning("Failed to capture sheet image: %s", exc)
    return None
