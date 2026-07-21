"""
High-level trajectory GIF generation for online-emulation experiments.

Orchestrates :mod:`~next_action_pred_eval.utils.image_utils.gif` (pure PIL)
and :mod:`~next_action_pred_eval.utils.image_utils.screenshots` (xlwings) to
produce annotated animated GIFs that visualise USER and MODEL steps.

Two main entry-points:

* ``create_attributed_trajectory_gif``  -- USER steps + accepted MODEL
  predictions only.
* ``create_detailed_attribution_trajectory_gif`` -- full trajectory including
  ACCEPTED / REJECTED labels for every MODEL prediction.
"""

import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

from next_action_pred_eval.utils.image_utils.gif import (
    draw_trajectory_headers,
    extract_operation_names,
)
from next_action_pred_eval.utils.image_utils.screenshots import (
    configure_app_for_automation,
    create_configured_app,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional xlwings import
# ---------------------------------------------------------------------------

try:
    import xlwings as xw
    from xlwings.constants import (
        BorderWeight,
        BordersIndex,
        DeleteShiftDirection,
        HAlign,
        LineStyle,
        UnderlineStyle,
        VAlign,
    )

    _HAS_XLWINGS = True
except ImportError:
    _HAS_XLWINGS = False
    xw = None  # type: ignore[assignment]

# Conditional openpyxl import (used only for column-letter arithmetic)
try:
    from openpyxl.utils import column_index_from_string, get_column_letter

    _HAS_OPENPYXL = True
except ImportError:
    _HAS_OPENPYXL = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_attributed_trajectory_gif(
    experiment_dir: str,
    output_path: Optional[str] = None,
    user_frame_duration: int = 300,
    model_frame_duration: int = 3000,
    header_height: int = 80,
    debug_pngs: bool = False,
    max_row_limit: int = 100,
    max_col_limit: int = 100,
) -> str:
    """
    Create a GIF showing USER steps and accepted MODEL predictions.

    Each frame carries left/right sidebars:
    - Left:  "Step X of N" + progress bar + operation names
    - Right: "USER" or "MODEL" with cumulative stats

    Args:
        experiment_dir: Path to the experiment directory that contains
            ``predictions/content_timeline.json``.
        output_path: Destination for the GIF.  Defaults to
            ``<experiment_dir>/attributed_trajectory.gif``.
        user_frame_duration: Milliseconds per USER frame.
        model_frame_duration: Milliseconds per MODEL frame.
        header_height: Sidebar layout parameter (pixels).
        debug_pngs: If ``True``, keep intermediate PNGs in a ``debug_pngs/``
            subdirectory.
        max_row_limit: Cap row extent when computing the capture range.
        max_col_limit: Cap column extent when computing the capture range.

    Returns:
        The path to the created GIF file.

    Raises:
        FileNotFoundError: If the timeline JSON does not exist.
        ValueError: If no frames could be generated.
        RuntimeError: If xlwings is not installed.
    """
    if not _HAS_XLWINGS:
        raise RuntimeError(
            "xlwings is required for trajectory GIF generation but is not installed."
        )

    if output_path is None:
        output_path = os.path.join(experiment_dir, "attributed_trajectory.gif")

    timeline_path = os.path.join(experiment_dir, "predictions", "content_timeline.json")
    if not os.path.exists(timeline_path):
        raise FileNotFoundError(f"Timeline file not found: {timeline_path}")

    frames_data = _parse_timeline_attributed(timeline_path)
    if not frames_data:
        raise ValueError("No frames to generate")

    max_range = _compute_max_range(timeline_path, max_row_limit, max_col_limit)

    screenshots = _generate_frame_screenshots(
        frames_data=frames_data,
        max_range=max_range,
        experiment_dir=experiment_dir,
        include_rejected=False,
        debug_pngs=debug_pngs,
    )
    if not screenshots:
        raise ValueError("No screenshots generated")

    # Compose header frames and collect per-frame durations.
    frames_with_headers: List[Image.Image] = []
    frame_durations: List[int] = []
    total_frames = len(screenshots)
    user_count = 0
    model_count = 0

    for i, (screenshot_path, frame_info) in enumerate(screenshots):
        if frame_info["source"] == "USER":
            user_count += 1
            frame_durations.append(user_frame_duration)
        else:
            model_count += 1
            frame_durations.append(model_frame_duration)

        frame_img = Image.open(screenshot_path)
        op_names = extract_operation_names(frame_info.get("ops_to_apply", []))

        frame_with_header = draw_trajectory_headers(
            frame_img,
            step_num=i + 1,
            total_steps=total_frames,
            source=frame_info["source"],
            accepted=frame_info.get("accepted"),
            user_count=user_count,
            model_count=model_count,
            header_height=header_height,
            operation_names=op_names,
        )
        frames_with_headers.append(frame_with_header)

        if not debug_pngs:
            os.remove(screenshot_path)

    # Save animated GIF.
    if frames_with_headers:
        frames_with_headers[0].save(
            output_path,
            save_all=True,
            append_images=frames_with_headers[1:],
            duration=frame_durations,
            loop=0,
        )

    logger.info("Attributed trajectory GIF saved to %s (%d frames)", output_path, len(frames_with_headers))
    return output_path


def create_detailed_attribution_trajectory_gif(
    experiment_dir: str,
    output_path: Optional[str] = None,
    user_frame_duration: int = 300,
    model_frame_duration: int = 3000,
    header_height: int = 80,
    debug_pngs: bool = False,
    max_row_limit: int = 100,
    max_col_limit: int = 100,
) -> str:
    """
    Create a detailed GIF including rejected MODEL predictions.

    Rejected predictions are shown with a "MODEL REJECTED" label; the
    workbook state is then rolled back before the next USER frame.

    Args:
        experiment_dir: Path to experiment directory.
        output_path: Destination for the GIF.  Defaults to
            ``<experiment_dir>/detailed_trajectory.gif``.
        user_frame_duration: Milliseconds per USER frame.
        model_frame_duration: Milliseconds per MODEL frame.
        header_height: Sidebar layout parameter (pixels).
        debug_pngs: Keep intermediate PNGs if ``True``.
        max_row_limit: Cap row extent.
        max_col_limit: Cap column extent.

    Returns:
        The path to the created GIF file.
    """
    if not _HAS_XLWINGS:
        raise RuntimeError(
            "xlwings is required for trajectory GIF generation but is not installed."
        )

    if output_path is None:
        output_path = os.path.join(experiment_dir, "detailed_trajectory.gif")

    timeline_path = os.path.join(experiment_dir, "predictions", "content_timeline.json")
    if not os.path.exists(timeline_path):
        raise FileNotFoundError(f"Timeline file not found: {timeline_path}")

    frames_data = _parse_timeline_detailed(timeline_path)
    if not frames_data:
        raise ValueError("No frames to generate")

    max_range = _compute_max_range(timeline_path, max_row_limit, max_col_limit)

    screenshots = _generate_frame_screenshots(
        frames_data=frames_data,
        max_range=max_range,
        experiment_dir=experiment_dir,
        include_rejected=True,
        debug_pngs=debug_pngs,
    )
    if not screenshots:
        raise ValueError("No screenshots generated")

    frames_with_headers: List[Image.Image] = []
    frame_durations: List[int] = []
    total_frames = len(screenshots)
    user_count = 0
    model_accepted = 0
    model_rejected = 0

    for i, (screenshot_path, frame_info) in enumerate(screenshots):
        if frame_info["source"] == "USER":
            user_count += 1
            frame_durations.append(user_frame_duration)
        elif frame_info.get("accepted"):
            model_accepted += 1
            frame_durations.append(model_frame_duration)
        else:
            model_rejected += 1
            frame_durations.append(model_frame_duration)

        frame_img = Image.open(screenshot_path)
        op_names = extract_operation_names(frame_info.get("ops_to_apply", []))

        frame_with_header = draw_trajectory_headers(
            frame_img,
            step_num=i + 1,
            total_steps=total_frames,
            source=frame_info["source"],
            accepted=frame_info.get("accepted"),
            user_count=user_count,
            model_count=model_accepted,
            rejected_count=model_rejected,
            header_height=header_height,
            show_accepted_rejected=True,
            operation_names=op_names,
        )
        frames_with_headers.append(frame_with_header)

        if not debug_pngs:
            os.remove(screenshot_path)

    if frames_with_headers:
        frames_with_headers[0].save(
            output_path,
            save_all=True,
            append_images=frames_with_headers[1:],
            duration=frame_durations,
            loop=0,
        )

    logger.info("Detailed trajectory GIF saved to %s (%d frames)", output_path, len(frames_with_headers))
    return output_path


def create_all_trajectory_gifs(
    experiment_dir: str,
    output_dir: Optional[str] = None,
    user_frame_duration: int = 300,
    model_frame_duration: int = 3000,
    debug_pngs: bool = False,
    max_row_limit: int = 100,
    max_col_limit: int = 100,
) -> Tuple[str, str]:
    """
    Convenience wrapper that generates both the attributed and detailed
    trajectory GIFs in one call.

    Args:
        experiment_dir: Path to experiment directory.
        output_dir: Optional output directory (defaults to *experiment_dir*).
        user_frame_duration: Milliseconds per USER frame.
        model_frame_duration: Milliseconds per MODEL frame.
        debug_pngs: Keep intermediate PNGs if ``True``.
        max_row_limit: Cap row extent.
        max_col_limit: Cap column extent.

    Returns:
        ``(attributed_gif_path, detailed_gif_path)`` tuple.
    """
    if output_dir is None:
        output_dir = experiment_dir

    attributed_output = os.path.join(output_dir, "attributed_trajectory.gif")
    detailed_output = os.path.join(output_dir, "detailed_trajectory.gif")

    attributed_path = create_attributed_trajectory_gif(
        experiment_dir,
        output_path=attributed_output,
        user_frame_duration=user_frame_duration,
        model_frame_duration=model_frame_duration,
        debug_pngs=debug_pngs,
        max_row_limit=max_row_limit,
        max_col_limit=max_col_limit,
    )
    detailed_path = create_detailed_attribution_trajectory_gif(
        experiment_dir,
        output_path=detailed_output,
        user_frame_duration=user_frame_duration,
        model_frame_duration=model_frame_duration,
        debug_pngs=debug_pngs,
        max_row_limit=max_row_limit,
        max_col_limit=max_col_limit,
    )
    return attributed_path, detailed_path


# ---------------------------------------------------------------------------
# Timeline parsing helpers
# ---------------------------------------------------------------------------


def _parse_timeline_attributed(timeline_path: str) -> List[Dict[str, Any]]:
    """
    Parse ``content_timeline.json`` to produce frames for USER steps and
    *accepted* MODEL predictions only.

    Each USER operation becomes one frame; each accepted MODEL prediction
    (which may contain multiple operations) becomes one frame.

    Returns a list of frame dicts with keys:
    - ``source``: ``"USER"`` or ``"MODEL"``
    - ``ops_to_apply``: new operations to apply for this frame
    - ``cumulative_ops``: full operation list after this frame
    - ``accepted``: ``True`` for MODEL predictions
    """
    frames: List[Dict[str, Any]] = []

    with open(timeline_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    timeline = data.get("timeline", [])
    if not timeline:
        return frames

    final_ops = timeline[-1].get("history_after", [])

    # Build map of accepted-prediction start indices.
    prediction_ranges: Dict[int, Tuple[int, list]] = {}
    for entry in timeline:
        prediction = entry.get("prediction", {})
        if len(prediction.get("accepted_by", [])) > 0:
            hist_before = len(entry.get("history_before", []))
            hist_after = len(entry.get("history_after", []))
            pred_ops = prediction.get("predicted_operations", [])
            prediction_ranges[hist_before] = (hist_after, pred_ops)

    i = 0
    while i < len(final_ops):
        if i in prediction_ranges:
            end_idx, _pred_ops = prediction_ranges[i]
            frames.append(
                {
                    "source": "MODEL",
                    "accepted": True,
                    "ops_to_apply": final_ops[i:end_idx],
                    "cumulative_ops": final_ops[:end_idx],
                }
            )
            i = end_idx
        else:
            frames.append(
                {
                    "source": "USER",
                    "ops_to_apply": [final_ops[i]],
                    "cumulative_ops": final_ops[: i + 1],
                }
            )
            i += 1

    return frames


def _parse_timeline_detailed(timeline_path: str) -> List[Dict[str, Any]]:
    """
    Parse ``content_timeline.json`` to produce frames for **all** steps,
    including rejected MODEL predictions.

    Rejected predictions are annotated with ``is_rejected=True`` and a
    ``rollback_ops`` key so that the screenshot generator can restore state
    after displaying the rejected frame.
    """
    frames: List[Dict[str, Any]] = []

    with open(timeline_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    timeline = data.get("timeline", [])
    if not timeline:
        return frames

    final_ops = timeline[-1].get("history_after", [])

    all_predictions: List[Dict[str, Any]] = []
    for entry in timeline:
        prediction = entry.get("prediction", {})
        is_accepted = len(prediction.get("accepted_by", [])) > 0
        all_predictions.append(
            {
                "start_idx": len(entry.get("history_before", [])),
                "end_idx": len(entry.get("history_after", [])),
                "is_accepted": is_accepted,
                "pred_ops": prediction.get("predicted_operations", []),
                "history_before": entry.get("history_before", []),
            }
        )

    all_predictions.sort(key=lambda x: x["start_idx"])

    i = 0
    pred_idx = 0

    while i < len(final_ops):
        if pred_idx < len(all_predictions) and all_predictions[pred_idx]["start_idx"] == i:
            pred = all_predictions[pred_idx]

            if pred["is_accepted"]:
                end_idx = pred["end_idx"]
                frames.append(
                    {
                        "source": "MODEL",
                        "accepted": True,
                        "is_rejected": False,
                        "ops_to_apply": final_ops[i:end_idx],
                        "cumulative_ops": final_ops[:end_idx],
                    }
                )
                i = end_idx
            else:
                frames.append(
                    {
                        "source": "MODEL",
                        "accepted": False,
                        "is_rejected": True,
                        "ops_to_apply": pred["pred_ops"],
                        "cumulative_ops": pred["history_before"] + pred["pred_ops"],
                        "rollback_to": i,
                        "rollback_ops": pred["history_before"],
                    }
                )
                # Do *not* advance ``i`` -- next iteration picks up from here.

            pred_idx += 1
        else:
            frames.append(
                {
                    "source": "USER",
                    "ops_to_apply": [final_ops[i]],
                    "cumulative_ops": final_ops[: i + 1],
                }
            )
            i += 1

    return frames


def _compute_max_range(
    timeline_path: str,
    max_row_limit: int = 100,
    max_col_limit: int = 100,
) -> str:
    """
    Scan all operations in the timeline to determine the maximum extent of
    cell references, then return a range string such as ``"A1:Z100"``.

    Uses openpyxl utilities for column-letter conversion.  If openpyxl is not
    installed, falls back to a sensible default.
    """
    if not _HAS_OPENPYXL:
        logger.warning(
            "openpyxl not installed -- using default range A1:Z100 for screenshot capture."
        )
        return "A1:Z100"

    max_row = 1
    max_col = 1

    def _parse_range_str(range_str: str) -> Tuple[int, int]:
        pattern = r"([A-Za-z]+)(\d+)"
        matches = re.findall(pattern, range_str)
        mr, mc = 1, 1
        for col_str, row_str in matches:
            if len(col_str) > 3:
                continue  # skip sheet-name fragments
            mr = max(mr, int(row_str))
            mc = max(mc, column_index_from_string(col_str))
        return mr, mc

    def _extract_ranges_from_symbolic(symbolic: str) -> List[str]:
        parts = symbolic.split(" | ")
        if len(parts) >= 2:
            return [parts[1]]
        return []

    with open(timeline_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    for entry in data.get("timeline", []):
        for op in entry.get("history_after", []):
            if isinstance(op, str):
                for rs in _extract_ranges_from_symbolic(op):
                    r, c = _parse_range_str(rs)
                    max_row = max(max_row, r)
                    max_col = max(max_col, c)
            elif isinstance(op, dict):
                for key in ("range", "targetRange"):
                    if key in op:
                        r, c = _parse_range_str(op[key])
                        max_row = max(max_row, r)
                        max_col = max(max_col, c)

        prediction = entry.get("prediction", {})
        for op in prediction.get("predicted_operations", []):
            if isinstance(op, str):
                for rs in _extract_ranges_from_symbolic(op):
                    r, c = _parse_range_str(rs)
                    max_row = max(max_row, r)
                    max_col = max(max_col, c)
            elif isinstance(op, dict):
                for key in ("range", "targetRange"):
                    if key in op:
                        r, c = _parse_range_str(op[key])
                        max_row = max(max_row, r)
                        max_col = max(max_col, c)

    # Padding.
    max_row = min(max_row + 5, max_row_limit)
    max_col = min(max_col + 2, max_col_limit)

    col_letter = get_column_letter(max_col)
    logger.info(
        "Computed max range: A1:%s%d (limits: %d rows, %d cols)",
        col_letter,
        max_row,
        max_row_limit,
        max_col_limit,
    )
    return f"A1:{col_letter}{max_row}"


# ---------------------------------------------------------------------------
# Screenshot generation (xlwings-heavy)
# ---------------------------------------------------------------------------


def _generate_frame_screenshots(
    frames_data: List[Dict[str, Any]],
    max_range: str,
    experiment_dir: str,
    include_rejected: bool = False,
    debug_pngs: bool = False,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Execute operations frame-by-frame in Excel via xlwings and capture a
    screenshot after each step.

    For rejected predictions the workbook is saved before applying the
    prediction, captured, then restored from the save so the next frame
    starts from the correct state.

    Returns a list of ``(screenshot_path, frame_info)`` tuples.
    """
    if not _HAS_XLWINGS:
        logger.error("xlwings not available -- cannot generate frame screenshots.")
        return []

    screenshots: List[Tuple[str, Dict[str, Any]]] = []
    temp_dir = tempfile.mkdtemp(prefix="gif_frames_")
    debug_dir = os.path.join(experiment_dir, "debug_pngs") if debug_pngs else None

    if debug_pngs and debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    state_file = os.path.join(temp_dir, "state_backup.xlsx")

    try:
        app = xw.App(visible=False)
        configure_app_for_automation(app)

        try:
            wb = app.books.add()
            exec_env = _build_exec_env(wb, app)
            frame_idx = 0

            for frame_data in frames_data:
                source = frame_data["source"]
                is_rejected = frame_data.get("is_rejected", False)

                if is_rejected:
                    wb.save(state_file)

                ops_to_apply = frame_data.get("ops_to_apply", [])
                if ops_to_apply:
                    _apply_operations(ops_to_apply, exec_env)

                screenshot_path = os.path.join(temp_dir, f"frame_{frame_idx:04d}.png")
                _capture_screenshot(wb, screenshot_path, max_range)

                if debug_pngs and debug_dir:
                    suffix = (
                        "REJECTED"
                        if is_rejected
                        else ("ACCEPTED" if frame_data.get("accepted") else "")
                    )
                    debug_path = os.path.join(
                        debug_dir, f"frame_{frame_idx:04d}_{source}_{suffix}.png"
                    )
                    shutil.copy(screenshot_path, debug_path)
                    logger.debug("Debug PNG saved: %s", debug_path)

                screenshots.append(
                    (
                        screenshot_path,
                        {
                            "source": source,
                            "accepted": frame_data.get("accepted"),
                            "is_rejected": is_rejected,
                            "ops_to_apply": ops_to_apply,
                        },
                    )
                )
                frame_idx += 1

                # Rollback after rejected prediction.
                if is_rejected and os.path.exists(state_file):
                    wb.close()
                    wb = app.books.open(state_file, update_links=False)
                    exec_env = _build_exec_env(wb, app)

            return screenshots

        finally:
            try:
                wb.close()
            except Exception:
                pass
            try:
                app.quit()
            except Exception:
                pass

    except Exception:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise


# ---------------------------------------------------------------------------
# xlwings execution helpers
# ---------------------------------------------------------------------------


def _build_exec_env(wb, app) -> Dict[str, Any]:
    """Build a dict of helpers for executing operations against *wb*."""

    def get_sheet(name: Optional[str] = None):
        if name is None:
            return wb.sheets.active
        try:
            return wb.sheets[name]
        except Exception:
            return wb.sheets.add(name)

    def ensure_sheet(name: str):
        return get_sheet(name)

    def copy_range(source, dest, paste="all"):
        source.copy()
        dest.paste(paste=paste)
        wb.app.api.CutCopyMode = False

    def set_border_side(rng, border_index, line_style, weight, color):
        rng.api.Borders(border_index).LineStyle = line_style
        rng.api.Borders(border_index).Weight = weight
        if color and color != "None":
            color_hex = color.lstrip("#")
            r = int(color_hex[0:2], 16)
            g = int(color_hex[2:4], 16)
            b = int(color_hex[4:6], 16)
            rng.api.Borders(border_index).Color = r + (g * 256) + (b * 65536)

    def set_border_outside(rng, line_style, weight, color):
        sides = [
            BordersIndex.xlEdgeTop,
            BordersIndex.xlEdgeBottom,
            BordersIndex.xlEdgeLeft,
            BordersIndex.xlEdgeRight,
        ]
        for side in sides:
            set_border_side(rng, side, line_style, weight, color)

    def set_border_all(rng, line_style, weight, color):
        set_border_outside(rng, line_style, weight, color)
        try:
            set_border_side(rng, BordersIndex.xlInsideHorizontal, line_style, weight, color)
        except Exception:
            pass
        try:
            set_border_side(rng, BordersIndex.xlInsideVertical, line_style, weight, color)
        except Exception:
            pass

    return {
        "wb": wb,
        "app": app,
        "xw": xw,
        "VAlign": VAlign,
        "HAlign": HAlign,
        "DeleteShiftDirection": DeleteShiftDirection,
        "BordersIndex": BordersIndex,
        "LineStyle": LineStyle,
        "BorderWeight": BorderWeight,
        "UnderlineStyle": UnderlineStyle,
        "get_sheet": get_sheet,
        "ensure_sheet": ensure_sheet,
        "copy_range": copy_range,
        "set_border_side": set_border_side,
        "set_border_outside": set_border_outside,
        "set_border_all": set_border_all,
    }


def _apply_operations(operations: List, exec_env: Dict[str, Any]) -> None:
    """
    Apply a list of operations to the workbook via xlwings.

    Operations may be either symbolic strings (``"INPUT | STA!A1 | value"``)
    or dicts (``{"type": "INPUT", "range": "A1", ...}``).  Conversion to
    ``Operation`` objects is performed via ``next_action_pred_eval.core``
    (remapped from the old ``excel_converter`` package).
    """
    try:
        from next_action_pred_eval.core.symbolic_converter import symbolic_to_operations
        from next_action_pred_eval.core.operations import Operation
    except ImportError:
        logger.warning(
            "next_action_pred_eval.core not available -- cannot apply operations. "
            "Falling back to no-op."
        )
        return

    wb = exec_env["wb"]

    if operations and isinstance(operations[0], str):
        try:
            ops = symbolic_to_operations(operations)
        except Exception as exc:
            logger.warning("Failed to parse symbolic operations: %s", exc)
            return
    else:
        ops = []
        for op_dict in operations:
            if not isinstance(op_dict, dict):
                continue
            try:
                ops.append(Operation.from_dict(op_dict))
            except Exception as exc:
                logger.warning("Failed to parse operation %s: %s", op_dict.get("type", "unknown"), exc)

    for op in ops:
        try:
            sheet_name = (
                op.cell_range.sheet
                if hasattr(op, "cell_range") and op.cell_range
                else None
            )
            if sheet_name:
                sheet = exec_env["ensure_sheet"](sheet_name)
            else:
                sheet = wb.sheets.active

            var_name = _sanitize_sheet_name(sheet_name) if sheet_name else "sheet"
            code = op.to_xlwings(var_name)
            if not code:
                continue

            local_env = {var_name: sheet, "sheet": sheet, "wb": wb, **exec_env}
            exec(code, local_env)  # noqa: S102

        except Exception as exc:
            logger.warning("Failed to apply operation: %s", exc)


def _sanitize_sheet_name(name: str) -> str:
    """Sanitise a sheet name for use as a Python variable name."""
    if not name:
        return "sheet"
    sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    if sanitized and not sanitized[0].isalpha():
        sanitized = "sheet_" + sanitized
    return sanitized or "sheet"


def _capture_screenshot(wb, output_path: str, capture_range: str = "A1:Z100") -> None:
    """
    Capture the active sheet of *wb* as a PNG.

    Falls back to a white placeholder image with an error message if the
    xlwings ``to_png`` call fails.
    """
    try:
        sheet = wb.sheets.active
        rng = sheet.range(capture_range)
        rng.to_png(output_path)
    except Exception as exc:
        logger.warning("Screenshot error: %s", exc)
        img = Image.new("RGB", (800, 600), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((400, 300), f"Error: {str(exc)[:50]}", fill=(255, 0, 0), anchor="mm")
        img.save(output_path)
