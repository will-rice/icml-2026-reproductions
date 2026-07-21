"""
xlwings-based Excel screenshot capture utilities.

Provides functions for capturing spreadsheet ranges as PNG images, with
optional row/column header overlays.  All xlwings usage is guarded behind
``try/except ImportError`` so that the module can be imported on systems
where xlwings is not installed (e.g. Linux CI).
"""

import logging
import time
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, TypeVar, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional xlwings import
# ---------------------------------------------------------------------------

try:
    import xlwings as xw
    from xlwings import App, Book, Sheet

    _HAS_XLWINGS = True
except ImportError:
    _HAS_XLWINGS = False
    xw = None  # type: ignore[assignment]
    logger.warning(
        "xlwings is not installed. Screenshot capture functions will return "
        "empty results. Install xlwings to enable Excel screenshot support."
    )

# Conditional pywintypes import (Windows COM error type)
try:
    from pywintypes import com_error as _com_error  # type: ignore[import-untyped]
except ImportError:
    # Provide a dummy that will never match any raised exception.
    class _com_error(Exception):  # type: ignore[no-redef]
        pass


T = TypeVar("T")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class WhichSheets(Enum):
    """Selector for which sheets to capture from a workbook."""

    Active = "active"
    All = "all"
    First = "first"


# ---------------------------------------------------------------------------
# App lifecycle helpers
# ---------------------------------------------------------------------------


def is_app_alive(app) -> bool:
    """
    Check whether an xlwings ``App`` instance is still alive and responsive.

    Returns ``False`` when *app* is ``None``, xlwings is unavailable, or the
    underlying COM / AppleScript connection is broken.
    """
    if not _HAS_XLWINGS or app is None:
        return False
    try:
        _ = app.pid
        return True
    except (_com_error, AttributeError, Exception):
        return False


def create_configured_app(visible: bool = False, add_book: bool = False):
    """
    Create a new xlwings ``App`` pre-configured for unattended automation.

    Suppresses all dialogs, alerts, screen-updating, and event firing so that
    no user interaction is required.

    Args:
        visible: Whether the Excel window should be visible.
        add_book: Whether to add an empty workbook on creation.

    Returns:
        A configured ``xw.App`` instance, or ``None`` if xlwings is not installed.
    """
    if not _HAS_XLWINGS:
        logger.warning("xlwings not available -- cannot create Excel app.")
        return None

    _MAX_CREATE_RETRIES = 3
    for attempt in range(_MAX_CREATE_RETRIES):
        try:
            app = xw.App(visible=visible, add_book=add_book)
            return configure_app_for_automation(app)
        except Exception as exc:
            if attempt < _MAX_CREATE_RETRIES - 1:
                wait = 3 * (attempt + 1)  # 3s, 6s
                logger.warning(
                    f"Failed to create Excel app (attempt {attempt + 1}/"
                    f"{_MAX_CREATE_RETRIES}): {exc}. Retrying in {wait}s..."
                )
                time.sleep(wait)
            else:
                raise


def configure_app_for_automation(app):
    """
    Suppress alerts, screen-updating, link-update prompts, and events on an
    existing xlwings ``App`` instance.

    Args:
        app: An ``xw.App`` instance.

    Returns:
        The same *app*, now configured.
    """
    if not _HAS_XLWINGS or app is None:
        return app

    app.display_alerts = False
    app.screen_updating = False
    try:
        app.api.AskToUpdateLinks = False
    except AttributeError:
        pass
    try:
        app.api.EnableEvents = False
    except AttributeError:
        pass
    return app


def ensure_app_alive(app, visible: bool = False, add_book: bool = False):
    """
    Return *app* if it is responsive, otherwise create a fresh configured app.

    Args:
        app: The ``xw.App`` to check.
        visible: Passed through to ``create_configured_app`` if a new app is needed.
        add_book: Passed through to ``create_configured_app`` if a new app is needed.

    Returns:
        A valid ``xw.App`` instance (the original or a new one), or ``None``
        if xlwings is not installed.
    """
    if is_app_alive(app):
        return app

    # Try to close the dead app gracefully.
    _safe_close_app(app)

    return create_configured_app(visible=visible, add_book=add_book)


def with_app_recovery(
    func: Callable[..., T],
    app,
    *args,
    max_retries: int = 2,
    visible: bool = False,
    add_book: bool = False,
    **kwargs,
) -> Tuple[T, object]:
    """
    Execute *func* with automatic app recovery on COM / RPC errors.

    If the function fails because the underlying Excel process died (RPC
    error), this helper recreates the ``App`` and retries up to *max_retries*
    times.

    *func* must accept an ``app`` keyword argument.

    Returns:
        ``(result, current_app)`` tuple.
    """
    if not _HAS_XLWINGS:
        raise RuntimeError("xlwings is not installed; cannot execute with app recovery.")

    RPC_ERROR_CODE = -2147023174  # "The RPC server is unavailable"

    last_exception: Optional[Exception] = None
    current_app = app

    for attempt in range(max_retries + 1):
        try:
            current_app = ensure_app_alive(current_app, visible=visible, add_book=add_book)
            result = func(*args, app=current_app, **kwargs)
            return result, current_app

        except _com_error as exc:
            last_exception = exc
            if hasattr(exc, "hresult") and exc.hresult == RPC_ERROR_CODE:
                logger.warning(
                    "RPC error detected (attempt %d/%d), recreating Excel app...",
                    attempt + 1,
                    max_retries + 1,
                )
                _safe_close_app(current_app)
                current_app = create_configured_app(visible=visible, add_book=add_book)
                time.sleep(1)
            else:
                raise

        except Exception as exc:
            if "RPC server" in str(exc) or "-2147023174" in str(exc):
                last_exception = exc
                logger.warning(
                    "RPC error detected (attempt %d/%d), recreating Excel app...",
                    attempt + 1,
                    max_retries + 1,
                )
                _safe_close_app(current_app)
                current_app = create_configured_app(visible=visible, add_book=add_book)
                time.sleep(1)
            else:
                raise

    # All retries exhausted.
    raise last_exception  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Range / screenshot helpers
# ---------------------------------------------------------------------------


def get_extended_used_range(ws, max_index: Optional[int] = None) -> str:
    """
    Compute the used range of *ws* extended to encompass any shapes / charts.

    Adds a one-row / one-column padding around the result and optionally clips
    rows and columns to *max_index*.

    Args:
        ws: An ``xw.Sheet`` instance.
        max_index: If given, clip row and column indices to this value.

    Returns:
        An Excel address string such as ``"$A$1:$Z$50"``.
    """
    if not _HAS_XLWINGS:
        return "$A$1:$A$1"

    used_range = ws.used_range
    min_row, min_col = used_range.row, used_range.column
    max_row = used_range.last_cell.row
    max_col = used_range.last_cell.column

    # Expand for shapes / charts that extend beyond the cell used range.
    for shape in ws.api.Shapes:
        min_row = min(min_row, shape.TopLeftCell.Row)
        min_col = min(min_col, shape.TopLeftCell.Column)
        max_row = max(max_row, shape.BottomRightCell.Row)
        max_col = max(max_col, shape.BottomRightCell.Column)

    # One-cell padding.
    min_row = max(1, min_row - 1)
    min_col = max(1, min_col - 1)
    max_row = max(1, max_row + 1)
    max_col = max(1, max_col + 1)

    if max_index is not None:
        min_row = min(min_row, max_index)
        min_col = min(min_col, max_index)
        max_row = min(max_row, max_index)
        max_col = min(max_col, max_index)

    return (
        f"${xw.utils.col_name(min_col)}${min_row}"
        f":${xw.utils.col_name(max_col)}${max_row}"
    )


# ---------------------------------------------------------------------------
# High-level image-saving functions
# ---------------------------------------------------------------------------


def save_images_add_headings(
    book,
    output: Union[str, Path],
    sheets: Union[str, List[str], WhichSheets] = WhichSheets.All,
    app=None,
    max_index: Optional[int] = 100,
) -> Dict[str, str]:
    """
    Capture screenshots of workbook sheets **with** row/column header overlays.

    The function temporarily inserts a header row and header column into the
    worksheet, captures the range as a PNG, then removes the inserted headers
    so that the workbook is left unchanged.

    Args:
        book: An ``xw.Book``, or a file-system path to an ``.xlsx`` file.
        output: Output directory (one PNG per sheet) or a single file path.
        sheets: Which sheets to process -- a name, list of names, or a
                ``WhichSheets`` enum member.
        app: Optional ``xw.App``.  Created internally if ``None``.
        max_index: Passed to ``get_extended_used_range`` to cap dimensions.

    Returns:
        Dict mapping sheet name to the address string that was captured.
        Returns an empty dict if xlwings is not installed.
    """
    if not _HAS_XLWINGS:
        logger.warning("xlwings not available -- returning empty result from save_images_add_headings.")
        return {}

    close_app_flag = False
    close_book = False

    if app is None:
        app = create_configured_app(visible=False, add_book=False)
        close_app_flag = True
    if isinstance(book, (str, Path)):
        book_path = str(book)
        try:
            book = app.books.open(book_path, update_links=False, read_only=False)
        except Exception as open_err:
            # Write-protected or password-protected file — fall back to
            # read-only capture without header overlays.
            err_msg = str(open_err)
            if "read-only" in err_msg.lower() or "password" in err_msg.lower() or "Open method" in err_msg:
                logger.warning(
                    f"Cannot open {book_path} for writing ({open_err}). "
                    f"Falling back to no-headings capture."
                )
                return save_images_no_headings(
                    book=book_path, output=output, sheets=sheets,
                    app=app, max_index=max_index,
                )
            raise
        close_book = True

    try:
        sheet_names = _resolve_sheet_names(book, sheets)
        output = Path(output)
        _ensure_output_dir(output)

        sheet_ranges: Dict[str, str] = {}

        for sheet_name in sheet_names:
            sheet = book.sheets[sheet_name]
            address = get_extended_used_range(sheet, max_index=max_index)
            output_file = _sheet_output_path(output, sheet_name, len(sheet_names))
            _capture_range_with_headings(sheet, address, output_file, app)
            sheet_ranges[sheet_name] = address
    finally:
        if close_book:
            try:
                book.close(save_changes=False)
            except Exception:
                pass
        if close_app_flag:
            _safe_close_app(app)

    return sheet_ranges


def save_images_no_headings(
    book,
    output: Union[str, Path],
    sheets: Union[str, List[str], WhichSheets] = WhichSheets.All,
    app=None,
    max_index: Optional[int] = 100,
) -> Dict[str, str]:
    """
    Capture screenshots of workbook sheets **without** header overlays.

    Uses the native xlwings ``Range.to_png()`` method directly on the
    extended used range.

    Args:
        book: An ``xw.Book``, or a file-system path to an ``.xlsx`` file.
        output: Output directory (one PNG per sheet) or a single file path.
        sheets: Which sheets to process.
        app: Optional ``xw.App``.
        max_index: Passed to ``get_extended_used_range``.

    Returns:
        Dict mapping sheet name to the address string that was captured.
        Returns an empty dict if xlwings is not installed.
    """
    if not _HAS_XLWINGS:
        logger.warning("xlwings not available -- returning empty result from save_images_no_headings.")
        return {}

    close_app_flag = False
    close_book = False

    if app is None:
        app = create_configured_app(visible=False, add_book=False)
        close_app_flag = True
    if isinstance(book, (str, Path)):
        book = app.books.open(str(book), update_links=False, read_only=True)
        close_book = True

    sheet_names = _resolve_sheet_names(book, sheets)
    output = Path(output)
    _ensure_output_dir(output)

    outputs: Dict[str, str] = {}

    for sheet_name in sheet_names:
        sheet = book.sheets[sheet_name]
        address = get_extended_used_range(sheet, max_index=max_index)
        range_object = sheet.range(address)
        output_file = _sheet_output_path(output, sheet_name, len(sheet_names))
        range_object.to_png(str(output_file))
        outputs[sheet_name] = address

    if close_book:
        book.close()
    if close_app_flag:
        _safe_close_app(app)

    return outputs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_sheet_names(book, sheets) -> List[str]:
    """Normalise a *sheets* argument into a list of sheet-name strings."""
    if isinstance(sheets, str):
        return [sheets]
    if isinstance(sheets, list):
        return sheets
    if sheets == WhichSheets.First:
        return [book.sheet_names[0]]
    if sheets == WhichSheets.Active:
        return [book.sheets.active.name]
    if sheets == WhichSheets.All:
        return list(book.sheet_names)
    # Fallback: treat as iterable.
    return list(sheets)


def _ensure_output_dir(output: Path) -> None:
    """Create parent directories for *output*."""
    if output.suffix == "":
        output.mkdir(parents=True, exist_ok=True)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)


def _sheet_output_path(output: Path, sheet_name: str, total_sheets: int) -> Path:
    """Determine the destination PNG path for a given sheet."""
    if output.is_dir():
        return output / f"{sheet_name}.png"
    if output.suffix != "" and total_sheets > 1:
        return output.with_name(f"{output.stem}-{sheet_name}.png")
    return output


def _capture_range_with_headings(
    sheet,
    cell_range: str,
    output_path: Union[str, Path],
    app=None,
    retries: int = 3,
) -> None:
    """
    Capture a screenshot of *cell_range* with temporary row/column headers.

    Inserts a header row (column letters) and header column (row numbers),
    captures the expanded range, then removes the inserted row/column so the
    workbook is left in its original state.
    """
    if not _HAS_XLWINGS:
        return

    close_app_flag = False
    close_book = False

    if app is None:
        app = create_configured_app(visible=False, add_book=False)
        close_app_flag = True

    if isinstance(sheet, tuple):
        sheet_path, sheet_name = sheet
        sheet = app.books.open(str(sheet_path), update_links=False, read_only=True).sheets[sheet_name]
        close_book = True

    # Unprotect if needed.
    if sheet.api.ProtectContents:
        try:
            sheet.api.Unprotect("")
        except Exception as exc:
            raise ValueError(
                f"Could not unprotect sheet {sheet.name}: {exc}, password may be required."
            )

    range_obj = sheet.range(cell_range)
    first_col = range_obj.column
    first_row = range_obj.row
    last_col = first_col + range_obj.columns.count - 1
    last_row = first_row + range_obj.rows.count - 1

    # Insert header row and column.
    sheet.api.Rows(first_row).Insert()
    sheet.api.Columns(first_col).Insert()

    # Fill header row with column letters.
    for i in range(last_col - first_col + 1):
        col_letter = xw.utils.col_name(first_col + i)
        sheet.cells(first_row, first_col + 1 + i).value = col_letter

    # Fill header column with row numbers.
    for j in range(last_row - first_row + 1):
        sheet.cells(first_row + 1 + j, first_col).value = str(first_row + j)

    # Format header row.
    header_row_range = sheet.range(
        sheet.cells(first_row, first_col + 1),
        sheet.cells(first_row, last_col + 1),
    )
    header_row_range.clear_formats()
    header_row_range.color = (240, 240, 240)
    header_row_range.api.HorizontalAlignment = -4108  # xlCenter
    header_row_range.api.Borders(11).LineStyle = 1
    header_row_range.api.Borders(11).Color = 0xE0E0E0

    # Format header column.
    header_col_range = sheet.range(
        sheet.cells(first_row + 1, first_col),
        sheet.cells(last_row + 1, first_col),
    )
    header_col_range.clear_formats()
    header_col_range.color = (240, 240, 240)
    header_col_range.api.HorizontalAlignment = -4108  # xlCenter
    header_col_range.api.Borders(12).LineStyle = 1
    header_col_range.api.Borders(12).Color = 0xE0E0E0

    # Capture the expanded range (including headers).
    expanded_range = sheet.range(
        sheet.cells(first_row, first_col),
        sheet.cells(last_row + 1, last_col + 1),
    )
    remaining = retries
    while remaining > 0:
        try:
            expanded_range.to_png(str(output_path))
            break
        except Exception as exc:
            if (
                len(exc.args) > 2
                and len(exc.args[2]) > 2
                and exc.args[2][2] == "CopyPicture method of Range class failed"
            ):
                remaining -= 1
                if remaining == 0:
                    raise
                time.sleep(0.1 * min(1, retries - remaining))
            else:
                raise

    # Clean up: remove the inserted header row and column.
    sheet.api.Rows(first_row).Delete()
    sheet.api.Columns(first_col).Delete()

    if close_book:
        sheet.book.close()
    if close_app_flag:
        _safe_close_app(app)


def _safe_close_app(app) -> None:
    """Attempt to close *app* without raising."""
    if app is None:
        return
    try:
        app.display_alerts = False
        # Close any lingering open books without saving
        for book in list(app.books):
            try:
                book.close(save_changes=False)
            except Exception:
                pass
        app.quit()
    except Exception:
        pass
