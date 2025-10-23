"""Logging formatter utility for clean, organized console output."""

import logging

logger = logging.getLogger(__name__)

# Box drawing characters
BOX_WIDTH = 45
TOP_LEFT = "┌"
TOP_RIGHT = "┐"
BOTTOM_LEFT = "└"
BOTTOM_RIGHT = "┘"
HORIZONTAL = "─"
VERTICAL = "│"
SEPARATOR = "─"


def log_stage_header(stage_num: int, stage_name: str):
    """
    Log a formatted stage header with Unicode box.

    Args:
        stage_num: Stage number (1-6)
        stage_name: Name of the stage in UPPERCASE
    """
    title = f" STAGE {stage_num}: {stage_name} "
    padding = BOX_WIDTH - len(title) - 2
    left_pad = padding // 2
    right_pad = padding - left_pad

    top_line = TOP_LEFT + HORIZONTAL * (BOX_WIDTH - 2) + TOP_RIGHT
    middle_line = VERTICAL + " " * left_pad + title + " " * right_pad + VERTICAL
    bottom_line = BOTTOM_LEFT + HORIZONTAL * (BOX_WIDTH - 2) + BOTTOM_RIGHT

    logger.info("")
    logger.info(top_line)
    logger.info(middle_line)
    logger.info(bottom_line)
    logger.info("")


def log_separator():
    """Log a separator line between stages."""
    logger.info("")
    logger.info(SEPARATOR * BOX_WIDTH)
    logger.info("")


def log_section(section_name: str):
    """
    Log a section header within a stage.

    Args:
        section_name: Section name with dashes (e.g., "--- Symbolic Execution ---")
    """
    logger.info(f"  {section_name}")


def log_info(message: str, indent: int = 2):
    """
    Log an informational message.

    Args:
        message: Message to log
        indent: Number of spaces to indent (default: 2)
    """
    logger.info(" " * indent + message)


def log_ok(message: str, indent: int = 4):
    """
    Log a success message with [OK] indicator.

    Args:
        message: Success message
        indent: Number of spaces to indent (default: 4)
    """
    logger.info(" " * indent + f"[OK] {message}")


def log_fail(message: str, indent: int = 4):
    """
    Log a failure message with [FAIL] indicator.

    Args:
        message: Failure message
        indent: Number of spaces to indent (default: 4)
    """
    logger.info(" " * indent + f"[FAIL] {message}")


def log_skip(message: str, indent: int = 4):
    """
    Log a skipped action with [SKIP] indicator.

    Args:
        message: Skip message
        indent: Number of spaces to indent (default: 4)
    """
    logger.info(" " * indent + f"[SKIP] {message}")


def log_result(message: str, indent: int = 2):
    """
    Log a stage result/outcome.

    Args:
        message: Result message
        indent: Number of spaces to indent (default: 2)
    """
    logger.info("")
    logger.info(" " * indent + f"RESULT: {message}")


def log_substep(message: str, indent: int = 4):
    """
    Log a sub-step within a process.

    Args:
        message: Sub-step message
        indent: Number of spaces to indent (default: 4)
    """
    logger.info(" " * indent + message)


def log_debug_substep(message: str, indent: int = 4):
    """
    Log a debug-level sub-step (only visible at DEBUG level).

    Args:
        message: Debug message
        indent: Number of spaces to indent (default: 4)
    """
    logger.debug(" " * indent + message)
