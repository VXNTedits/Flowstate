import os
import logging

logger = logging.getLogger(__name__)


def get_project_root() -> str:
    """
    Returns the absolute path to the root directory of the project.

    This is based on the assumption that this script is located in
    `src/utils` relative to the project root.

    Returns:
        str: The absolute path to the project root directory.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def get_relative_path(relative_path: str) -> str:
    """
    Get the absolute path relative to the project root directory.

    Args:
        relative_path (str): The relative path to be joined.

    Returns:
        str: The absolute path.
    """
    project_root = get_project_root()
    return os.path.join(project_root, relative_path)
