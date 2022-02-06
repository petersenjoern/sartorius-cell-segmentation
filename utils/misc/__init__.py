import pathlib
from typing import List

def get_items_on_path(path: pathlib.Path) -> List[pathlib.Path]:
    """
    Function to combine directory path with individual items on path
    """

    items_on_path = []
    for filepath in path.iterdir():
        items_on_path.append(filepath)
    return items_on_path




