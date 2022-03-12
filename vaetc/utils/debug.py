import os
import sys
import inspect
from typing import Optional, Any, Dict, List, TypeVar

import numpy as np

def debug_print(*message, prefix: Optional[str] = None):
    """
    Show a debug message in the form of "[prefix] message" to stderr.

    Args:
        message (str): message to show
        prefix (:class:`str`, optional): prefix to show; script path if None
    """

    if prefix is None:

        call_stack = inspect.stack()
        script_path = call_stack[1].filename if len(call_stack) >= 2 else "??"
        script_path = os.path.relpath(script_path)
        script_line = call_stack[1].lineno

        prefix_message = f"{script_path}:{script_line}"
    
    else:

        prefix_message = str(prefix)

    print(f"[{prefix_message}] {' '.join(map(str, message))}", file=sys.stderr)