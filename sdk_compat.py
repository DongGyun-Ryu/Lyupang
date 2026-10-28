"""Process-local compatibility for kwcli 1.0.3's Windows ACL checks.

The SDK checks only icacls return codes but captures its localized output with
Python's default text decoder. UTF-8 mode does not change icacls' output code
page. Capture these unused streams as bytes, without changing ACL commands or
the standard subprocess module used by the rest of the application.
"""
from __future__ import annotations

import ntpath
import sys


class _ACLSubprocess:
    def __init__(self, delegate):
        self._delegate = delegate

    def __getattr__(self, name):
        return getattr(self._delegate, name)

    def run(self, *args, **kwargs):
        command = args[0] if args else kwargs.get("args")
        if (isinstance(command, (list, tuple)) and command and isinstance(command[0], str)
                and ntpath.basename(command[0]).lower() in {"icacls", "icacls.exe"}
                and kwargs.get("capture_output") is True and kwargs.get("text") is True
                and kwargs.get("encoding") is None and kwargs.get("errors") is None):
            kwargs = {**kwargs, "text": False}
        return self._delegate.run(*args, **kwargs)


def configure_windows_acl_capture():
    """Leave installed SDK files and all ACL/return-code decisions unchanged."""
    if sys.platform != "win32":
        return False
    from kiwoom.core import platform_paths
    if not isinstance(platform_paths.subprocess, _ACLSubprocess):
        platform_paths.subprocess = _ACLSubprocess(platform_paths.subprocess)
    return True
