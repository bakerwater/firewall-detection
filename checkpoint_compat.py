from __future__ import annotations

import importlib
import pathlib
import sys


def install_pathlib_checkpoint_compat() -> bool:
    """Make Python 3.13 pathlib checkpoints loadable on older Python versions.

    Python 3.13 serializes ``Path`` instances as ``pathlib._local.PosixPath``.
    Python 3.10 exposes the same class as ``pathlib.PosixPath`` and treats
    ``pathlib`` as a module rather than a package. Registering the standard
    module under the newer private name lets pickle resolve the class without
    changing the checkpoint or attempting to install a fake dependency.
    """

    if "pathlib._local" in sys.modules:
        return False
    try:
        importlib.import_module("pathlib._local")
        return False
    except ModuleNotFoundError:
        sys.modules["pathlib._local"] = pathlib
        return True

