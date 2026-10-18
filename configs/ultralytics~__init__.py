"""Local Ultralytics fork used for paired RGB-IR OBB detection.

The source is split into four roots for GitHub one-level file packaging.
Only the OBB and detection APIs are published by this reduced distribution.
"""
from pathlib import Path
__version__ = '8.0.238+etfnetsjpa.10.2'
_PACKAGE_DIR = Path(__file__).resolve().parent
_REPOSITORY_DIR = _PACKAGE_DIR.parent
for _name in ('ultra_modeling', 'ultra_runtime', 'ultra_services'):
    _component = _REPOSITORY_DIR / _name
    if _component.is_dir():
        __path__.append(str(_component))
from ultralytics.models import YOLO
from ultralytics.utils import SETTINGS as settings
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download
__all__ = ('__version__', 'YOLO', 'checks', 'download', 'settings')
