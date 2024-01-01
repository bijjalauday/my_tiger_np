# silence warnings
from .core.base_utils import silence_common_warnings as _silence_warnings
from .version import version

__version__ = version

_silence_warnings()
