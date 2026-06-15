"""teragent.intent — 意图分类子系统

多层漏斗分类器 + 确认门控
"""

from teragent.intent.classifier import IntentClassifier, IntentType
from teragent.intent.confirmation import _CONFIRM_TIMEOUT, _M1_CONFIRM_TIMEOUT, ConfirmationGate

# Re-export timeout constants with public names (the underscore-prefixed names
# in confirmation.py are implementation details; the public API uses these aliases).
CONFIRM_TIMEOUT = _CONFIRM_TIMEOUT
M1_CONFIRM_TIMEOUT = _M1_CONFIRM_TIMEOUT

__all__ = [
    "IntentClassifier",
    "IntentType",
    "ConfirmationGate",
    "CONFIRM_TIMEOUT",
    "M1_CONFIRM_TIMEOUT",
]
