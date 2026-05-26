"""teragent.core.compilers — TAP Compiler implementations

Importing this module registers all built-in compilers with TAPCompilerRegistry.

Available compilers:
    - default: Generic OpenAI-compatible (multi-turn context injection)
    - glm: GLM-optimized (recency effect, Chinese constraints)
    - anthropic: Anthropic-optimized (XML tags, system+user separation)
    - deepseek: DeepSeek-optimized (minimalist, inlined constraints)
"""

from teragent.core.compiler import TAPCompilerRegistry

# Import compiler modules to trigger registration
from teragent.core.compilers import default as _default
from teragent.core.compilers import glm as _glm
from teragent.core.compilers import anthropic as _anthropic
from teragent.core.compilers import deepseek as _deepseek

__all__ = [
    "DefaultCompiler",
    "GLMCompiler",
    "AnthropicCompiler",
    "DeepSeekCompiler",
    "TAPCompilerRegistry",
]

# Re-export compiler classes for convenience
DefaultCompiler = _default.DefaultCompiler
GLMCompiler = _glm.GLMCompiler
AnthropicCompiler = _anthropic.AnthropicCompiler
DeepSeekCompiler = _deepseek.DeepSeekCompiler
