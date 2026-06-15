"""teragent.core.compilers — TAP Compiler implementations

Importing this module registers all built-in compilers with TAPCompilerRegistry.

Available compilers:
    - default: Generic OpenAI-compatible (multi-turn context injection)
    - glm: GLM-optimized (recency effect, Chinese constraints)
    - anthropic: Anthropic-optimized (XML tags, system+user separation)
    - deepseek: DeepSeek-optimized (minimalist, inlined constraints)
    - deepseek_v4: DeepSeek V4 (thinking mode, Flash/Pro variants, 1M context)
    - glm_5: GLM-5 (recency effect, 200K compression, long-horizon tasks)
    - glm_52: GLM-52 (1M context, dual thinking, preserved reasoning)
    - glm_5v_turbo: GLM-5V-Turbo (vision analysis, multimodal)
    - minimax_m3: MiniMax M3 (multimodal, MSA full-text injection, desktop ops)
"""

from teragent.core.compiler import TAPCompilerRegistry
from teragent.core.compilers import anthropic as _anthropic
from teragent.core.compilers import deepseek as _deepseek
from teragent.core.compilers import deepseek_v4 as _deepseek_v4

# Import compiler modules to trigger registration
from teragent.core.compilers import default as _default
from teragent.core.compilers import glm as _glm
from teragent.core.compilers import glm_5 as _glm_5
from teragent.core.compilers import glm_52 as _glm_52
from teragent.core.compilers import glm_5v_turbo as _glm_5v_turbo
from teragent.core.compilers import minimax_m3 as _minimax_m3

__all__ = [
    "DefaultCompiler",
    "GLMCompiler",
    "AnthropicCompiler",
    "DeepSeekCompiler",
    "DeepSeekV4Compiler",
    "GLM5Compiler",
    "GLM52Compiler",
    "GLM5VTurboCompiler",
    "MiniMaxM3Compiler",
    "TAPCompilerRegistry",
]

# Re-export compiler classes for convenience
DefaultCompiler = _default.DefaultCompiler
GLMCompiler = _glm.GLMCompiler
AnthropicCompiler = _anthropic.AnthropicCompiler
DeepSeekCompiler = _deepseek.DeepSeekCompiler
DeepSeekV4Compiler = _deepseek_v4.DeepSeekV4Compiler
GLM5Compiler = _glm_5.GLM5Compiler
GLM52Compiler = _glm_52.GLM52Compiler
GLM5VTurboCompiler = _glm_5v_turbo.GLM5VTurboCompiler
MiniMaxM3Compiler = _minimax_m3.MiniMaxM3Compiler
