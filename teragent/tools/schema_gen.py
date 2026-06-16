"""teragent.tools.schema_gen — Schema 自动生成

从 Python 函数签名自动生成 JSON Schema，用于 OpenAI function calling 格式。

支持:
- 基础类型：str, int, float, bool, list, dict
- Annotated[str, "description"]
- Annotated[str, Field(description=..., max_length=...)]
- Optional[type]
- 默认值
- 跳过 ctx/context/RunContext 参数

类型映射:
    Python 类型       → JSON Schema 类型
    str              → {"type": "string"}
    int              → {"type": "integer"}
    float            → {"type": "number"}
    bool             → {"type": "boolean"}
    list             → {"type": "array"}
    dict             → {"type": "object"}
    Optional[X]      → {"type": ["X_type", "null"]}
    list[ItemType]   → {"type": "array", "items": {...}}
"""
from __future__ import annotations

import inspect
from typing import (
    Any,
    Callable,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)


# Python type → JSON Schema type mapping
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _unwrap_optional(tp: type) -> tuple[type, bool]:
    """检查是否为 Optional 类型，返回 (内部类型, 是否Optional)

    Optional[X] 等价于 X | None，即 Union[X, None]。
    本函数从 Union 中提取非 None 的类型。

    Args:
        tp: 待检查的类型

    Returns:
        (内部类型, 是否为Optional) 元组
    """
    origin = get_origin(tp)
    if origin is Union:
        args = get_args(tp)
        none_types = [a for a in args if a is type(None)]
        non_none_types = [a for a in args if a is not type(None)]
        if none_types and non_none_types:
            return non_none_types[0], True
    return tp, False


def _extract_annotated_info(tp: type) -> tuple[type, dict]:
    """从 Annotated 类型提取信息和描述

    支持:
    - Annotated[str, "description"]
    - Annotated[str, Field(description=..., max_length=...)]
    - 任何具有 description 和约束属性的对象

    Args:
        tp: 待提取的 Annotated 类型

    Returns:
        (基础类型, 约束信息字典) 元组
    """
    origin = get_origin(tp)
    if origin is not None and hasattr(tp, "__metadata__"):
        base_type = origin
        description = ""
        constraints = {}
        for meta in tp.__metadata__:
            if isinstance(meta, str):
                description = meta
            elif hasattr(meta, "description"):
                description = meta.description
                for attr in (
                    "max_length", "min_length",
                    "ge", "le", "gt", "lt",
                    "multiple_of",
                ):
                    if hasattr(meta, attr):
                        val = getattr(meta, attr)
                        if val is not None:
                            constraints[_py_attr_to_json(attr)] = val
        return base_type, {"description": description, **constraints}
    return tp, {}


def _py_attr_to_json(attr: str) -> str:
    """将 Python 属性名转换为 JSON Schema 键名

    Args:
        attr: Python 属性名（如 max_length）

    Returns:
        JSON Schema 键名（如 maxLength）
    """
    mapping = {
        "max_length": "maxLength",
        "min_length": "minLength",
        "ge": "minimum",
        "le": "maximum",
        "gt": "exclusiveMinimum",
        "lt": "exclusiveMaximum",
        "multiple_of": "multipleOf",
    }
    return mapping.get(attr, attr)


def _type_to_schema(tp: type) -> dict:
    """将 Python 类型转换为 JSON Schema

    递归处理 Optional、Union、Annotated、list[ItemType] 等复合类型。

    Args:
        tp: Python 类型

    Returns:
        JSON Schema 字典
    """
    # Handle Optional (Union[X, None])
    inner, is_optional = _unwrap_optional(tp)
    if is_optional:
        schema = _type_to_schema(inner)
        if "type" in schema:
            # Convert "type": "string" → "type": ["string", "null"]
            existing_type = schema.pop("type")
            schema = {
                "type": [existing_type, "null"],
                **schema,
            }
        return schema

    # Handle non-Optional Union (e.g., Union[str, int])
    origin = get_origin(tp)
    if origin is Union:
        args = get_args(tp)
        # Collect JSON Schema types for each Union member
        member_schemas = [_type_to_schema(arg) for arg in args]
        member_types = []
        extra_keys = {}
        for ms in member_schemas:
            if "type" in ms:
                t = ms["type"]
                if isinstance(t, list):
                    member_types.extend(t)
                else:
                    member_types.append(t)
                # Merge non-type keys (like description) from first member
                for k, v in ms.items():
                    if k != "type" and k not in extra_keys:
                        extra_keys[k] = v
            else:
                # Complex schema — use anyOf
                return {"anyOf": member_schemas}
        result = {"type": member_types}
        result.update(extra_keys)
        return result

    # Handle Annotated
    inner, annotated_info = _extract_annotated_info(inner)
    if annotated_info:
        base_schema = _type_to_schema(inner)
        return {**base_schema, **annotated_info}

    # Handle basic types
    if inner in _TYPE_MAP:
        return {"type": _TYPE_MAP[inner]}

    # Handle list[ItemType]
    if origin is list:
        args = get_args(inner)
        if args:
            return {"type": "array", "items": _type_to_schema(args[0])}
        return {"type": "array"}

    # Handle dict
    if origin is dict or inner is dict:
        return {"type": "object"}

    # Fallback — 未知类型默认为 string
    return {"type": "string"}


# Parameters to skip when generating schema
_SKIP_PARAMS: set[str] = {"ctx", "context", "self", "cls"}


def generate_schema_from_hints(fn: Callable, requires_context: bool = False) -> dict:
    """从函数签名自动生成 JSON Schema

    生成 OpenAI function calling 格式的 JSON Schema，包含 properties
    和 required 字段。

    Args:
        fn: Python 函数
        requires_context: 是否包含 context 参数（ctx/context/RunContext 类型参数）

    Returns:
        OpenAI function calling 格式的 JSON Schema 字典

    示例:
        def search(query: str, limit: int = 10) -> list:
            '''Search for items'''
            ...

        schema = generate_schema_from_hints(search)
        # {
        #     "type": "object",
        #     "properties": {
        #         "query": {"type": "string"},
        #         "limit": {"type": "integer"},
        #     },
        #     "required": ["query"],
        # }
    """
    try:
        hints = get_type_hints(fn, include_extras=True)
    except Exception:
        hints = {}

    sig = inspect.signature(fn)
    properties: dict[str, dict] = {}
    required: list[str] = []

    skip = set(_SKIP_PARAMS)
    if not requires_context:
        # Also skip RunContext-typed params
        for pname, hint_tp in hints.items():
            type_str = str(hint_tp)
            if "RunContext" in type_str:
                skip.add(pname)

    for param_name, param in sig.parameters.items():
        if param_name in skip:
            continue

        # Get type hint
        hint = hints.get(param_name)

        if hint is not None:
            prop = _type_to_schema(hint)
        else:
            # No hint, default to string
            prop = {"type": "string"}

        properties[param_name] = prop

        # Required if no default value
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required

    return schema
