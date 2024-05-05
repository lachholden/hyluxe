import builtins
import inspect
import io
from typing import Optional

import hy  # to set builtin macros
from hy.reader.hy_reader import HyReader
from hy.reader.mangling import unmangle
from lsprotocol import types


def _unmangle_signature(sig: inspect.Signature) -> str:
    # TODO destructuring?
    sig_str = ""
    for parameter in sig.parameters.values():
        if parameter.name == "_hy_compiler":
            continue

        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            sig_str += " #**"
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            sig_str += " #*"

        if parameter.annotation != inspect.Parameter.empty:
            annotated_name = f"#^ {unmangle(parameter.name)} {unmangle(parameter.annotation.__name__)}"
        else:
            annotated_name = unmangle(parameter.name)

        if parameter.default != inspect.Parameter.empty:
            sig_str += f" [{annotated_name} {parameter.default}]"
        else:
            sig_str += f" {annotated_name}"

    return sig_str


def _core_macro_completions() -> list[types.CompletionItem]:
    return [
        types.CompletionItem(
            label=unmangle(func_name),
            kind=types.CompletionItemKind.Keyword,
            documentation=inspect.getdoc(func),
            label_details=types.CompletionItemLabelDetails(
                detail=_unmangle_signature(inspect.signature(func)),
                description=inspect.getmodule(func).__name__,
            ),
        )
        for func_name, func in builtins._hy_macros.items()
    ]


def get_completion():
    return types.CompletionList(is_incomplete=False, items=_core_macro_completions())
