import builtins
import importlib
import inspect
import io
import itertools
from typing import Any, Optional

import hy  # to set builtin macros
import hy.models
from hy.reader.hy_reader import HyReader
from hy.reader.mangling import unmangle
from lsprotocol import types
from pygls.workspace import TextDocument


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


def _inspect_icon(obj: Any) -> Optional[types.CompletionItemKind]:
    if inspect.ismodule(obj):
        return types.CompletionItemKind.Module
    elif inspect.isabstract(obj):
        return types.CompletionItemKind.Interface
    elif inspect.isclass(obj):
        return types.CompletionItemKind.Class
    elif inspect.ismethod(obj):
        return types.CompletionItemKind.Method
    elif inspect.isfunction(obj):
        return types.CompletionItemKind.Function
    elif inspect.isgeneratorfunction(obj):
        return types.CompletionItemKind.Function
    elif inspect.isgenerator(obj):
        return types.CompletionItemKind.Function
    else:
        return None


def _import_completions(
    import_expression: hy.models.Expression,
) -> list[types.CompletionItem]:
    completions = []
    i = 1
    while i < len(import_expression):
        assert isinstance(import_expression[i], hy.models.Symbol)

        if i + 1 < len(import_expression) and isinstance(
            import_expression[i + 1], hy.models.List
        ):
            # from ... import ...
            try:
                module = importlib.import_module(import_expression[i][:])
                for import_sym_name in import_expression[i + 1]:
                    import_sym = getattr(module, import_sym_name[:])
                    completions.append(
                        types.CompletionItem(
                            label=import_sym.__name__,
                            kind=_inspect_icon(import_sym)
                            or types.CompletionItemKind.Property,
                            documentation=inspect.getdoc(import_sym),
                            label_details=types.CompletionItemLabelDetails(
                                detail=_unmangle_signature(
                                    inspect.signature(import_sym)
                                ),
                                description=inspect.getmodule(import_sym).__name__,
                            ),
                        )
                    )
            except:
                pass
            i += 2
        else:
            # import ...
            try:
                module = importlib.import_module(import_expression[i][:])
                completions.append(
                    types.CompletionItem(
                        label=module.__name__,
                        kind=types.CompletionItemKind.Module,
                        documentation=inspect.getdoc(module),
                    )
                )
            except:
                completions.append(
                    types.CompletionItem(
                        label=import_expression[i][:],
                        kind=types.CompletionItemKind.Module,
                    )
                )
            i += 1

    return completions


def _document_completions(doc: TextDocument) -> list[types.CompletionItem]:
    reader = HyReader(use_current_readers=False)
    forms = reader.parse(io.StringIO(doc.source))

    completion_items = []

    for top_level_form in forms:
        if isinstance(top_level_form, hy.models.Expression):
            if len(top_level_form) > 0:
                if top_level_form[0][:] == "import":
                    # get completions from import statement
                    completion_items.extend(_import_completions(top_level_form))

    return completion_items


def get_completion(doc: TextDocument, pos: types.Position):
    return types.CompletionList(
        is_incomplete=False,
        items=_core_macro_completions() + _document_completions(doc),
    )
