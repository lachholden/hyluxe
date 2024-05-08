"""Converting from our classes to LSP-interoperable forms."""

import inspect
from typing import Optional

from hy.reader.mangling import unmangle
from hyluxe.server.tagged_form_tree import (
    ScopedIdentifier,
    ScopedIdentifierKind,
    TaggedFormTree,
)
from lsprotocol import types as lsp


def tagged_form_to_range(form: TaggedFormTree) -> lsp.Range:
    return lsp.Range(
        start=lsp.Position(line=form.start_line - 1, character=form.start_col - 1),
        end=lsp.Position(line=form.end_line - 1, character=form.end_col),
    )


def unmangle_signature(sig: inspect.Signature) -> str:
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


def convert_scoped_identifier_kind(
    kind: ScopedIdentifierKind,
) -> Optional[lsp.CompletionItemKind]:
    if kind == ScopedIdentifierKind.Module:
        return lsp.CompletionItemKind.Module
    elif kind == ScopedIdentifierKind.Variable:
        return lsp.CompletionItemKind.Variable
    elif kind == ScopedIdentifierKind.HyMacro:
        return lsp.CompletionItemKind.Function
    elif kind == ScopedIdentifierKind.HyReader:
        return lsp.CompletionItemKind.Operator
    elif kind == ScopedIdentifierKind.HyMacroCore:
        return lsp.CompletionItemKind.Keyword


def scoped_identifier_to_completion(ident: ScopedIdentifier) -> lsp.CompletionItem:
    return lsp.CompletionItem(
        label=ident.name,
        kind=convert_scoped_identifier_kind(ident.kind),
        documentation=ident.documentation,
        label_details=lsp.CompletionItemLabelDetails(
            detail=unmangle_signature(ident.signature) if ident.signature else None,
            description=ident.module_path,
        ),
    )


def scoped_identifier_kind_to_semantic_token_type(
    kind: ScopedIdentifierKind,
) -> Optional[lsp.SemanticTokenTypes]:
    if kind == ScopedIdentifierKind.Module:
        return lsp.SemanticTokenTypes.Namespace
    elif kind == ScopedIdentifierKind.Variable:
        return lsp.SemanticTokenTypes.Variable
    elif kind == ScopedIdentifierKind.HyMacro:
        return lsp.SemanticTokenTypes.Macro
    elif kind == ScopedIdentifierKind.HyReader:
        return lsp.SemanticTokenTypes.Macro
    elif kind == ScopedIdentifierKind.HyMacroCore:
        return lsp.SemanticTokenTypes.Keyword
