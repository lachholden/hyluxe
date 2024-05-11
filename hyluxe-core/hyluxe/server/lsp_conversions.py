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

SCOPED_IDENTIFIER_KIND_TO_COMPLETION_ITEM_KIND = {
    ScopedIdentifierKind.Module: lsp.CompletionItemKind.Module,
    ScopedIdentifierKind.Variable: lsp.CompletionItemKind.Variable,
    ScopedIdentifierKind.HyMacro: lsp.CompletionItemKind.Function,
    ScopedIdentifierKind.HyReader: lsp.CompletionItemKind.Operator,
    ScopedIdentifierKind.HyMacroCore: lsp.CompletionItemKind.Keyword,
    ScopedIdentifierKind.Class: lsp.CompletionItemKind.Class,
    ScopedIdentifierKind.Method: lsp.CompletionItemKind.Method,
    ScopedIdentifierKind.Function: lsp.CompletionItemKind.Function,
}


SCOPED_IDENTIFIER_KIND_TO_SEMANTIC_TOKEN_TYPE = {
    ScopedIdentifierKind.Module: lsp.SemanticTokenTypes.Namespace,
    ScopedIdentifierKind.Variable: lsp.SemanticTokenTypes.Variable,
    ScopedIdentifierKind.HyMacro: lsp.SemanticTokenTypes.Macro,
    ScopedIdentifierKind.HyReader: lsp.SemanticTokenTypes.Macro,
    ScopedIdentifierKind.HyMacroCore: lsp.SemanticTokenTypes.Keyword,
    ScopedIdentifierKind.Class: lsp.SemanticTokenTypes.Class,
    ScopedIdentifierKind.Method: lsp.SemanticTokenTypes.Method,
    ScopedIdentifierKind.Function: lsp.SemanticTokenTypes.Function,
}


def tagged_form_to_range(form: TaggedFormTree) -> lsp.Range:
    return lsp.Range(
        start=lsp.Position(line=form.start_line - 1, character=form.start_col - 1),
        end=lsp.Position(line=form.end_line - 1, character=form.end_col),
    )


def unmangle_signature_parameters(sig: inspect.Signature) -> str:
    # TODO destructuring? return values?
    sig_str = ""
    for parameter in sig.parameters.values():
        if parameter.name == "_hy_compiler":
            continue

        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            sig_str += " #**"
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            sig_str += " #*"

        if parameter.annotation != inspect.Parameter.empty:
            annotated_name = f"#^ {unmangle(parameter.name)} {unmangle(getattr(parameter.annotation, '__name__', str(parameter.annotation)))}"
        else:
            annotated_name = unmangle(parameter.name)

        if parameter.default != inspect.Parameter.empty:
            sig_str += f" [{annotated_name} {parameter.default}]"
        else:
            sig_str += f" {annotated_name}"

    return sig_str.lstrip(" ")


def unmangle_return_type(sig: inspect.Signature) -> Optional[str]:
    if sig.return_annotation != inspect.Signature.empty:
        return f"#^ {unmangle(getattr(sig.return_annotation, '__name__', str(sig.return_annotation)))}"


def full_signature_string(ident: ScopedIdentifier, include_kind=True) -> str:
    assert ident.signature
    sig_str = ""
    if include_kind:
        sig_str += f":{ident.kind.value} "
    if ret := unmangle_return_type(ident.signature):
        sig_str += ret + " "
    sig_str += f"({ident.parent_name}." if ident.parent_name else "("
    sig_str += f"{ident.name} {unmangle_signature_parameters(ident.signature)})"
    return sig_str


def scoped_identifier_to_completion(ident: ScopedIdentifier) -> lsp.CompletionItem:
    return lsp.CompletionItem(
        label=ident.name,
        kind=SCOPED_IDENTIFIER_KIND_TO_COMPLETION_ITEM_KIND[ident.kind],
        documentation=ident.documentation,
        label_details=lsp.CompletionItemLabelDetails(
            detail=(" " + full_signature_string(ident) if ident.signature else None),
            description=ident.parent_name or f":{ident.kind.value}",
        ),
    )


def hover_doc(ident: ScopedIdentifier) -> lsp.MarkupContent:
    hover = ""
    if ident.signature:
        hover += "```hy\n"
        hover += full_signature_string(ident)
        hover += "\n```\n\n---\n\n"

    if ident.documentation:
        hover += ident.documentation

    return lsp.MarkupContent(
        kind=lsp.MarkupKind.Markdown,
        value=hover,
    )
