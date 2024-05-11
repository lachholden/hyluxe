import itertools
from collections import OrderedDict
from typing import Optional

import hy  # to set builtin macros
import hy.reader.exceptions
from hy.reader.mangling import unmangle
from hyluxe.server.lsp_conversions import (
    SCOPED_IDENTIFIER_KIND_TO_SEMANTIC_TOKEN_TYPE,
    hover_doc,
    scoped_identifier_to_completion,
    tagged_form_to_range,
)
from hyluxe.server.tagged_form_tree import (
    ScopedIdentifierKind,
    TaggedFormTree,
    attr_name_to_identifier_try_getattr,
)
from lsprotocol import types
from pygls.server import LanguageServer


class HyLanguageServer(LanguageServer):
    def __init__(self, *args):
        super().__init__(*args)

        self.tagged_trees = {}


hy_server = HyLanguageServer("hyluxe-hy", "v0.1")


@hy_server.feature(types.TEXT_DOCUMENT_DID_OPEN)
def did_open(server: HyLanguageServer, params: types.DidOpenTextDocumentParams):
    doc = server.workspace.get_document(params.text_document.uri)
    tagged_model = TaggedFormTree.parse_hy(doc.source)
    try:
        tagged_model = TaggedFormTree.parse_hy(doc.source)
        server.tagged_trees[doc.uri] = tagged_model
    except hy.reader.exceptions.LexException as e:
        server.show_message_log(str(e))
        del server.tagged_trees[doc.uri]


@hy_server.feature(types.TEXT_DOCUMENT_DID_CHANGE)
def did_change(server: HyLanguageServer, params: types.DidChangeTextDocumentParams):
    server.show_message_log(str(params.content_changes))
    doc = server.workspace.get_document(params.text_document.uri)
    try:
        tagged_model = TaggedFormTree.parse_hy(doc.source)
        server.tagged_trees[doc.uri] = tagged_model
    except hy.reader.exceptions.LexException as e:
        # We failed to parse. *If* this is because we just added a single . and this
        # made everything unworkable, then the previous analysis should hold well
        # enough & this way we still get good autocomplete for the next dotted segment
        if (
            not len(params.content_changes) == 1
            or not params.content_changes[0].text == "."
        ):
            server.show_message_log(str(e))
            del server.tagged_trees[doc.uri]


@hy_server.feature(
    types.TEXT_DOCUMENT_COMPLETION,
    types.CompletionOptions(trigger_characters=["."]),
)
def completions(
    server: HyLanguageServer,
    params: Optional[types.CompletionParams] = None,
) -> types.CompletionList:
    """Returns completion items."""
    doc = server.workspace.get_document(params.text_document.uri)
    try:
        tagged_model = server.tagged_trees[doc.uri]
    except KeyError:
        return
    enclosing_models = tagged_model.get_models_enclosing_position(
        params.position.line + 1, params.position.character + 1
    )

    # Get context surrounding current typing - which character starts this form, and how
    # much we've typed so far
    current_form = ""
    current_form_start = "\n"
    current_line = params.position.line
    current_col = params.position.character - 1

    while True:
        next_char = doc.source.splitlines()[current_line][current_col]
        if current_col < 0 or next_char in [" ", "[", "(", "{", "`", "~", '"']:
            if current_col >= 0:
                current_form_start = next_char
            break

        current_col -= 1
        current_form = next_char + current_form

    server.show_message_log(f"{current_form} // {current_form_start}")

    # now, based on context, we can filter the in-scope identifiers
    in_scope_identifiers = list(
        itertools.chain.from_iterable(
            [model.scoped_identifiers for model in enclosing_models]
        )
    )
    show_identifiers = in_scope_identifiers.copy()

    # if this form is not first in a (func call), then there's no need to complete macros
    if current_form_start != "(":
        show_identifiers = [
            i
            for i in show_identifiers
            if i.kind
            not in [ScopedIdentifierKind.HyMacro, ScopedIdentifierKind.HyMacroCore]
        ]

    # If we've started a dotted identifier, then only get matches that continue it
    if "." in current_form:
        show_identifiers = [
            i for i in show_identifiers if i.name.startswith(current_form)
        ]
        for i in show_identifiers:
            i.name = i.name.removeprefix(current_form)

    # If we've *just* started a dotted identifier, then also query its attrs
    if current_form.endswith("."):
        if current_obj := {i.name: i.py_obj for i in in_scope_identifiers}.get(
            current_form.rstrip(".")
        ):
            show_identifiers += [
                attr_name_to_identifier_try_getattr(current_obj, unmangle(n))
                for n in dir(current_obj)
            ]

    # TODO import and require only suggest modules

    return types.CompletionList(
        is_incomplete=False,
        items=[scoped_identifier_to_completion(ident) for ident in show_identifiers],
    )


@hy_server.feature(types.TEXT_DOCUMENT_HOVER)
def hover(
    server: HyLanguageServer, params: Optional[types.HoverParams] = None
) -> Optional[types.Hover]:
    doc = server.workspace.get_document(params.text_document.uri)
    try:
        tagged_model = server.tagged_trees[doc.uri]
    except KeyError:
        return
    enclosing_models = tagged_model.get_models_enclosing_position(
        params.position.line + 1, params.position.character + 1
    )
    for m in enclosing_models:
        if m.this_identifier:
            enclosing_model = m
            break
    else:
        return
    return types.Hover(
        contents=hover_doc(enclosing_model.this_identifier),
        range=tagged_form_to_range(enclosing_model),
    )


_semantic_tokens_legend = list(types.SemanticTokenTypes._value2member_map_.keys())

# ^ return (function args #* argss)


@hy_server.feature(
    types.TEXT_DOCUMENT_SEMANTIC_TOKENS_FULL,
    types.SemanticTokensLegend(
        token_types=_semantic_tokens_legend,
        token_modifiers=[],
    ),
)
def semantic_tokens(
    server: HyLanguageServer, params: types.SemanticTokensParams
) -> types.SemanticTokens:

    doc = server.workspace.get_document(params.text_document.uri)
    try:
        tagged_model = server.tagged_trees[doc.uri]
    except KeyError:
        return

    token_data = []
    last_line = 0
    last_start = 0

    for form in tagged_model.forms_with_identifiers():
        semantic_kind = SCOPED_IDENTIFIER_KIND_TO_SEMANTIC_TOKEN_TYPE[
            form.this_identifier.kind
        ]
        if semantic_kind is None or form.start_line != form.end_line:
            continue
        if form.start_line - 1 != last_line:
            last_start = 0
        line_delta = form.start_line - 1 - last_line
        start_delta = form.start_col - 1 - last_start
        length = form.end_col - form.start_col + 1
        legend_index = _semantic_tokens_legend.index(semantic_kind.value)
        token_data += [line_delta, start_delta, length, legend_index, 0]
        last_line = form.start_line - 1
        last_start = form.start_col - 1

    return types.SemanticTokens(data=token_data)


def main():
    hy_server.start_io()


if __name__ == "__main__":
    main()
