import itertools
from collections import OrderedDict
from typing import Optional

import hy  # to set builtin macros
from hyluxe.server.lsp_conversions import (
    SCOPED_IDENTIFIER_KIND_TO_SEMANTIC_TOKEN_TYPE,
    hover_doc,
    scoped_identifier_to_completion,
    tagged_form_to_range,
)
from hyluxe.server.tagged_form_tree import ScopedIdentifier, TaggedFormTree
from lsprotocol import types
from pygls.server import LanguageServer


class HyLanguageServer(LanguageServer):
    pass


hy_server = HyLanguageServer("hyluxe-hy", "v0.1")


@hy_server.feature(types.TEXT_DOCUMENT_DID_OPEN)
def did_open(server: HyLanguageServer, params: types.DidOpenTextDocumentParams):
    doc = server.workspace.get_document(params.text_document.uri)
    tagged_model = TaggedFormTree.parse_hy(doc.source)
    server.show_message_log(str(tagged_model))


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
    tagged_model = TaggedFormTree.parse_hy(doc.source)
    enclosing_models = tagged_model.get_models_enclosing_position(
        params.position.line + 1, params.position.character + 1
    )
    return types.CompletionList(
        is_incomplete=False,
        items=[
            scoped_identifier_to_completion(ident)
            for ident in itertools.chain.from_iterable(
                [model.scoped_identifiers for model in enclosing_models]
            )
        ],
    )


@hy_server.feature(types.TEXT_DOCUMENT_HOVER)
def hover(
    server: HyLanguageServer, params: Optional[types.HoverParams] = None
) -> Optional[types.Hover]:
    doc = server.workspace.get_document(params.text_document.uri)
    tagged_model = TaggedFormTree.parse_hy(doc.source)
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
    tagged_model = TaggedFormTree.parse_hy(doc.source)

    token_data = []
    last_line = 0
    last_start = 0

    for form in tagged_model.forms_with_identifiers():
        semantic_kind = SCOPED_IDENTIFIER_KIND_TO_SEMANTIC_TOKEN_TYPE[
            form.this_identifier.kind
        ]
        if semantic_kind is None or form.start_line != form.end_line:
            continue
        server.show_message_log(repr(form))
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
