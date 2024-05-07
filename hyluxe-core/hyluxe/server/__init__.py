import itertools
from typing import Optional

import hy  # to set builtin macros
from hyluxe.server.lsp_conversions import (scoped_identifier_to_completion,
                                           tagged_form_to_range)
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
    types.CompletionOptions(trigger_characters=["(", " "]),
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
) -> types.Hover:
    doc = server.workspace.get_document(params.text_document.uri)
    tagged_model = TaggedFormTree.parse_hy(doc.source)
    enclosing_model = tagged_model.get_models_enclosing_position(
        params.position.line + 1, params.position.character + 1
    )[0]
    server.show_message_log(
        f"line: {params.position.line}  col: {params.position.character}"
    )
    server.show_message_log(str(enclosing_model))
    return types.Hover(
        contents=types.MarkupContent(
            kind=types.MarkupKind.Markdown,
            value=str(enclosing_model.this_identifier) or "HOVER",
        ),
        range=tagged_form_to_range(enclosing_model),
    )


# @hy_server.feature(
#     types.TEXT_DOCUMENT_SEMANTIC_TOKENS_FULL,
#     types.SemanticTokensLegend(token_types=["operator"], token_modifiers=[]),
# )
# def semantic_tokens(
#     server: HyLanguageServer, params: types.SemanticTokensParams
# ) -> types.SemanticTokens:
#     doc = server.workspace.get_document(params.text_document.uri)
#     tagged_model = TaggedFormTree.parse_hy(doc.source)
#     for form in tagged_model.forms_with_identifiers():
#         pass
#         types.SemanticTokensLegend


def main():
    hy_server.start_io()


if __name__ == "__main__":
    main()
