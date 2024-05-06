import builtins
import inspect
import io
import itertools
from typing import Optional

import hy  # to set builtin macros
from hy.reader.hy_reader import HyReader
from hy.reader.mangling import unmangle
from hyluxe.server.tagged_models import ScopedIdentifier, TaggedModel
from lsprotocol import types
from pygls.server import LanguageServer


class HyLanguageServer(LanguageServer):
    pass


hy_server = HyLanguageServer("hyluxe-hy", "v0.1")


@hy_server.feature(types.TEXT_DOCUMENT_DID_OPEN)
def did_open(server: HyLanguageServer, params: types.DidOpenTextDocumentParams):
    doc = server.workspace.get_document(params.text_document.uri)
    reader = HyReader(use_current_readers=False)
    forms = reader.parse(io.StringIO(doc.source))
    tagged_model = TaggedModel.create_root_model(forms)
    server.show_message_log(str(tagged_model))


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


def scoped_identifier_to_completion(ident: ScopedIdentifier) -> types.CompletionItem:
    return types.CompletionItem(
        label=ident.name,
        kind=ident.kind,
        documentation=ident.documentation,
        label_details=types.CompletionItemLabelDetails(
            detail=_unmangle_signature(ident.signature) if ident.signature else None,
            description=ident.module_path,
        ),
    )


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
    reader = HyReader(use_current_readers=False)
    forms = reader.parse(io.StringIO(doc.source))
    tagged_model = TaggedModel.create_root_model(forms)
    enclosing_models = tagged_model.get_models_enclosing_position(
        params.position.line - 1, params.position.character - 1
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
    reader = HyReader(use_current_readers=False)
    forms = reader.parse(io.StringIO(doc.source))
    tagged_model = TaggedModel.create_root_model(forms)
    enclosing_model = tagged_model.get_models_enclosing_position(
        params.position.line - 1, params.position.character - 1
    )[0]
    server.show_message_log(str(enclosing_model))
    return types.Hover(
        contents=types.MarkupContent(
            kind=types.MarkupKind.Markdown,
            value=str(enclosing_model.identifier) or "HOVER",
        )
    )


def main():
    hy_server.start_io()


if __name__ == "__main__":
    main()
