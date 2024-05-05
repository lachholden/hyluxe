import builtins
import inspect
import io
from typing import Optional

import hy  # to set builtin macros
from hy.reader.hy_reader import HyReader
from hy.reader.mangling import unmangle
from hyluxe.server.completion import get_completion
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
    server.show_message_log(str(next(forms)))


@hy_server.feature(
    types.TEXT_DOCUMENT_COMPLETION,
    types.CompletionOptions(trigger_characters=["(", " "]),
)
def completions(
    params: Optional[types.CompletionParams] = None,
) -> types.CompletionList:
    """Returns completion items."""
    return get_completion()


@hy_server.feature(types.TEXT_DOCUMENT_HOVER)
def hover(params: types.HoverParams):
    return types.Hover(
        contents=types.MarkupContent(
            kind=types.MarkupKind.Markdown,
            value="*THIiiiS* is **Hy** hover",
        ),
        range=types.Range(start=params.position, end=params.position),
    )


def main():
    hy_server.start_io()


if __name__ == "__main__":
    main()
