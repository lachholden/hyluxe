from lsprotocol import types
from pygls.server import LanguageServer


class HyLanguageServer(LanguageServer):
    pass


hy_server = HyLanguageServer("hyluxe-hy", "v0.1")


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
