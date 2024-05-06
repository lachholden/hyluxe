from __future__ import annotations

import builtins
import inspect
import io
import itertools
import types
from dataclasses import dataclass, field
from math import inf
from typing import Any, Iterable, Optional

import hy  # for side-effects
import hy.models
import toolz.functoolz
from funcparserlib.parser import NoParseError, many, maybe
from hy.core.result_macros import importlike, module_name_pattern
from hy.model_patterns import pexpr, sym
from hy.reader.hy_reader import HyReader
from hy.reader.mangling import unmangle
from lsprotocol import types as lsp


@dataclass
class ScopedIdentifier:
    name: str
    kind: lsp.CompletionItemKind
    documentation: Optional[str] = None
    signature: Optional[inspect.Signature] = None
    module_path: Optional[str] = None
    py_obj: Optional[Any] = None


# CORE MACROS


def _core_macro_completions() -> list[ScopedIdentifier]:
    return [
        ScopedIdentifier(
            name=unmangle(func_name),
            kind=lsp.CompletionItemKind.Keyword,
            documentation=inspect.getdoc(func),
            signature=inspect.signature(func),
            module_path=inspect.getmodule(func).__name__,
            py_obj=func,
        )
        for func_name, func in builtins._hy_macros.items()
    ]


# PARSING IMPORTS

_import_parser = sym("import") + many(module_name_pattern + maybe(importlike))


def _match_import_expr(model: hy.models.Object) -> list[ScopedIdentifier]:
    """Attempts to parse the given model as an import expression.

    Returns a list of ScopedIdentifiers defined if model is an import statement, or the
    empty list otherwise.
    """
    if not isinstance(model, hy.models.Expression):
        return []
    try:
        entries = _import_parser.parse(model)
    except NoParseError:
        return []

    # Successfully parsed as import expression
    new_scoped_identifiers = []
    for mod, as_or_froms in entries:
        if as_or_froms is None:
            # "import xyz"
            new_scoped_identifiers.append(
                ScopedIdentifier(
                    name=mod[:],
                    kind=lsp.CompletionItemKind.Module,
                )
            )
        elif as_or_froms[0] == hy.models.Keyword("as"):
            # "import xyz as abc"
            # TODO as
            pass
        else:
            # "from abc import def, ghi as jk"
            for as_or_from in as_or_froms:
                for ident, ident_as in as_or_from:
                    if ident_as is None:
                        new_scoped_identifiers.append(
                            ScopedIdentifier(
                                name=ident[:],
                                kind=lsp.CompletionItemKind.Variable,
                            )
                        )
                    else:
                        # TODO as
                        pass
        return new_scoped_identifiers


@dataclass(frozen=True)
class TaggedFormTree:
    """Wraps a `hy.models.Object` to attach data useful for the language server."""

    # tree structure
    children: list[TaggedFormTree]

    # file location
    start_line: int
    start_col: int
    end_line: int
    end_col: int

    # tagged information
    scoped_identifiers: list[ScopedIdentifier]
    this_identifier: Optional[ScopedIdentifier]

    def get_models_enclosing_position(
        self, line: int, col: int
    ) -> list[TaggedFormTree]:
        """Recursively get the list of TaggedModels enclosing a given position.

        Models in the returned list are ordered from the bottom of the tree up
        """
        if (
            (self.start_line < line < self.end_line)
            or (self.start_line == line and self.start_col <= col)
            or (self.end_line == line and self.end_col >= col)
        ):
            inner_lists = [
                c.get_models_enclosing_position(line, col) for c in self.children
            ]
            return list(itertools.chain.from_iterable(inner_lists + [[self]]))
        else:
            return []

    @classmethod
    def parse_hy(cls, source: str):
        reader = HyReader(use_current_readers=False)
        line_count = len(source.splitlines())
        col_count = len(source.splitlines()[-1])
        forms = reader.parse(io.StringIO(source))
        # create a root Sequence, as hy doesn't by default
        root = hy.models.Sequence(forms)
        root.start_line = 1
        root.end_line = line_count
        root.start_column = 1
        root.end_column = col_count
        # construct the tree from the root
        return cls.construct_from_hy_model(root, is_root=True)

    @classmethod
    def construct_from_hy_model(cls, hy_model: hy.models.Object, is_root=False):
        children = []
        scoped_identifiers = []

        # If this is the root model, then all of the core macros come in scope here.
        if is_root:
            scoped_identifiers.extend(_core_macro_completions())

        # If this model is a sequence, check the children for any expressions that
        # introduce identifiers at *this* scope one level higher (e.g. import, setv,
        # etc.), while also recursively constructing the tree.
        if isinstance(hy_model, hy.models.Sequence):
            for child_model in hy_model:
                scoped_identifiers.extend(_match_import_expr(child_model))
                children.append(cls.construct_from_hy_model(child_model))

        return TaggedFormTree(
            children=children,
            start_line=hy_model.start_line,
            start_col=hy_model.start_column,
            end_line=hy_model.end_line,
            end_col=hy_model.end_column,
            scoped_identifiers=scoped_identifiers,
            this_identifier=None,
        )
