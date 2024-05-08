from __future__ import annotations

import builtins
import inspect
import io
import itertools
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Generator, Iterable, Optional, Union

import hy  # for side-effects
import hy.models
import toolz.functoolz
from funcparserlib.parser import NoParseError, many, maybe
from hy.core.result_macros import importlike, module_name_pattern
from hy.model_patterns import pexpr, sym
from hy.reader.hy_reader import HyReader
from hy.reader.mangling import unmangle


class ScopedIdentifierKind(Enum):
    Module = auto()
    Variable = auto()
    HyMacro = auto()
    HyReader = auto()
    HyMacroCore = auto()


@dataclass
class ScopedIdentifier:
    name: str
    kind: ScopedIdentifierKind
    documentation: Optional[str] = None
    signature: Optional[inspect.Signature] = None
    module_path: Optional[str] = None
    py_obj: Optional[Any] = None


def dotted_name_components(model: hy.models.Object) -> Optional[list[str]]:
    """Converts a hy Object to its dotted name components, if possible.

    i.e. 'abc -> None
         '(. abc def) -> None
         '(. [abc def]) -> None
         'abc.def -> ["abc", "def"]

    Note that the output of the Hy reader represents the last two examples with a
    similar tree. We distinguish them by checking whether all parts have the same
    positions, which is the case for the 'abc.def form.
    """

    if isinstance(model, hy.models.Expression):
        if not model[0] == hy.models.Symbol("."):
            return

        if isinstance(model[1], Union[list, hy.models.Sequence]):
            dotted_segments = model[1]
        else:
            dotted_segments = model[1:]

        dotted_components = []
        for dot_segment in dotted_segments:
            if not isinstance(dot_segment, hy.models.Symbol):
                return None

            if (
                not dot_segment.start_column == model.start_column
                or not dot_segment.end_column == model.end_column
            ):
                return

            # TODO for now, don't handle method calls/relative imports with this parsing
            if dot_segment == hy.models.Symbol(None):
                return

            dotted_components.append(dot_segment[:])

        return dotted_components


# CORE MACROS
def _core_macro_completions() -> list[ScopedIdentifier]:
    return [
        ScopedIdentifier(
            name=unmangle(func_name),
            kind=ScopedIdentifierKind.HyMacroCore,
            documentation=inspect.getdoc(func),
            signature=inspect.signature(func),
            module_path=inspect.getmodule(func).__name__,
            py_obj=func,
        )
        for func_name, func in builtins._hy_macros.items()
    ]


# PARSING IMPORTS
_import_parser = sym("import") + many(module_name_pattern + maybe(importlike))


def _identifiers_from_plain_import(
    mod_expr: hy.models.Object,
) -> list[ScopedIdentifier]:

    if isinstance(mod_expr, hy.models.Symbol):
        return [ScopedIdentifier(name=mod_expr[:], kind=ScopedIdentifierKind.Module)]
    else:
        dotted_module_parts = dotted_name_components(mod_expr)
        if dotted_module_parts is None:
            return []
        return [
            ScopedIdentifier(
                name=".".join(dotted_module_parts[: i + 1]),
                kind=ScopedIdentifierKind.Module,
            )
            for i, dmp in enumerate(dotted_module_parts)
        ]


def _identifiers_from_from_import(
    mod_expr: hy.models.Object, ident_expr: hy.models.Object
) -> list[ScopedIdentifier]:
    return [
        ScopedIdentifier(name=ident_expr[:], kind=ScopedIdentifierKind.Variable)
    ]  # TODO


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
            new_scoped_identifiers.extend(_identifiers_from_plain_import(mod))
        elif as_or_froms[0] == hy.models.Keyword("as"):
            # "import xyz as abc"
            # TODO as
            pass
        else:
            # "from abc import def, ghi as jk"
            # include the actual base module too, even though it's not technically in
            # scope
            new_scoped_identifiers.extend(_identifiers_from_plain_import(mod))
            for as_or_from in as_or_froms:
                for ident, ident_as in as_or_from:
                    new_scoped_identifiers.extend(
                        _identifiers_from_from_import(mod, ident)
                    )
        return new_scoped_identifiers


class _NoOpReaderMacroTable:
    """When we parse a Hy source file to get the model tree, we don't want to have to
    run reader macros.

    This class can replace `HyReader.reader_macros`, and acts as though every called
    reader macro exists and does nothing.

    This has some tradeoffs:
      - pro: we don't have to find or evaluate reader macros as we parse (good for speed
        and reliability - don't want a buggy reader macro to crash the whole language
        server)
      - con: it restricts reader macros that we can feasibly use to those that maintain
        relatively standard form syntax (this is probably a fine restriction)
    """

    # hy_reader.py:390
    def __contains__(self, _):
        return True

    # hy_reader.py:391
    def __getitem__(self, _):
        return lambda _, __: None


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
            # multi-line model and we're on a line in the middle
            (self.start_line < line < self.end_line)
            # multi-line model and we're on the start line after the start col
            or (
                self.start_line == line
                and self.end_line > line
                and self.start_col <= col
            )
            # multi-line model and we're on the end line before the end col
            or (
                self.end_line == line and self.start_line < line and self.end_col >= col
            )
            # single-line model and we're between the start and end cols
            or (
                self.start_line == self.end_line == line
                and self.start_col <= col <= self.end_col
            )
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
        reader.reader_macros = _NoOpReaderMacroTable()  # type: ignore
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
    def construct_from_hy_model(
        cls,
        hy_model: hy.models.Object,
        is_root=False,
        in_scope_identifiers: dict[str, ScopedIdentifier] = {},
    ):
        this_level_scoped_identifiers = []

        # If this is the root model, then all of the core macros come in scope here.
        if is_root:
            this_level_scoped_identifiers.extend(_core_macro_completions())

        # If this model is a sequence, check the children for any expressions that
        # introduce identifiers at *this* scope one level higher (e.g. import, setv,
        # etc.)
        if isinstance(hy_model, hy.models.Sequence):
            for child_model in hy_model:
                this_level_scoped_identifiers.extend(_match_import_expr(child_model))

        # If this model is a single plain symbol, search for its string value in the
        # identifiers currently in scope
        this_identifier = None
        if isinstance(hy_model, hy.models.Symbol):
            symbol_name = hy_model[:]
            # We want to skip for now the macros that can be invoked directly by syntax
            # (i.e. form1.form2, 'form, `form, ~form, ~@form) as they need special
            # handling.
            if symbol_name not in [
                ".",
                "quote",
                "quasiquote",
                "unquote",
                "unquote-splice",
            ] and (scoped_identifier := in_scope_identifiers.get(symbol_name)):
                this_identifier = scoped_identifier

        # If this model is a plain dotted symbol, then we want to diverge from the Hy
        # model AST a bit and create our sub-forms a bit more carefully
        if dotted_components := dotted_name_components(hy_model):
            new_children = []
            for i, dot_component in enumerate(dotted_components):
                dot_context = ".".join(dotted_components[:i])

                child = TaggedFormTree(
                    children=[],
                    start_line=hy_model.start_line,
                    end_line=hy_model.start_line,
                    start_col=(
                        hy_model.start_column + len(dot_context) + (1 if i != 0 else 0)
                    ),
                    end_col=(
                        hy_model.start_column
                        + len(dot_context)
                        + len(dot_component)
                        - (0 if i != 0 else 1)
                    ),
                    scoped_identifiers=[],
                    this_identifier=in_scope_identifiers.get(
                        ".".join(dotted_components[: i + 1])
                    ),
                )
                new_children.append(child)

            return TaggedFormTree(
                children=new_children,
                start_line=hy_model.start_line,
                start_col=hy_model.start_column,
                end_line=hy_model.end_line,
                end_col=hy_model.end_column,
                scoped_identifiers=[],
                this_identifier=None,
            )

        # Now iterate through the child models to recursively create tagged forms.
        child_tagged_forms = []
        if isinstance(hy_model, hy.models.Sequence):
            for child_model in hy_model:
                child_tagged_forms.append(
                    cls.construct_from_hy_model(
                        child_model,
                        in_scope_identifiers=(
                            in_scope_identifiers
                            | {i.name: i for i in this_level_scoped_identifiers}
                        ),
                    )
                )

        return TaggedFormTree(
            children=child_tagged_forms,
            start_line=hy_model.start_line,
            start_col=hy_model.start_column,
            end_line=hy_model.end_line,
            end_col=hy_model.end_column,
            scoped_identifiers=this_level_scoped_identifiers,
            this_identifier=this_identifier,
        )

    def forms_with_identifiers(self) -> Generator[TaggedFormTree, None, None]:
        if self.this_identifier:
            yield self

        for child in self.children:
            for form in child.forms_with_identifiers():
                yield form
