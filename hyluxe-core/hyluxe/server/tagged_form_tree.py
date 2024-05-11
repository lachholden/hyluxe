"""Contains classes and functions to convert a Hy hy.models AST into a TaggedFormTree.
"""

from __future__ import annotations

import builtins
import importlib
import inspect
import io
import itertools
import pydoc
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
from hy.reader.mangling import mangle, unmangle


class ScopedIdentifierKind(Enum):
    """The possible  kinds of identifiers that exist in a particular scope."""

    Module = "module"
    Variable = "variable"
    Class = "class"
    Method = "method"
    Function = "function"
    HyMacro = "macro"
    HyReader = "reader"
    HyMacroCore = "core-macro"


@dataclass
class ScopedIdentifier:
    """An identifier that exists in a particular scope in the TaggedFormTree."""

    name: str  # should by hy-style, unmangled
    kind: ScopedIdentifierKind
    documentation: Optional[str] = None
    signature: Optional[inspect.Signature] = None
    parent_name: Optional[str] = None
    py_obj: Optional[Any] = None


def _dotted_name_components(model: hy.models.Object) -> Optional[list[str]]:
    """Converts a hy model to its dotted components, if it's a simple dotted expression.

    i.e. 'abc -> None
         '(. abc def) -> None
         '(. [abc def]) -> None
         'abc.def -> ["abc", "def"]

    Note that the output of the Hy reader represents the last two examples with a
    similar tree. We distinguish them by checking whether all parts have the same
    positions, which is the case for the 'abc.def form.
    """

    if isinstance(model, hy.models.Expression):
        if not len(model) > 1 or not model[0] == hy.models.Symbol("."):
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


def attr_name_to_identifier_try_getattr(
    object: Any, lookup_name: str
) -> ScopedIdentifier:
    """Create a ScopedIdentifier for an attr with a give (hy-style, unmangled) name.

    Will attempt to call getattr(object, mangle(lookup_name)) to set the py_obj for the
    returned identifier.
    """
    if attr := getattr(object, mangle(lookup_name), None):
        kind, signature = ScopedIdentifierKind.Variable, None
        if inspect.ismodule(attr):
            kind = ScopedIdentifierKind.Module
        elif inspect.isclass(attr):
            kind = ScopedIdentifierKind.Class
        elif inspect.ismethod(attr) or inspect.ismethoddescriptor(attr):
            kind = ScopedIdentifierKind.Method
        elif inspect.isfunction(attr) or inspect.isroutine(attr):
            kind = ScopedIdentifierKind.Function
        try:
            signature = inspect.signature(attr)
        except (ValueError, TypeError):
            pass

        return ScopedIdentifier(
            name=lookup_name,
            kind=kind,
            signature=signature,
            documentation=pydoc.getdoc(attr),
            py_obj=attr,
            parent_name=unmangle(getattr(object, "__name__", None)),
        )
    else:
        return ScopedIdentifier(
            name=lookup_name,
            kind=ScopedIdentifierKind.Variable,
            parent_name=unmangle(getattr(object, "__name__", None)),
        )


# CORE MACROS
def _core_macro_identifiers() -> list[ScopedIdentifier]:
    import builtins

    import hy

    """Get the scoped identifiers corresponding to Hy's core macros."""
    return [
        ScopedIdentifier(
            name=unmangle(func_name),
            kind=ScopedIdentifierKind.HyMacroCore,
            documentation=pydoc.getdoc(func),
            signature=inspect.signature(func),
            parent_name=unmangle(inspect.getmodule(func).__name__),
            py_obj=func,
        )
        for func_name, func in builtins._hy_macros.items()
    ]


def _python_builtin_identifiers() -> list[ScopedIdentifier]:
    """Get the scoped identifiers corresponding to Python's builtins"""
    import builtins

    return [
        attr_name_to_identifier_try_getattr(builtins, unmangle(n))
        for n in dir(builtins)
        if n not in ["True", "False", "None"]  # gets weird if they're included
    ]


# PARSING IMPORTS
_import_parser = sym("import") + many(module_name_pattern + maybe(importlike))


def _module_name_to_identifier_try_import(module_name: str) -> ScopedIdentifier:
    """Create a ScopedIdentifier for a module with a given name.

    Will attempt to import the module with the given name to attach the corresponding
    module object and docs to the ScopedIdentifier.
    """
    try:
        mod = importlib.import_module(mangle(module_name))  # TODO relative imports
        return ScopedIdentifier(
            name=unmangle(module_name),
            kind=ScopedIdentifierKind.Module,
            documentation=pydoc.getdoc(mod),
            py_obj=mod,
        )
    except ModuleNotFoundError:
        return ScopedIdentifier(
            name=unmangle(module_name),
            kind=ScopedIdentifierKind.Module,
        )


def _identifiers_from_plain_import(
    mod_expr: hy.models.Object,
) -> list[ScopedIdentifier]:
    """Get the identifiers introduced corresponding to the modules for an import ...

    i.e. import abc.def.ghi will create and return identifiers for abc, abc.def, and
    abc.def.ghi
    """

    if isinstance(mod_expr, hy.models.Symbol):
        return [_module_name_to_identifier_try_import(mod_expr[:])]
    else:
        dotted_module_parts = _dotted_name_components(mod_expr)
        if dotted_module_parts is None:
            return []

        module_identifiers = []
        for i in range(len(dotted_module_parts)):
            module_name = ".".join(dotted_module_parts[: i + 1])
            module_identifiers.append(
                _module_name_to_identifier_try_import(module_name)
            )

        return module_identifiers


def _identifiers_from_from_import(
    mod_expr: hy.models.Object, ident_expr: hy.models.Object
) -> list[ScopedIdentifier]:
    """Get the scoped identifiers introduced by a from ... import ..."""
    module_obj = _module_name_to_identifier_try_import(mod_expr).py_obj
    return [attr_name_to_identifier_try_getattr(module_obj, ident_expr[:])]  # TODO


def _match_import_expr(model: hy.models.Object) -> list[ScopedIdentifier]:
    """Attempts to parse the given model as an import expression.

    Returns a list of ScopedIdentifiers defined if model is an import statement, or an
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
    def __getitem__(self, key):
        if reader := HyReader.DEFAULT_TABLE.get(key):
            return reader
        else:
            return lambda __, _: None


@dataclass(frozen=True)
class TaggedFormTree:
    """Forms an analagous tree structure to `hy.models.Object`.

    Attaches data useful for the language server.
    """

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

        Models in the returned list are ordered from the bottom of the tree up.
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

        # If this is the root model, then all of the core macros and builtins come in
        # scope here.
        if is_root:
            this_level_scoped_identifiers.extend(_core_macro_identifiers())
            this_level_scoped_identifiers.extend(_python_builtin_identifiers())

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
        if dotted_components := _dotted_name_components(hy_model):
            new_children = []
            for i, dot_component in enumerate(dotted_components):
                dot_context = ".".join(dotted_components[:i])

                # Let's try to figure out which, if any, identifier this specific
                # section of the dot-chain this dot_component represents.
                # First, check if we've saved an in-scope identifier with the full
                # dotted name at all:
                this_identifier = in_scope_identifiers.get(
                    ".".join(dotted_components[: i + 1])
                )

                # If we've found nothing, let's try getting the attr on the identifier
                # found for the previous component
                if not this_identifier and i > 0 and new_children[-1].this_identifier:
                    this_identifier = attr_name_to_identifier_try_getattr(
                        new_children[-1].this_identifier.py_obj, dot_component
                    )

                # If this_identifier is still None, then too bad - we didn't find
                # anything

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
                    this_identifier=this_identifier,
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
        """Get all forms that have a corresponding identifier representation set."""
        if self.this_identifier:
            yield self

        for child in self.children:
            for form in child.forms_with_identifiers():
                yield form
