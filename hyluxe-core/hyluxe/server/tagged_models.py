from __future__ import annotations

import builtins
import inspect
import itertools
import types
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import hy  # for side-effects
import hy.models
import toolz.functoolz
from funcparserlib.parser import NoParseError, many, maybe
from hy.core.result_macros import importlike, module_name_pattern
from hy.model_patterns import pexpr, sym
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


@dataclass
class TaggedModel:
    """Wraps a `hy.models.Object` to attach data useful for the language server.

    Creates a tree structure with recursive `TaggedModel.children` lists. This tree
    structure mirrors that created by `hy.models.Sequence`.
    """

    # special signal type and value
    RootModelSignalType = types.EllipsisType
    ROOT_MODEL = ...

    # class variable and decorator to store functions that tag models with values - each
    # function will get called on each model recursively down the tree, and get the
    # chance to return a modified version if it had relevant data to add
    tagger_functions = []

    @classmethod
    def tagger_function(cls, f):
        """Register a tagger function to add data to the tagged models."""
        cls.tagger_functions.append(f)
        return f

    # tree structure fields
    hy_model: hy.models.Object | RootModelSignalType
    parent: TaggedModel | RootModelSignalType
    children: list[TaggedModel] = field(default_factory=list)

    # tagged data fields
    scoped_identifiers: list[ScopedIdentifier] = field(default_factory=list)

    @classmethod
    def create_root_model(cls, hy_forms: Iterable[hy.models.Object]):
        """Recursively create a TaggedModel tree from the root based on a list of hy
        forms.
        """
        root_model = cls(
            hy_model=cls.ROOT_MODEL,
            parent=cls.ROOT_MODEL,
        )
        root_model = toolz.functoolz.compose(*cls.tagger_functions)(root_model)
        root_model.children = [cls.tag_hy_model(m, root_model) for m in hy_forms]
        return root_model

    @classmethod
    def tag_hy_model(
        cls, hy_model: hy.models.Object, parent: TaggedModel
    ) -> TaggedModel:
        """Create and tag a new TaggedModel from a hy model and a parent TaggedModel."""
        tagged_model = cls(hy_model=hy_model, parent=parent)
        tagged_model = toolz.functoolz.compose(*cls.tagger_functions)(tagged_model)
        if isinstance(tagged_model.hy_model, hy.models.Sequence):
            tagged_model.children = [
                cls.tag_hy_model(m, tagged_model) for m in tagged_model.hy_model
            ]

        return tagged_model

    def get_models_enclosing_position(self, line: int, col: int) -> list[TaggedModel]:
        """Recursively get the list of TaggedModels enclosing a given position.

        Models in the returned list are ordered from the top of the tree down.
        """
        if self.hy_model == TaggedModel.ROOT_MODEL or (
            self.hy_model.start_line <= line <= self.hy_model.end_line
            and self.hy_model.start_column <= col <= self.hy_model.end_column
        ):
            inner_lists = [
                c.get_models_enclosing_position(line, col) for c in self.children
            ]
            return list(itertools.chain.from_iterable([[self]] + inner_lists))
        else:
            return []

    def __repr__(self):
        return (
            f"TaggedModel({str(self.hy_model)}, "
            + f"{len(self.children)} children, "
            + f"{len(self.scoped_identifiers)} identifiers)"
        )


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


@TaggedModel.tagger_function
def tag_core_macros(model: TaggedModel) -> TaggedModel:
    if model.hy_model == TaggedModel.ROOT_MODEL:
        model.scoped_identifiers.extend(_core_macro_completions())
    return model


# IMPORTS
_import_parser = sym("import") + many(module_name_pattern + maybe(importlike))


@TaggedModel.tagger_function
def tag_imports(model: TaggedModel) -> TaggedModel:
    """If model is an import expression, add the imported modules/identifiers in the
    scoped variables of the parent model.
    """
    if not isinstance(model.hy_model, hy.models.Expression):
        return model

    try:
        entries = _import_parser.parse(model.hy_model)
        for mod, as_or_froms in entries:
            if as_or_froms is None:
                # "import xyz"
                model.parent.scoped_identifiers.append(
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
                            model.parent.scoped_identifiers.append(
                                ScopedIdentifier(
                                    name=ident[:],
                                    kind=lsp.CompletionItemKind.Variable,
                                )
                            )
                        else:
                            # TODO as
                            pass
        return model
    except NoParseError:
        return model
