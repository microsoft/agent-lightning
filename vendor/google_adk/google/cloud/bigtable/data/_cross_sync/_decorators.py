# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Contains a set of AstDecorator classes, which define the behavior of CrossSync decorators.
Each AstDecorator class is used through @CrossSync.<decorator_name>
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    import ast
    from typing import Callable, Any


class AstDecorator:
    """
    Helper class for CrossSync decorators used for guiding ast transformations.

    AstDecorators are accessed in two ways:
    1. The decorations are used directly as method decorations in the async client,
        wrapping existing classes and methods
    2. The decorations are read back when processing the AST transformations when
        generating sync code.

    This class allows the same decorator to be used in both contexts.

    Typically, AstDecorators act as a no-op in async code, and the arguments simply
    provide configuration guidance for the sync code generation.
    """

    @classmethod
    def decorator(cls, *args, **kwargs) -> Callable[..., Any]:
        """
        Provides a callable that can be used as a decorator function in async code

        AstDecorator.decorate is called by CrossSync when attaching decorators to
        the CrossSync class.

        This method creates a new instance of the class, using the arguments provided
        to the decorator, and defers to the async_decorator method of the instance
        to build the wrapper function.

        Arguments:
            *args: arguments to the decorator
            **kwargs: keyword arguments to the decorator
        """
        # decorators with no arguments will provide the function to be wrapped
        # as the first argument. Pull it out if it exists
        func = None
        if len(args) == 1 and callable(args[0]):
            func = args[0]
            args = args[1:]
        # create new AstDecorator instance from given decorator arguments
        new_instance = cls(*args, **kwargs)
        # build wrapper
        wrapper = new_instance.async_decorator()
        if wrapper is None:
            # if no wrapper, return no-op decorator
            return func or (lambda f: f)
        elif func:
            # if we can, return single wrapped function
            return wrapper(func)
        else:
            # otherwise, return decorator function
            return wrapper

    def async_decorator(self) -> Callable[..., Any] | None:
        """
        Decorator to apply the async_impl decorator to the wrapped function

        Default implementation is a no-op
        """
        return None

    def sync_ast_transform(
        self, wrapped_node: ast.AST, transformers_globals: dict[str, Any]
    ) -> ast.AST | None:
        """
        When this decorator is encountered in the ast during sync generation, this method is called
        to transform the wrapped node.

        If None is returned, the node will be dropped from the output file.

        Args:
            wrapped_node: ast node representing the wrapped function or class that is being wrapped
            transformers_globals: the set of globals() from the transformers module. This is used to access
                ast transformer classes that live outside the main codebase
        Returns:
            transformed ast node, or None if the node should be dropped
        """
        return wrapped_node

    @classmethod
    def get_for_node(cls, node: ast.Call | ast.Attribute | ast.Name) -> "AstDecorator":
        """
        Build an AstDecorator instance from an ast decorator node

        The right subclass is found by comparing the string representation of the
        decorator name to the class name. (Both names are converted to lowercase and
        underscores are removed for comparison). If a matching subclass is found,
        a new instance is created with the provided arguments.

        Args:
            node: ast.Call node representing the decorator
        Returns:
            AstDecorator instance corresponding to the decorator
        Raises:
            ValueError: if the decorator cannot be parsed
        """
        import ast

        # expect decorators in format @CrossSync.<decorator_name>
        # (i.e. should be an ast.Call or an ast.Attribute)
        root_attr = node.func if isinstance(node, ast.Call) else node
        if not isinstance(root_attr, ast.Attribute):
            raise ValueError("Unexpected decorator format")
        # extract the module and decorator names
        if "CrossSync" in ast.dump(root_attr):
            decorator_name = root_attr.attr
            got_kwargs: dict[str, Any] = (
                {str(kw.arg): cls._convert_ast_to_py(kw.value) for kw in node.keywords}
                if hasattr(node, "keywords")
                else {}
            )
            got_args = (
                [cls._convert_ast_to_py(arg) for arg in node.args]
                if hasattr(node, "args")
                else []
            )
            # convert to standardized representation
            formatted_name = decorator_name.replace("_", "").lower()
            for subclass in cls.get_subclasses():
                if subclass.__name__.lower() == formatted_name:
                    return subclass(*got_args, **got_kwargs)
            raise ValueError(f"Unknown decorator encountered: {decorator_name}")
        else:
            raise ValueError("Not a CrossSync decorator")

    @classmethod
    def get_subclasses(cls) -> Iterable[type["AstDecorator"]]:
        """
        Get all subclasses of AstDecorator

        Returns:
            list of all subclasses of AstDecorator
        """
        for subclass in cls.__subclasses__():
            yield from subclass.get_subclasses()
            yield subclass

    @classmethod
    def _convert_ast_to_py(cls, ast_node: ast.expr | None) -> Any:
        """
        Helper to convert ast primitives to python primitives. Used when unwrapping arguments
        """
        import ast

        if ast_node is None:
            return None
        if isinstance(ast_node, ast.Constant):
            return ast_node.value
        if isinstance(ast_node, ast.List):
            return [cls._convert_ast_to_py(node) for node in ast_node.elts]
        if isinstance(ast_node, ast.Tuple):
            return tuple(cls._convert_ast_to_py(node) for node in ast_node.elts)
        if isinstance(ast_node, ast.Dict):
            return {
                cls._convert_ast_to_py(k): cls._convert_ast_to_py(v)
                for k, v in zip(ast_node.keys, ast_node.values)
            }
        # unsupported node type
        return ast_node


class ConvertClass(AstDecorator):
    """
    Class decorator for guiding generation of sync classes

    Args:
        sync_name: use a new name for the sync class
        replace_symbols: a dict of symbols and replacements to use when generating sync class
        docstring_format_vars: a dict of variables to replace in the docstring
        rm_aio: if True, automatically strip all asyncio keywords from method. If false,
            only keywords wrapped in CrossSync.rm_aio() calls to be removed.
        add_mapping_for_name: when given, will add a new attribute to CrossSync,
            so the original class and its sync version can be accessed from CrossSync.<name>
    """

    def __init__(
        self,
        sync_name: str | None = None,
        *,
        replace_symbols: dict[str, str] | None = None,
        docstring_format_vars: dict[str, tuple[str | None, str | None]] | None = None,
        rm_aio: bool = False,
        add_mapping_for_name: str | None = None,
    ):
        self.sync_name = sync_name
        self.replace_symbols = replace_symbols
        docstring_format_vars = docstring_format_vars or {}
        self.async_docstring_format_vars = {
            k: v[0] or "" for k, v in docstring_format_vars.items()
        }
        self.sync_docstring_format_vars = {
            k: v[1] or "" for k, v in docstring_format_vars.items()
        }
        self.rm_aio = rm_aio
        self.add_mapping_for_name = add_mapping_for_name

    def async_decorator(self):
        """
        Use async decorator as a hook to update CrossSync mappings
        """
        from .cross_sync import CrossSync

        if not self.add_mapping_for_name and not self.async_docstring_format_vars:
            # return None if no changes needed
            return None

        new_mapping = self.add_mapping_for_name

        def decorator(cls):
            if new_mapping:
                CrossSync.add_mapping(new_mapping, cls)
            if self.async_docstring_format_vars:
                cls.__doc__ = cls.__doc__.format(**self.async_docstring_format_vars)
            return cls

        return decorator

    def sync_ast_transform(self, wrapped_node, transformers_globals):
        """
        Transform async class into sync copy
        """
        import ast
        import copy

        # copy wrapped node
        wrapped_node = copy.deepcopy(wrapped_node)
        # update name
        if self.sync_name:
            wrapped_node.name = self.sync_name
        # strip CrossSync decorators
        if hasattr(wrapped_node, "decorator_list"):
            wrapped_node.decorator_list = [
                d for d in wrapped_node.decorator_list if "CrossSync" not in ast.dump(d)
            ]
        else:
            wrapped_node.decorator_list = []
        # strip async keywords if specified
        if self.rm_aio:
            wrapped_node = transformers_globals["AsyncToSync"]().visit(wrapped_node)
        # add mapping decorator if needed
        if self.add_mapping_for_name:
            wrapped_node.decorator_list.append(
                ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="CrossSync", ctx=ast.Load()),
                        attr="add_mapping_decorator",
                        ctx=ast.Load(),
                    ),
                    args=[
                        ast.Constant(value=self.add_mapping_for_name),
                    ],
                    keywords=[],
                )
            )
        # replace symbols if specified
        if self.replace_symbols:
            wrapped_node = transformers_globals["SymbolReplacer"](
                self.replace_symbols
            ).visit(wrapped_node)
        # update docstring if specified
        if self.sync_docstring_format_vars:
            docstring = ast.get_docstring(wrapped_node)
            if docstring:
                wrapped_node.body[0].value = ast.Constant(
                    value=docstring.format(**self.sync_docstring_format_vars)
                )
        return wrapped_node


class Convert(ConvertClass):
    """
    Method decorator to mark async methods to be converted to sync methods

    Args:
        sync_name: use a new name for the sync method
        replace_symbols: a dict of symbols and replacements to use when generating sync method
        docstring_format_vars: a dict of variables to replace in the docstring
        rm_aio: if True, automatically strip all asyncio keywords from method. If False,
            only the signature `async def` is stripped. Other keywords must be wrapped in
            CrossSync.rm_aio() calls to be removed.
    """

    def __init__(
        self,
        sync_name: str | None = None,
        *,
        replace_symbols: dict[str, str] | None = None,
        docstring_format_vars: dict[str, tuple[str | None, str | None]] | None = None,
        rm_aio: bool = True,
    ):
        super().__init__(
            sync_name=sync_name,
            replace_symbols=replace_symbols,
            docstring_format_vars=docstring_format_vars,
            rm_aio=rm_aio,
            add_mapping_for_name=None,
        )

    def sync_ast_transform(self, wrapped_node, transformers_globals):
        """
        Transform async method into sync
        """
        import ast

        # replace async function with sync function
        converted = ast.copy_location(
            ast.FunctionDef(
                wrapped_node.name,
                wrapped_node.args,
                wrapped_node.body,
                wrapped_node.decorator_list
                if hasattr(wrapped_node, "decorator_list")
                else [],
                wrapped_node.returns if hasattr(wrapped_node, "returns") else None,
            ),
            wrapped_node,
        )
        # transform based on arguments
        return super().sync_ast_transform(converted, transformers_globals)


class Drop(AstDecorator):
    """
    Method decorator to drop methods or classes from the sync output
    """

    def sync_ast_transform(self, wrapped_node, transformers_globals):
        """
        Drop from sync output
        """
        return None


class Pytest(AstDecorator):
    """
    Used in place of pytest.mark.asyncio to mark tests

    When generating sync version, also runs rm_aio to remove async keywords from
    entire test function

    Args:
        rm_aio: if True, automatically strip all asyncio keywords from test code.
            Defaults to True, to simplify test code generation.
    """

    def __init__(self, rm_aio=True):
        self.rm_aio = rm_aio

    def async_decorator(self):
        import pytest

        return pytest.mark.asyncio

    def sync_ast_transform(self, wrapped_node, transformers_globals):
        """
        convert async to sync
        """
        import ast

        # always convert method to sync
        converted = ast.copy_location(
            ast.FunctionDef(
                wrapped_node.name,
                wrapped_node.args,
                wrapped_node.body,
                wrapped_node.decorator_list
                if hasattr(wrapped_node, "decorator_list")
                else [],
                wrapped_node.returns if hasattr(wrapped_node, "returns") else None,
            ),
            wrapped_node,
        )
        # convert entire body to sync if rm_aio is set
        if self.rm_aio:
            converted = transformers_globals["AsyncToSync"]().visit(converted)
        return converted


class PytestFixture(AstDecorator):
    """
    Used in place of pytest.fixture or pytest.mark.asyncio to mark fixtures

    Args:
        *args: all arguments to pass to pytest.fixture
        **kwargs: all keyword arguments to pass to pytest.fixture
    """

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def async_decorator(self):
        import pytest_asyncio  # type: ignore

        return lambda f: pytest_asyncio.fixture(*self._args, **self._kwargs)(f)

    def sync_ast_transform(self, wrapped_node, transformers_globals):
        import ast
        import copy

        arg_nodes = [
            a if isinstance(a, ast.expr) else ast.Constant(value=a) for a in self._args
        ]
        kwarg_nodes = []
        for k, v in self._kwargs.items():
            if not isinstance(v, ast.expr):
                v = ast.Constant(value=v)
            kwarg_nodes.append(ast.keyword(arg=k, value=v))

        new_node = copy.deepcopy(wrapped_node)
        if not hasattr(new_node, "decorator_list"):
            new_node.decorator_list = []
        new_node.decorator_list.append(
            ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="pytest", ctx=ast.Load()),
                    attr="fixture",
                    ctx=ast.Load(),
                ),
                args=arg_nodes,
                keywords=kwarg_nodes,
            )
        )
        return new_node
