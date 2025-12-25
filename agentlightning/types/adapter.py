# Copyright (c) Microsoft. All rights reserved.

"""Data formats used by adapters, usually the target format converted from trace spans."""

from __future__ import annotations

from typing import Callable, Dict, Generic, Iterable, Iterator, MutableSequence, TypeVar

T = TypeVar("T")


class Tree(Generic[T]):

    def __init__(self, item: T, children: MutableSequence[Tree[T]]) -> None:
        self.item = item
        self.children = children

    def traverse(self) -> Iterable[T]:
        yield self.item
        for child in self.children:
            yield from child.traverse()

    def count(self) -> int:
        return 1 + sum(child.count() for child in self.children)

    def __iter__(self) -> Iterator[T]:
        return iter(self.traverse())

    def __len__(self) -> int:
        return self.count()

    def add(self, child: Tree[T]) -> None:
        self.children.append(child)

    def prune(self, predicate: Callable[[T], bool]) -> Tree[T]:
        return Tree(self.item, [child.prune(predicate) for child in self.children if predicate(child.item)])

    def visualize(self, filename: str, item_to_str: Callable[[T], str]) -> None:
        """Render the tree with Graphviz for debugging purposes.

        Args:
            filename: Base filename for the generated `.png` diagram (without extension).

        !!! note

            The method requires the optional `graphviz` dependency to be available in the runtime
            environment.
        """
        import graphviz

        dot = graphviz.Digraph(comment="Tree")

        def visit(node: Tree[T]):
            dot.node(str(id(node)), item_to_str(node.item))  # type: ignore
            for child in node.children:
                visit(child)
                dot.edge(str(id(node)), str(id(child)))  # type: ignore

        visit(self)
        dot.render(filename, format="png", cleanup=True)  # type: ignore


class ChatCompletionCall(TypedDict):
    pass
