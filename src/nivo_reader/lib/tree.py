from collections.abc import Generator, Callable, Iterator
from typing import TypeVar, cast

Node = TypeVar("Node")
Value = TypeVar("Value")
type Tree[N] = Generator[N | Tree[N], None, None]


def tree_scan1[N, V](
    tree: Tree[N],
    func: Callable[[V, N], V],
    acc: V,
) -> Iterator[tuple[N, V]]:
    it = iter(tree)
    node: N = next(it)  # first yield  → this node's value
    new_acc = func(acc, node)
    yield node, new_acc
    for subtree in it:  # remaining yields → child subtrees
        yield from tree_scan(subtree, func, new_acc)


def scan_tree_leaves[N, L, V](
    node: N, func: Callable[[V, N], V], acc: V, is_leaf: Callable[[N], bool]
) -> Iterator[tuple[N, V]]:
    if is_leaf(node):
        yield node, acc
    else:
        for subtree in iter(node):
            yield from scan_tree_leaves(subtree, func, func(acc, node), is_leaf)
