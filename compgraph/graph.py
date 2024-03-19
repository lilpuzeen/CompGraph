import json
import typing as tp

from abc import abstractmethod, ABC
from . import operations as ops
from . import external_sort as exts
from .operations import TRow, TRowsIterable, TRowsGenerator


TNode = tp.TypeVar("TNode", 'Node', 'NodeFromFile', 'NodeFromIter')


class AbstractNode(ABC):
    """Abstract node of computational graph"""
    @abstractmethod
    def __call__(self, sources: tp.Dict[str, tp.Callable[[TRow], TRowsIterable]]) -> TRowsGenerator:
        pass


class Node(AbstractNode):
    """
    Node of computational graph.

    :param operation: 'Operation' type fabric function
    :param parents: list of parent nodes
    """
    def __init__(self, operation: ops.Operation, parents: tp.List[TNode]) -> None:
        self.operation = operation
        self.parents = parents

    def __call__(self, sources: tp.Dict[str, tp.Callable[[TRow], TRowsIterable]]) -> TRowsGenerator:
        """Starts computation in node, calling parent nodes if necessary

        :param sources: data sources
        """
        yield from self.operation(*[parent(sources) for parent in self.parents])


class NodeFromFile(AbstractNode):
    """
    Input node of computational graph which reads data from file.

    :param filename: filename to read from
    :param parser: parser from string to 'TRow'
    """
    def __init__(self, filename: str, parser: tp.Callable[[str], ops.TRow] = json.loads):
        self.filename = filename
        self.operation = ops.ReadFromFile(parser)

    def __call__(self, sources: tp.Dict[str, tp.Callable[[TRow], TRowsIterable]]) -> TRowsGenerator:
        with open(self.filename, 'r') as file:
            for line in file:
                yield from self.operation(line)


class NodeFromIter(AbstractNode):
    """
    Input node of computational graph which reads data from iterator

    :param iterator_name: name of kwarg to use as data source
    """
    def __init__(self, iterator_name: str) -> None:
        self.iterator_name = iterator_name

    def __call__(self, sources: tp.Dict[str, tp.Callable[[TRow], TRowsIterable]]) -> TRowsGenerator:
        yield from sources[self.iterator_name]()


class Graph:
    """Computational graph implementation"""

    def __init__(self) -> None:
        self.operation: ops.Operation = ops.ReadIterFactory('')
        self.parents: list[Graph] | None = None
        self.join_parent: Graph | None = None
        self.input: ops.TRow | None = None

    @staticmethod
    def graph_from_iter(name: str) -> 'Graph':
        """Construct new graph which reads data from row iterator (in form of sequence of Rows
        from 'kwargs' passed to 'run' method) into graph data-flow
        Use ops.ReadIterFactory
        :param name: name of kwarg to use as data source
        """
        graph = Graph()
        graph.operation = ops.ReadIterFactory(name)
        return graph

    @staticmethod
    def graph_from_file(filename: str, parser: tp.Callable[[str], ops.TRow] = json.loads) -> ops.TRowsGenerator:
        """Construct new graph extended with operation for reading rows from file
        Use ops.Read
        :param filename: filename to read from
        :param parser: parser from string to Row
        """
        return ops.Read(filename, parser)()

    @staticmethod
    def graph_from_graph(graph: 'Graph') -> 'Graph':
        """Construct new graph extended with operation for reading rows from another graph

        :param graph: graph to read from
        """
        return Graph(tail=graph.tail)

    def map(self, mapper: ops.Mapper) -> 'Graph':
        """Construct new graph extended with map operation with particular mapper
        :param mapper: mapper to use
        """
        graph = Graph()
        graph.operation = ops.Map(mapper)
        graph.parents = [self]
        return graph

    def reduce(self, reducer: ops.Reducer, keys: tp.Sequence[str]) -> 'Graph':
        """Construct new graph extended with reduce operation with particular reducer
        :param reducer: reducer to use
        :param keys: keys for grouping
        """
        graph = Graph()
        graph.operation = ops.Reduce(reducer, tuple(keys))
        graph.parents = [self]
        return graph

    def sort(self, keys: tp.Sequence[str]) -> 'Graph':
        """Construct new graph extended with sort operation
        :param keys: sorting keys (typical is tuple of strings)
        """
        graph = Graph()
        graph.operation = exts.ExternalSort(keys)
        graph.parents = [self]
        return graph

    def join(self, joiner: ops.Joiner, join_graph: 'Graph', keys: tp.Sequence[str]) -> 'Graph':
        """Construct new graph extended with join operation with another graph
        :param joiner: join strategy to use
        :param join_graph: other graph to join with
        :param keys: keys for grouping
        """
        graph = Graph()
        graph.operation = ops.Join(joiner, keys)
        graph.parents = [self, join_graph]
        return graph

    def run(self, **kwargs: tp.Any) -> ops.TRowsGenerator:
        """Single method to start execution; data sources passed as kwargs"""
        self.input = kwargs
        if self.parents is not None:
            streams = [p.run(**self.input) for p in self.parents]
            return self.operation(*streams)
        else:
            return self.operation(**self.input)


__all__ = ['Graph']