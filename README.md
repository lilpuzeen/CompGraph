## Graph Computing
`computation graphs` `map` `reduce` `join` `sort` `MapReduce`

### Overview of the Project
This library facilitates graph-based computations.

Graph-based table computations involve _computational graphs_. These graphs are essentially a series of predefined operations that can be applied to various datasets.

A Table is a list of dictionaries, where each dictionary represents a table row, and the keys of the dictionary correspond to the table's columns.

For ease of understanding, we assume all input table rows have identical key sets.

#### The Importance of Computational Graphs
The core advantage of computational graphs lies in their ability to decouple the operation sequence definition from its execution. This separation enables operations to be executed in different environments (e.g., defining a graph within a Python interpreter and executing it on a GPU) and allows for simultaneous, parallel processing across multiple cluster machines to handle large data sets efficiently within a reasonable timeframe.

### Interface
A computational graph is structured around data entry points and subsequent operations on this data.
#### Data Entry Points
* graph_from_iter
```python
graph = Graph.graph_from_iter('input')
```

* graph_from_file
```python
iter_of_rows = Graph.graph_from_file(filename, parser)
```

* graph_copy
```python
another_graph = Graph.graph_from_graph(graph)
```

#### Defined operations

* Map - Transforms a single row into another row.
* Reduce - Processes groups of rows based on keys to produce a new set of rows.
* Join - Merges two graphs into one.
* Sort - Organizes rows according to their keys.

#### Execution
To execute the graph, use:
```python
graph.run(input=lambda: iter([{'key': 'value'}]))
```

or

```python
graph.run(input=lambda: Graph.graph_from_file(filename, parser)))
```

### Installation
In order to install this package, run:
```commandline
pip install compgraph
```

### Sample usage
First, construct a graph using a series of operations, then execute it with your data:

```python
graph = Graph().operation1(...)
               .operation2(...)
               .operation2(...)
result = graph.run()
```