import click
import json

from compgraph.algorithms import word_count_graph
from compgraph import Graph


@click.command()
@click.argument('output_filepath', type=click.Path(), default='result_word_count.txt')
@click.argument('input_filepath', type=click.Path(exists=True), default='../resources/text_corpus.txt')
def main(input_filepath: str, output_filepath: str) -> None:
    graph = word_count_graph(input_stream_name='input')
    result = graph.run(input=lambda: Graph.graph_from_file(input_filepath, lambda x: json.loads(x)))
    with open(output_filepath, 'w') as out:
        for row in result:
            print(json.dumps(row), file=out)


if __name__ == "__main__":
    main()
