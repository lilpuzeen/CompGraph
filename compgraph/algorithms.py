from . import Graph, operations
import math


def word_count_graph(input_stream_name: str, text_column: str = 'text', count_column: str = 'count') -> Graph:
	"""Constructs graph which counts words in text_column of all rows passed"""
	return Graph.graph_from_iter(input_stream_name) \
		.map(operations.FilterPunctuation(text_column)) \
		.map(operations.LowerCase(text_column)) \
		.map(operations.Split(text_column)) \
		.sort([text_column]) \
		.reduce(operations.Count(count_column), [text_column]) \
		.sort([count_column, text_column])


def inverted_index_graph(input_stream_name: str, doc_column: str = 'doc_id', text_column: str = 'text',
                         result_column: str = 'tf_idf', n: int = 3) -> Graph:
	"""Constructs graph which calculates tf-idf for every word/document pair"""

	def _calculate_tf_idf(row: operations.TRow) -> float:
		w1 = row['docs']
		w2 = row['count_doc']
		return math.log((w1 / w2))

	graph = (Graph.graph_from_iter(input_stream_name).
	         map(operations.FilterPunctuation(text_column)).
	         map(operations.LowerCase(text_column)).
	         map(operations.Split(text_column)).
	         sort([doc_column, text_column]))

	graph_document_count = (graph.reduce(operations.Count('docs'), [doc_column]).
	                        reduce(operations.Count('docs'), []))

	idf = (graph.reduce(operations.Count('count'), [doc_column, text_column]).
	       sort([text_column]).
	       reduce(operations.Count('count_doc'), [text_column]).
	       join(operations.InnerJoiner(), graph_document_count, []))

	return graph \
		.sort([doc_column]) \
		.reduce(operations.TermFrequency(text_column), [doc_column]) \
		.sort([text_column]) \
		.join(operations.InnerJoiner(), idf, [text_column]) \
		.map(operations.Apply(_calculate_tf_idf, result_column)) \
		.map(operations.Product(['tf', result_column], result_column)) \
		.reduce(operations.TopN(result_column, n), [text_column]) \
		.map(operations.Project([doc_column, text_column, result_column])) \
		.sort([doc_column, text_column])


def pmi_graph(input_stream_name: str, doc_column: str = 'doc_id', text_column: str = 'text',
              result_column: str = 'pmi', n: int = 10) -> Graph:
	"""Constructs graph which gives for every document the top 10 words ranked by pointwise mutual information"""

	def _calculate_pmi(row: operations.TRow) -> float:
		w1_w2 = row['tf_doc']
		w1 = row['tf_sum']
		return math.log((w1_w2 / w1))

	graph = Graph.graph_from_iter(input_stream_name) \
		.map(operations.FilterPunctuation(text_column)) \
		.map(operations.LowerCase(text_column)) \
		.map(operations.Split(text_column)) \
		.sort([doc_column, text_column])

	graph_filtered = graph \
		.reduce(operations.Count('count'), [doc_column, text_column]) \
		.map(operations.Filter(lambda x: x['count'] >= 2 and len(x[text_column]) > 4))

	graph = graph.join(operations.InnerJoiner(), graph_filtered, [doc_column, text_column])

	graph_sum = graph \
		.reduce(operations.TermFrequency(text_column, result_column='tf_sum'), []) \
		.sort([text_column])

	return graph \
		.reduce(operations.TermFrequency(text_column, result_column='tf_doc'), [doc_column]) \
		.sort([text_column]) \
		.join(operations.InnerJoiner(), graph_sum, [text_column]) \
		.map(operations.Apply(_calculate_pmi, result_column)) \
		.map(operations.Project([doc_column, text_column, result_column])) \
		.sort([doc_column, result_column]) \
		.reduce(operations.TopN(result_column, n), [doc_column])


def yandex_maps_graph(input_stream_name_time: str, input_stream_name_length: str,
                      enter_time_column: str = 'enter_time', leave_time_column: str = 'leave_time',
                      edge_id_column: str = 'edge_id', start_coord_column: str = 'start', end_coord_column: str = 'end',
                      weekday_result_column: str = 'weekday', hour_result_column: str = 'hour',
                      speed_result_column: str = 'speed') -> Graph:
	"""Constructs graph which measures average speed in km/h depending on the weekday and hour"""
	raise NotImplementedError


__all__ = ['word_count_graph', 'inverted_index_graph', 'pmi_graph', 'yandex_maps_graph']