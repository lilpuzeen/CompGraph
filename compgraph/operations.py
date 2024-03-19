import copy
import functools
import heapq
from collections import defaultdict
import string
from abc import abstractmethod, ABC
import typing as tp
import itertools
import math

TRow = dict[str, tp.Any]
TRowsIterable = tp.Iterable[TRow]
TRowsGenerator = tp.Generator[TRow, None, None]


class Operation(ABC):
    @abstractmethod
    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        pass


class Read(Operation):
    def __init__(self, filename: str, parser: tp.Callable[[str], TRow]) -> None:
        self.filename = filename
        self.parser = parser

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        with open(self.filename) as f:
            for line in f:
                yield self.parser(line)


class ReadIterFactory(Operation):
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for row in kwargs[self.name]():
            yield row


class ReadFromFile(Operation):
    def __init__(self, parser: tp.Callable[[TRow], TRow]) -> None:
        self.parser = parser

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        yield from (self.parser(row) for row in rows)


# Operations


class Mapper(ABC):
    """Base class for mappers"""
    @abstractmethod
    def __call__(self, row: TRow) -> TRowsGenerator:
        """
        :param row: one table row
        """
        pass


class Map(Operation):
    def __init__(self, mapper: Mapper) -> None:
        self.mapper = mapper

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for row in rows:
            yield from self.mapper(row)


class Reducer(ABC):
    """Base class for reducers"""
    @abstractmethod
    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        """
        :param rows: table rows
        """
        pass


class Reduce(Operation):
    def __init__(self, reducer: Reducer, keys: tp.Sequence[str]) -> None:
        self.reducer = reducer
        self.keys = keys

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for key, group in itertools.groupby(rows, lambda x: [x.get(k) for k in self.keys]):
            yield from self.reducer(self.keys, group)


class Joiner(ABC):
    """Base class for joiners"""
    def __init__(self, suffix_a: str = '_1', suffix_b: str = '_2') -> None:
        self._a_suffix = suffix_a
        self._b_suffix = suffix_b

    @abstractmethod
    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        """
        :param keys: join keys
        :param rows_a: left table rows
        :param rows_b: right table rows
        """
        pass


class Join(Operation):
    def __init__(self, joiner: Joiner, keys: tp.Sequence[str]):
        self.keys = keys
        self.joiner = joiner
        self.join_iter = iter([])

    def __call__(self, left_table: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        """
        Join two tables based on specified keys.

        @param left_table: The first table to join.
        @param args: Tuple where the second table to join is expected.
        @return: Generator yielding the result of the join.
        """
        if len(args):
            self.join_iter = args[0]
        a_groupby_gen = itertools.groupby(left_table, lambda x: [x.get(k) for k in self.keys])
        key_a, group_a = next(a_groupby_gen, (None, iter([])))
        b_groupby_gen = itertools.groupby(self.join_iter, lambda x: [x.get(k) for k in self.keys])
        key_b, group_b = next(b_groupby_gen, (None, iter([])))
        while key_a is not None or key_b is not None:
            if key_a == key_b:
                yield from self.joiner(self.keys, group_a, group_b)
                key_a, group_a = next(a_groupby_gen, (None, iter([])))
                key_b, group_b = next(b_groupby_gen, (None, iter([])))
            elif key_b is None or (key_a is not None and key_a < key_b):
                yield from self.joiner(self.keys, group_a, iter([]))
                key_a, group_a = next(a_groupby_gen, (None, iter([])))
            elif key_a is None or (key_b is not None and key_b < key_a):
                yield from self.joiner(self.keys, iter([]), group_b)
                key_b, group_b = next(b_groupby_gen, (None, iter([])))


# Dummy operators


class DummyMapper(Mapper):
    """Yield exactly the row passed"""
    def __call__(self, row: TRow) -> TRowsGenerator:
        yield row


class FirstReducer(Reducer):
    """Yield only first row from passed ones"""
    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        for row in rows:
            yield row
            break


# Mappers


class FilterPunctuation(Mapper):
    """Left only non-punctuation symbols"""
    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self.column = column

    def __call__(self, row: TRow) -> TRowsGenerator:
        yield {**row, self.column: ''.join(filter(lambda x: not x in string.punctuation, row[self.column]))}


class LowerCase(Mapper):
    """Replace column value with value in lower case"""
    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self.column = column

    @staticmethod
    def _lower_case(txt: str):
        return txt.lower()

    def __call__(self, row: TRow) -> TRowsGenerator:
        yield {**row, self.column: self._lower_case(row[self.column])}


class IDF(Mapper):
    """
    Calculate IDF

    :param row_count: column containing number of words
    :param docs_per_word: column containing number of docs with 'word' occurrences
    :param idf: resulting column
    """
    def __init__(self, row_count: str, docs_per_word: str, idf: str = "idf") -> None:
        self.row_count = row_count
        self.docs_per_word = docs_per_word
        self.idf = idf

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.idf] = math.log(row[self.row_count] / row[self.docs_per_word])
        yield row


class TFIDF(Mapper):
    def __init__(self, tf: str, idf: str, tfidf: str = "tfidf") -> None:
        """
        Calculate TFIDF

        :param tf: column containing tf
        :param idf: column containing idf
        :param tfidf: resulting column
        """
        self.tf = tf
        self.idf = idf
        self.tfidf = tfidf

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.tfidf] = row[self.tf] * row[self.idf]
        yield row


class PMI(Mapper):
    def __init__(self, tf: str, cf: str, pmi: str = "pmi") -> None:
        """
        Calculate PMI

        :param tf: frequency of i-th word in i-th doc
        :param cf: frequency of i-th word in all documents
        :param pmi: resulting column
        """
        self.tf = tf
        self.cf = cf
        self.pmi = pmi

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.pmi] = math.log(row[self.tf] / row[self.cf])
        yield row


class Split(Mapper):
    """Split row on multiple rows by separator"""
    def __init__(self, column: str, separator: str | None = None) -> None:
        """
        :param column: name of column to split
        :param separator: string to separate by
        """
        self.column = column
        self.separator = separator

    def __call__(self, row: TRow) -> TRowsGenerator:
        for value in row[self.column].split(self.separator):
            yield {**row, self.column: value}


class Product(Mapper):
    """Calculates product of multiple columns"""
    def __init__(self, columns: tp.Sequence[str], result_column: str = 'product') -> None:
        """
        :param columns: column names to product
        :param result_column: column name to save product in
        """
        self.columns = columns
        self.result_column = result_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        yield {**row, self.result_column: functools.reduce(lambda x, y: x * y, (row[column] for column in self.columns))}


class Filter(Mapper):
    """Remove records that don't satisfy some condition"""
    def __init__(self, condition: tp.Callable[[TRow], bool]) -> None:
        """
        :param condition: if condition is not true - remove record
        """
        self.condition = condition

    def __call__(self, row: TRow) -> TRowsGenerator:
        if self.condition(row):
            yield row


class Project(Mapper):
    """Leave only mentioned columns"""
    def __init__(self, columns: tp.Sequence[str]) -> None:
        """
        :param columns: names of columns
        """
        self.columns = columns

    def __call__(self, row: TRow) -> TRowsGenerator:
        yield {column: row[column] for column in self.columns}


class Apply(Mapper):
    """Apply function for row"""

    def __init__(self, func: tp.Callable[[TRow], tp.Any], result_column: str | None = None) -> None:
        """
        :param func: function to apply
        :param result_column: name for result column
        """
        self.func = func
        self.result_column = result_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        if self.result_column is None:
            self.func(row)
        else:
            row[self.result_column] = self.func(row)
        yield row


# Reducers


class TopN(Reducer):
    """Calculate top N by value"""
    def __init__(self, column: str, n: int) -> None:
        """
        :param column: column name to get top by
        :param n: number of top values to extract
        """
        self.column_max = column
        self.n = n

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        min_heap = []
        for index, row in enumerate(rows):
            key = row.get(self.column_max, 0)
            if len(min_heap) < self.n:
                heapq.heappush(min_heap, (key, index, row))
            else:
                heapq.heappushpop(min_heap, (key, index, row))

        top_n_rows = sorted((item for item in min_heap), key=lambda x: x[0], reverse=True)
        for _, _, row in top_n_rows:
            yield row


class TermFrequency(Reducer):
    """Calculate frequency of values in column"""
    def __init__(self, words_column: str, result_column: str = 'tf', exact_field: tp.Union[str, None] = None) -> None:
        """
        :param words_column: name for column with words
        :param result_column: name for result column
        """
        self.words_column = words_column
        self.result_column = result_column
        self.exact_field = exact_field

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        dictionary: tp.DefaultDict[str, int] = defaultdict(int)
        resulting_dict: tp.Dict[str, tp.Any] = dict()
        temp_sum = 0
        if self.exact_field is None:
            for row in rows:
                dictionary[row[self.words_column]] += 1
                temp_sum += 1
        else:
            for row in rows:
                dictionary[row[self.words_column]] += row[self.exact_field]
                temp_sum += row[self.exact_field]

        resulting_dict.update({key: row[key] for key in group_key})

        for key, value in dictionary.items():
            res_row = {self.words_column: key, self.result_column: value / temp_sum}
            res_row.update(resulting_dict)
            yield res_row


class Count(Reducer):
    """
    Count records by key
    Example for group_key=('a',) and column='count_a'
        {'a': 1, 'b': 5, 'c': 2}
        {'a': 1, 'b': 6, 'c': 1}
        =>
        {'a': 1, 'count_a': 2}
    """
    def __init__(self, column: str) -> None:
        """
        :param column: name for result column
        """
        self.column = column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        count = 0
        res_row: TRow = {}
        for row in rows:
            count += 1
            res_row = res_row or {key: row[key] for key in group_key}
        res_row[self.column] = count
        yield res_row


class Sum(Reducer):
    """
    Sum values aggregated by key
    Example for key=('a',) and column='b'
        {'a': 1, 'b': 2, 'c': 4}
        {'a': 1, 'b': 3, 'c': 5}
        =>
        {'a': 1, 'b': 5}
    """
    def __init__(self, column: str) -> None:
        """
        :param column: name for sum column
        """
        self.column = column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        sum_dict = {}
        for row in rows:
            group_value = row.get(group_key[0])
            sum_dict[group_value] = sum_dict.get(group_value, 0) + row.get(self.column, 0)

        for group, total in sum_dict.items():
            res_row = {self.column: total}
            res_row.update({key: row[key] for key in group_key})
            yield res_row


# Joiners


class InnerJoiner(Joiner):
    """Join with inner strategy"""

    def __init__(self, suffix_a: str = '_1', suffix_b: str = '_2') -> None:
        super(InnerJoiner, self).__init__(suffix_a, suffix_b)

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        rows_b = list(rows_b)
        for row_a in rows_a:
            for row_b in rows_b:
                new_row = copy.deepcopy(row_b)
                for key in row_a.keys():
                    if key in row_b.keys() and key not in keys:
                        new_row[key + self._a_suffix] = row_a[key]
                        new_row[key + self._b_suffix] = row_b[key]
                        new_row.pop(key)
                    else:
                        new_row[key] = row_a[key]
                yield new_row


class OuterJoiner(Joiner):
    """Join with outer strategy"""

    def __init__(self, suffix_a: str = '_1', suffix_b: str = '_2') -> None:
        super(OuterJoiner, self).__init__(suffix_a, suffix_b)

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        try:
            row_b: TRow = next(rows_b)
        except StopIteration:
            for row_a in rows_a:
                yield copy.deepcopy(row_a)
        else:
            rows_b_lst: list[TRow] = list(rows_b)
            rows_b_lst.insert(0, row_b)
            yield from RightJoiner(self._a_suffix, self._b_suffix)(keys, rows_a, rows_b_lst)


class LeftJoiner(Joiner):
    """Join with left strategy"""

    def __init__(self, suffix_a: str = '_1', suffix_b: str = '_2') -> None:
        super(LeftJoiner, self).__init__(suffix_a, suffix_b)

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        try:
            row_b: TRow = next(rows_b)
        except StopIteration:
            for row_a in rows_a:
                yield copy.deepcopy(row_a)
        else:
            rows_b_lst: list[TRow] = list(rows_b)
            rows_b_lst.insert(0, row_b)
            yield from InnerJoiner(self._a_suffix, self._b_suffix)(keys, rows_a, rows_b_lst)


class RightJoiner(Joiner):
    """Join with right strategy"""

    def __init__(self, suffix_a: str = '_1', suffix_b: str = '_2') -> None:
        super(RightJoiner, self).__init__(suffix_a, suffix_b)

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        try:
            row_a: TRow = next(rows_a)
        except StopIteration:
            for row_b in rows_b:
                yield copy.deepcopy(row_b)
        else:
            rows_a_lst: list[TRow] = list(rows_a)
            rows_a_lst.insert(0, row_a)
            yield from InnerJoiner(self._a_suffix, self._b_suffix)(keys, rows_b, rows_a_lst)
