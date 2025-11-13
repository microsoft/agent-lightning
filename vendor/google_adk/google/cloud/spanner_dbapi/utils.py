# Copyright 2020 Google LLC All rights reserved.
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

import re

re_UNICODE_POINTS = re.compile(r"([^\s]*[\u0080-\uFFFF]+[^\s]*)")


class PeekIterator:
    """
    Peek at the first element out of an iterator for the sake of operations
    like auto-population of fields on reading the first element.
    If next's result is an instance of list, it'll be converted into a tuple to
    conform with DBAPI v2's sequence expectations.

    :type source: list
    :param source: A list of source for the Iterator.
    """

    def __init__(self, source):
        itr_src = iter(source)

        self.__iters = []
        self.__index = 0

        try:
            head = next(itr_src)
            # Restitch and prepare to read from multiple iterators.
            self.__iters = [iter(itr) for itr in [[head], itr_src]]
        except StopIteration:
            pass

    def __next__(self):
        if self.__index >= len(self.__iters):
            raise StopIteration

        iterator = self.__iters[self.__index]
        try:
            head = next(iterator)
        except StopIteration:
            # That iterator has been exhausted, try with the next one.
            self.__index += 1
            return self.__next__()
        else:
            return tuple(head) if isinstance(head, list) else head

    def __iter__(self):
        return self


class StreamedManyResultSets:
    """Iterator to walk through several `StreamedResultsSet` iterators.
    This type of iterator is used by `Cursor.executemany()`
    method to iterate through several `StreamedResultsSet`
    iterators like they all are merged into single iterator.
    """

    def __init__(self):
        self._iterators = []
        self._index = 0

    def add_iter(self, iterator):
        """Add new iterator into this one.
        :type iterator: :class:`google.cloud.spanner_v1.streamed.StreamedResultSet`
        :param iterator: Iterator to merge into this one.
        """
        self._iterators.append(iterator)

    def __next__(self):
        """Return the next value from the currently streamed iterator.
        If the current iterator is streamed to the end,
        start to stream the next one.
        :rtype: list
        :returns: The next result row.
        """
        try:
            res = next(self._iterators[self._index])
        except StopIteration:
            self._index += 1
            res = self.__next__()
        except IndexError:
            raise StopIteration

        return res

    def __iter__(self):
        return self


def backtick_unicode(sql):
    """Check the SQL to be valid and split it by segments.

    :type sql: str
    :param sql: A SQL request.

    :rtype: str
    :returns: A SQL parsed by segments in unicode if initial SQL is valid,
              initial string otherwise.
    """
    matches = list(re_UNICODE_POINTS.finditer(sql))
    if not matches:
        return sql

    segments = []

    last_end = 0
    for match in matches:
        start, end = match.span()
        if sql[start] != "`" and sql[end - 1] != "`":
            segments.append(sql[last_end:start] + "`" + sql[start:end] + "`")
        else:
            segments.append(sql[last_end:end])

        last_end = end

    return "".join(segments)


def sanitize_literals_for_upload(s):
    """Convert literals in s, to be fit for consumption by Cloud Spanner.

    * Convert %% (escaped percent literals) to %. Percent signs must be escaped
      when values like %s are used as SQL parameter placeholders but Spanner's
      query language uses placeholders like @a0 and doesn't expect percent
      signs to be escaped.
    * Quote words containing non-ASCII, with backticks, for example föö to
    `föö`.

    :type s: str
    :param s: A string with literals to escaped for consumption by Cloud
              Spanner.

    :rtype: str
    :returns: A sanitized string for uploading.
    """
    return backtick_unicode(s.replace("%%", "%"))
