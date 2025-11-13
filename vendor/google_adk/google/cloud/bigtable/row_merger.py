from enum import Enum
from collections import OrderedDict
from google.cloud.bigtable.row import Cell, PartialRowData, InvalidChunk

_MISSING_COLUMN_FAMILY = "Column family {} is not among the cells stored in this row."
_MISSING_COLUMN = (
    "Column {} is not among the cells stored in this row in the column family {}."
)
_MISSING_INDEX = (
    "Index {!r} is not valid for the cells stored in this row for column {} "
    "in the column family {}. There are {} such cells."
)


class _State(Enum):
    ROW_START = "ROW_START"
    CELL_START = "CELL_START"
    CELL_IN_PROGRESS = "CELL_IN_PROGRESS"
    CELL_COMPLETE = "CELL_COMPLETE"
    ROW_COMPLETE = "ROW_COMPLETE"


class _PartialRow(object):
    __slots__ = [
        "row_key",
        "cells",
        "last_family",
        "last_family_cells",
        "last_qualifier",
        "last_qualifier_cells",
        "cell",
    ]

    def __init__(self, row_key):
        self.row_key = row_key
        self.cells = OrderedDict()

        self.last_family = None
        self.last_family_cells = OrderedDict()
        self.last_qualifier = None
        self.last_qualifier_cells = []

        self.cell = None


class _PartialCell(object):
    __slots__ = ["family", "qualifier", "timestamp", "labels", "value", "value_index"]

    def __init__(self):
        self.family = None
        self.qualifier = None
        self.timestamp = None
        self.labels = None
        self.value = None
        self.value_index = 0


class _RowMerger(object):
    """
    State machine to merge chunks from a response stream into logical rows.

    The implementation is a fairly linear state machine that is implemented as
    a method for every state in the _State enum. In general the states flow
    from top to bottom with some repetition. Each state handler will do some
    sanity checks, update in progress data and set the next state.

    There can be multiple state transitions for each chunk, i.e. a single chunk
    row will flow from ROW_START -> CELL_START -> CELL_COMPLETE -> ROW_COMPLETE
    in a single iteration.
    """

    __slots__ = ["state", "last_seen_row_key", "row"]

    def __init__(self, last_seen_row=b""):
        self.last_seen_row_key = last_seen_row
        self.state = _State.ROW_START
        self.row = None

    def process_chunks(self, response):
        """
        Process the chunks in the given response and yield logical rows.
        This class will maintain state across multiple response protos.
        """
        if response.last_scanned_row_key:
            if self.last_seen_row_key >= response.last_scanned_row_key:
                raise InvalidChunk("Last scanned row key is out of order")
            self.last_seen_row_key = response.last_scanned_row_key

        for chunk in response.chunks:
            if chunk.reset_row:
                self._handle_reset(chunk)
                continue

            if self.state == _State.ROW_START:
                self._handle_row_start(chunk)

            if self.state == _State.CELL_START:
                self._handle_cell_start(chunk)

            if self.state == _State.CELL_IN_PROGRESS:
                self._handle_cell_in_progress(chunk)

            if self.state == _State.CELL_COMPLETE:
                self._handle_cell_complete(chunk)

            if self.state == _State.ROW_COMPLETE:
                yield self._handle_row_complete(chunk)
            elif chunk.commit_row:
                raise InvalidChunk(
                    f"Chunk tried to commit row in wrong state (${self.state})"
                )

    def _handle_reset(self, chunk):
        if self.state == _State.ROW_START:
            raise InvalidChunk("Bare reset")
        if chunk.row_key:
            raise InvalidChunk("Reset chunk has a row key")
        if chunk.HasField("family_name"):
            raise InvalidChunk("Reset chunk has family_name")
        if chunk.HasField("qualifier"):
            raise InvalidChunk("Reset chunk has qualifier")
        if chunk.timestamp_micros:
            raise InvalidChunk("Reset chunk has a timestamp")
        if chunk.labels:
            raise InvalidChunk("Reset chunk has labels")
        if chunk.value:
            raise InvalidChunk("Reset chunk has a value")

        self.state = _State.ROW_START
        self.row = None

    def _handle_row_start(self, chunk):
        if not chunk.row_key:
            raise InvalidChunk("New row is missing a row key")
        if self.last_seen_row_key and self.last_seen_row_key >= chunk.row_key:
            raise InvalidChunk("Out of order row keys")

        self.row = _PartialRow(chunk.row_key)
        self.state = _State.CELL_START

    def _handle_cell_start(self, chunk):
        # Ensure that all chunks after the first one either are missing a row
        # key or the row is the same
        if self.row.cells and chunk.row_key and chunk.row_key != self.row.row_key:
            raise InvalidChunk("row key changed mid row")

        if not self.row.cell:
            self.row.cell = _PartialCell()

        # Cells can inherit family/qualifier from previous cells
        # However if the family changes, then qualifier must be specified as well
        if chunk.HasField("family_name"):
            self.row.cell.family = chunk.family_name.value
            self.row.cell.qualifier = None
        if not self.row.cell.family:
            raise InvalidChunk("missing family for a new cell")

        if chunk.HasField("qualifier"):
            self.row.cell.qualifier = chunk.qualifier.value
        if self.row.cell.qualifier is None:
            raise InvalidChunk("missing qualifier for a new cell")

        self.row.cell.timestamp = chunk.timestamp_micros
        self.row.cell.labels = chunk.labels

        if chunk.value_size > 0:
            # explicitly avoid pre-allocation as it seems that bytearray
            # concatenation performs better than slice copies.
            self.row.cell.value = bytearray()
            self.state = _State.CELL_IN_PROGRESS
        else:
            self.row.cell.value = chunk.value
            self.state = _State.CELL_COMPLETE

    def _handle_cell_in_progress(self, chunk):
        # if this isn't the first cell chunk, make sure that everything except
        # the value stayed constant.
        if self.row.cell.value_index > 0:
            if chunk.row_key:
                raise InvalidChunk("found row key mid cell")
            if chunk.HasField("family_name"):
                raise InvalidChunk("In progress cell had a family name")
            if chunk.HasField("qualifier"):
                raise InvalidChunk("In progress cell had a qualifier")
            if chunk.timestamp_micros:
                raise InvalidChunk("In progress cell had a timestamp")
            if chunk.labels:
                raise InvalidChunk("In progress cell had labels")

        self.row.cell.value += chunk.value
        self.row.cell.value_index += len(chunk.value)

        if chunk.value_size > 0:
            self.state = _State.CELL_IN_PROGRESS
        else:
            self.row.cell.value = bytes(self.row.cell.value)
            self.state = _State.CELL_COMPLETE

    def _handle_cell_complete(self, chunk):
        # since we are guaranteed that all family & qualifier cells are
        # contiguous, we can optimize away the dict lookup by caching the last
        # family/qualifier and simply comparing and appending
        family_changed = False
        if self.row.last_family != self.row.cell.family:
            family_changed = True
            self.row.last_family = self.row.cell.family
            self.row.cells[
                self.row.cell.family
            ] = self.row.last_family_cells = OrderedDict()

        if family_changed or self.row.last_qualifier != self.row.cell.qualifier:
            self.row.last_qualifier = self.row.cell.qualifier
            self.row.last_family_cells[
                self.row.cell.qualifier
            ] = self.row.last_qualifier_cells = []

        self.row.last_qualifier_cells.append(
            Cell(
                self.row.cell.value,
                self.row.cell.timestamp,
                self.row.cell.labels,
            )
        )

        self.row.cell.timestamp = 0
        self.row.cell.value = None
        self.row.cell.value_index = 0

        if not chunk.commit_row:
            self.state = _State.CELL_START
        else:
            self.state = _State.ROW_COMPLETE

    def _handle_row_complete(self, chunk):
        new_row = PartialRowData(self.row.row_key)
        new_row._cells = self.row.cells

        self.last_seen_row_key = new_row.row_key
        self.row = None
        self.state = _State.ROW_START

        return new_row

    def finalize(self):
        """
        Must be called at the end of the stream to ensure there are no unmerged
        rows.
        """
        if self.row or self.state != _State.ROW_START:
            raise ValueError("The row remains partial / is not committed.")
