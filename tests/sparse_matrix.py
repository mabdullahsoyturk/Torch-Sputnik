"""Defines primitive sparse matrix type for use with sparse ops."""
import numpy as np

import torch

import connectors
import initializers

def _dense_to_sparse(matrix):
    """Converts dense numpy matrix to a csr sparse matrix."""
    assert len(matrix.shape) == 2

    # Extract the nonzero values.
    values = matrix.compress((matrix != 0).flatten())

    # Calculate the offset of each row.
    mask = (matrix != 0).astype(np.int32)
    row_offsets = np.concatenate(([0], np.cumsum(np.add.reduce(mask, axis=1))),
                                 axis=0)

    # Create the row indices and sort them.
    row_indices = np.argsort(-1 * np.diff(row_offsets))

    # Extract the column indices for the nonzero values.
    x = mask * (np.arange(matrix.shape[1]) + 1)
    column_indices = x.compress((x != 0).flatten())
    column_indices = column_indices - 1

    # Cast the desired precision.
    values = values.astype(np.float32)
    row_indices, row_offsets, column_indices = [
        x.astype(np.int32) for x in
        [row_indices, row_offsets, column_indices]
    ]

    values = torch.from_numpy(values).cuda()
    row_indices = torch.from_numpy(row_indices).cuda()
    row_offsets = torch.from_numpy(row_offsets).cuda()
    column_indices = torch.from_numpy(column_indices).cuda()

    return values, row_indices, row_offsets, column_indices

class SparseTopology(object):
    """Describes a sparse matrix, with no values."""

    def __init__(self,
                 shape=None,
                 mask=None,
                 connector=connectors.Uniform(0.8),
                 dtype=torch.float32):
        if mask is None:
            assert shape is not None and len(shape) == 2
            mask = connector(np.ones(shape))
            self._shape = shape
        else:
            assert shape is None
            assert len(mask.shape) == 2
            self._shape = mask.shape
        self._dtype = dtype
        self._sparsity = 1.0 - np.count_nonzero(mask) / mask.size

        _, row_indices_, row_offsets_, column_indices_ = _dense_to_sparse(mask)

        self._row_indices = row_indices_
        self._row_offsets = row_offsets_
        self._column_indices = column_indices_

    @property
    def shape(self):
        return self._shape

    @property
    def size(self):
        return np.prod(self._shape)

    @property
    def dtype(self):
        return self._dtype

    @property
    def sparsity(self):
        return self._sparsity

    @property
    def row_indices(self):
        return self._row_indices

    @property
    def row_offsets(self):
        return self._row_offsets

    @property
    def column_indices(self):
        return self._column_indices