from typing import List, Type

import numpy

from megatron.core.datasets.indexed_dataset import _IndexWriter


class BufferWriter:
    def __init__(self, data_file, dtype, buffer_threshold=10**5):
        self.data_file = data_file
        self.dtype = dtype
        self.buffer_threshold = buffer_threshold
        self.buffer = []

    def reset_buffer(self):
        self.buffer = []

    def write(self):
        if self.buffer:
            buffer_array = numpy.array(self.buffer, dtype=self.dtype)
            self.data_file.write(buffer_array.tobytes(order="C"))
            self.reset_buffer()

    def add(self, lst: List):
        self.buffer.extend(lst)

        if len(self.buffer) >= self.buffer_threshold:
            self.write()


def indexed_dataset_init(
        self, bin_path: str, dtype: Type[numpy.number] = numpy.int32, multimodal: bool = False
) -> None:
    self.data_file = open(bin_path, "wb")
    self.dtype = dtype
    self.multimodal = multimodal

    self.sequence_lengths = []
    self.document_indices = [0]
    self.sequence_modes = [] if self.multimodal else None
    self.buffer_writer = BufferWriter(data_file=self.data_file, dtype=self.dtype)


def add_item_from_list(self, lst: List, mode: int = 0) -> None:
    """Add a single item to the dataset

    Args:
        lst (list): The item to add to the data file

        mode (int, optional): The mode for the item. Defaults to 0.
    """
    self.buffer_writer.add(lst)
    self.sequence_lengths.append(len(lst))
    if self.multimodal:
        self.sequence_modes.append(mode)


def indexed_dataset_finalize(self, idx_path: str) -> None:
    """Clean up and write the index (.idx) file

    Args:
        idx_path (str): The path to the index file
    """
    self.buffer_writer.write()
    self.data_file.close()
    with _IndexWriter(idx_path, self.dtype) as writer:
        writer.write(self.sequence_lengths, self.sequence_modes, self.document_indices)

