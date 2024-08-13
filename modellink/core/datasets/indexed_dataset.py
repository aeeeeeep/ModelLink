from typing import List

import numpy


def add_item_from_list(self, lst: List, mode: int = 0) -> None:
    """Add a single item to the dataset

    Args:
        lst (list): The item to add to the data file

        mode (int, optional): The mode for the item. Defaults to 0.
    """
    np_array = numpy.array(lst, dtype=self.dtype)
    self.data_file.write(np_array.tobytes(order="C"))
    self.sequence_lengths.append(len(lst))
    if self.multimodal:
        self.sequence_modes.append(mode)


