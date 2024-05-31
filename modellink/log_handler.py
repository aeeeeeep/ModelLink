import sys
from tqdm import tqdm
from torch import distributed as dist


def get_rank():
    try:
        return dist.get_rank()
    except Exception:
        return -1


def emit(self, record):
    rank = get_rank()
    if rank == 0 or rank == -1:
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


class StdoutOnlyRank0:
    def __init__(self):
        self.origin_stdout = sys.stdout

    def write(self, message):
        rank = get_rank()
        if rank == 0 or rank == -1:
            try:
                self.origin_stdout.write(message)
            except Exception:
                self.handleError(message)

    def flush(self):
        self.origin_stdout.flush()