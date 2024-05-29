from tqdm import tqdm
from torch import distributed as dist


def emit(self, record):
    try:
        rank = dist.get_rank()
    except Exception:
        rank = -1 # 如果获取rank失败，则设置为一个不合法的rank

    if rank == 0 or rank == -1:
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)