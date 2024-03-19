import time
import torch


def _compile_dependencies():
    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise ZeroDivisionError
    if torch.distributed.get_rank() % device_count == 0:
        start_time = time.time()
        print('> compiling dataset index builder ...')
        from megatron.core.datasets.utils import compile_helpers
        compile_helpers()
        print('>>> done with dataset index builder. Compilation time: {:.3f} '
              'seconds'.format(time.time() - start_time), flush=True)