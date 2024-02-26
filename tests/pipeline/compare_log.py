import json
import argparse
from typing import Tuple, Optional
import pandas as pd

class Comparator:
    def __init__(self,
                 base_path_prefix: str,
                 test_path_prefix: str,
                 loss_error_rate: float = 0.02,
                 perf_error_rate: float = 0.03,
                 mem_error_rate: float = 0.003,
                 compute_steps: int = 2000):
        self.base_path_prefix = base_path_prefix
        self.test_path_prefix = test_path_prefix
        self.loss_error_rate = loss_error_rate
        self.perf_error_rate = perf_error_rate
        self.mem_error_rate = mem_error_rate
        self.compute_steps = compute_steps
    
    def _read_check_loss_file(self) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, int]]:
        base_loss_pd = pd.read_csv(f"{self.base_path_prefix}_loss.tsv", sep='\t')
        test_loss_pd = pd.read_csv(f"{self.test_path_prefix}_loss.tsv", sep='\t')
        if len(base_loss_pd) < self.compute_steps or len(test_loss_pd) < self.compute_steps:
            print("The log doesn't have enough steps to compute!")
            return None
        
        base_loss_start = base_loss_pd.loss.ne(float('inf')).argmax()
        test_loss_start = test_loss_pd.loss.ne(float('inf')).argmax()
        if base_loss_start != test_loss_start:
            print("The validate loss step is not equal!")
            return None
        
        return base_loss_pd, test_loss_pd, base_loss_start



    def compare_memory(self) -> bool:
        base_mem_pd = pd.read_csv(f"{self.base_path_prefix}_memory.tsv", sep='\t')
        test_mem_pd = pd.read_csv(f"{self.test_path_prefix}_memory.tsv", sep='\t')
        base_mem_mean = base_mem_pd.memory.mean()
        test_mem_mean = test_mem_pd.memory.mean()
        if abs((test_mem_mean - base_mem_mean) / base_mem_mean) > self.mem_error_rate:
            print("Memory test failed!")
            return False
        
        print("Memory test pass!")
        return True

    def compare_perf(self) -> bool:
        result = self._read_check_loss_file()
        if not result:
            return False
        
        base_loss_pd, test_loss_pd, loss_start = result

        with open(f"{self.base_path_prefix}_parameters.json") as f:
            base_params = json.load(f)
        
        with open(f"{self.test_path_prefix}_parameters.json") as f:
            test_params = json.load(f)
        
        if base_params != test_params:
            print("The parameters are not equal")
            return False

        global_batch_size = base_params["global_batch_size"]
        seq_length = base_params["seq_length"]
        world_size = base_params["world_size"]
    
        # Here we need to skip the first steps until the training is stable
        base_itertime_mean = base_loss_pd[loss_start+1:self.compute_steps].iter_time.mean()
        test_itertime_mean = test_loss_pd[loss_start+1:self.compute_steps].iter_time.mean()

        base_perf = global_batch_size * seq_length / world_size / base_itertime_mean
        test_perf = global_batch_size * seq_length / world_size / test_itertime_mean

        if abs((test_perf - base_perf) / base_perf) > self.perf_error_rate:
            print("Perf test failed!")
            return False
        
        print("Perf test pass!")
        return True


    def compare_loss(self) -> bool:
        result = self._read_check_loss_file()
        if not result:
            return False
        
        base_loss_pd, test_loss_pd, loss_start = result
        
        loss_error_rates = (test_loss_pd[loss_start:self.compute_steps].loss - base_loss_pd[loss_start:self.compute_steps].loss) / base_loss_pd[loss_start:self.compute_steps].loss
        if abs(loss_error_rates.mean()) > self.loss_error_rate:
            print("Loss test failed!")
            return False
        
        print("Loss test pass!")
        return True


    def __call__(self) -> None:
        self.compare_loss()
        self.compare_perf()
        self.compare_memory()

def main(args):
    Comparator(args.base_path_prefix,
               args.test_path_prefix,
               args.loss_error_rate,
               args.perf_error_rate,
               args.mem_error_rate,
               args.compute_steps)()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare log data between baseline and test')
    parser.add_argument('base_path_prefix', help='The baseline path prefix')
    parser.add_argument('test_path_prefix', help='The test path prefix')
    parser.add_argument('--loss_error_rate', type=float, default=0.02, help='The loss error rate')
    parser.add_argument('--perf_error_rate', type=float, default=0.03, help='The perf error rate')
    parser.add_argument('--mem_error_rate', type=float, default=0.003, help='The memory error rate')
    parser.add_argument('--compute_steps', type=int, default=2000, help='The compute steps')
    args = parser.parse_args()
    main(args)