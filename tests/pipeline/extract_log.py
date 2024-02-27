from abc import ABC, abstractmethod
import json
import argparse

class BaseLogExtractor(ABC):
    def __init__(self, input_path:str, out_path_prefix:str):
        self.input_path = input_path
        self.out_path_prefix = out_path_prefix
        self.losses = []
        self.memories = []
        self.parameters = {"global_batch_size": 0,
                           "seq_length": 0,
                           "world_size": 0}

    @abstractmethod
    def _extract_parameter(self, line: str) -> None:
        pass

    @abstractmethod
    def _extract_iterline(self, line: str) -> None:
        pass

    @abstractmethod
    def _extract_memory(self, line) -> None:
        pass

    def _extract(self) -> None:
        with open(self.input_path) as f:
            for line in f:
                self._extract_parameter(line)
                self._extract_iterline(line)
                self._extract_memory(line)
                    
    
    def _save(self) -> None:
        loss_path = f"{self.out_path_prefix}_loss.tsv"
        with open(loss_path, 'w') as f:
            f.write("step\tloss\titer_time\n")
            for step, loss, iter_time in self.losses:
                f.write(f"{step}\t{loss}\t{iter_time}\n")
        
        memory_path = f"{self.out_path_prefix}_memory.tsv"
        with open(memory_path, 'w') as f:
            f.write("rank_id\tmemory\n")
            for rank_id, memory in sorted(self.memories):
                f.write(f"{rank_id}\t{memory}\n")
        
        parameters_path = f"{self.out_path_prefix}_parameters.json"
        with open(parameters_path, 'w') as f:
            json.dump(self.parameters, f, indent=4)
    
    
    def __call__(self):
        self._extract()
        self._save()

 
class MegatronLogExtractor(BaseLogExtractor):

    def _extract_parameter(self, line: str) -> None:
        for param in self.parameters.keys():
            if line.startswith(f"  {param}"):
                blank_pos = line.rfind(' ')
                self.parameters[param] = int(line[blank_pos:])

    def _extract_iterline(self, line: str):
        if not line.startswith(" iteration"):
            return

        backslash_pos = line.find('/')
        blank_pos = line.rfind(' ', 0, backslash_pos)
        step = line[blank_pos:backslash_pos]
        ms_pos = line.find('(ms):')
        pipe_pos = line.find('|', ms_pos)
        iter_time = line[ms_pos+6: pipe_pos-1]
        loss_pos = line.find('lm loss:')
        if loss_pos > 0:
            bar_pos = line.find('|', loss_pos)
            loss = line[loss_pos+9:bar_pos-1]
        else:
            loss = 'inf'
        self.losses.append((int(step), float(loss), float(iter_time)))
    
    def _extract_memory(self, line) -> None:
        if not line.startswith("[Rank"):
            return
        
        start = 0
        while start >= 0:
            rsb_pos = line.find(']', start)
            rankid = line[start+6:rsb_pos]
            mem_pos = line.find('allocated:', rsb_pos)
            pipe_pos = line.find('|', mem_pos)
            memory = line[mem_pos+11:pipe_pos-1]
            self.memories.append((int(rankid), float(memory)))
            start = line.find("[Rank", pipe_pos)

def main(args):
    if args.frame_kind.lower() == 'megatron':
        MegatronLogExtractor(args.input_path, args.output_path_prefix)()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract loss, performance and memory data from training log')
    parser.add_argument('frame_kind', help='The training frame: Megatron, Deepspeed or Torch')
    parser.add_argument('input_path', help='The training log path')
    parser.add_argument('output_path_prefix', help='The output path prefix')
    args = parser.parse_args()
    main(args)