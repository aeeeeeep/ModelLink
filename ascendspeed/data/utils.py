#!/usr/bin/env python3
# coding=utf-8
"""
Author: changwanli
since: 2023-08-31 15:00:51
LastTime: 2023-11-16 17:36:41
LastAuthor: changwanli
message: 
Copyright (c) 2022 Wuhan Artificial Intelligence Research. All Rights Reserved 
"""
import json
import logging
import os

from tqdm import tqdm


def print_rank_0(msg, log_file, rank=0, save_log=False):
    if rank <= 0:
        print(msg)
        if save_log:
            with open(log_file, "a") as f:
                f.write(msg + "\n")


def get_file_lines(file_path: str) -> int:
    """
    get number of file lines

    Args:
        file_path (str): file path

    Returns:
        int: number of file lines
    """
    with os.popen(f"wc -l {file_path}") as f:
        total_lines = int(f.readlines()[0].split()[0])
    return total_lines


class LazyFile(object):
    def __init__(self, file_path, save_index=False, map_func=None, verbose=True):
        """
        read special line from file with less memory (only load line offset in memory, read line from dick by offset)


        Args:
            file_path (_type_): _description_
            save_index (bool, optional): whether save index of file. Defaults to False.
        """
        self.file_path = file_path
        self.verbose = verbose
        self.file_lines_number = get_file_lines(file_path)
        file_index_path = file_path + ".index.json"
        if not os.path.exists(file_index_path):
            self.load(file_path)
            if save_index:
                logging.warning("save index my cause error when using muti-process")
                self.write_index(file_index_path)
        else:
            self.load_with_index(file_index_path, file_path)
        self.map_func = map_func
        

    def __del__(self):
        """
        close file
        """
        if hasattr(self, "fin"):
            self.fin.close()

    def load(self, file_path: str):
        """
        load file and generate index

        Args:
            file_path (str): _description_
        """
        self.offset_mapping = self.get_offset_mapping(file_path)
        # self.fin = open(file_path, "r", encoding="utf8")

    def get_offset_mapping(self, file_name: str) -> list:
        """
        generate line number to offset mapping

        Args:
            file_name (str): _description_

        Returns:
            list: offset mapping
        """
        key_2_offset_map = []
        with open(file_name, "r", encoding="utf-8") as f:
            offset = 0
            for idx, line in enumerate(
                tqdm(
                    iter(f.readline, ""),
                    total=self.file_lines_number,
                    desc=f"generate index of {file_name}",
                    disable=not self.verbose
                )
            ):
                key_2_offset_map.append(offset)
                offset = f.tell()
                if not line:
                    break
        return key_2_offset_map

    def write_index(self, output_file_path: str):
        """
        save index to output_file_path

        Args:
            output_file_path (str): _description_
        """
        json.dump(self.offset_mapping, open(output_file_path, "w"))
        logging.warning(f"save index to {output_file_path}")

    def load_with_index(self, index_file_path, file_path):
        """
        load file and index

        Args:
            index_file_path (_type_): _description_
            file_path (_type_): _description_
        """
        self.offset_mapping = json.load(open(index_file_path, "r"))

        assert len(self.offset_mapping) == self.file_lines_number, (
            f"totol lines of {file_path} is {self.file_lines_number}, but max index of"
            f" {index_file_path} is {len(self.offset_mapping)}"
        )
        # self.fin = open(file_path, "r", encoding="utf-8")
        logging.warning(f"load index of {file_path} from {index_file_path}")

    def __getitem__(self, key):
        """
        read special line from file by key

        Args:
            key (_type_): _description_

        Returns:
            _type_: _description_
        """
        offset = self.offset_mapping[key]
        with open(self.file_path, 'r',  encoding="utf-8") as f:
            f.seek(offset)
            value = f.readline().strip("\n")
            if self.map_func is not None:
                value = self.map_func(value)
            return value

        # self.fin.seek(offset)
        # value = self.fin.readline().strip("\n")
        # if self.map_func is not None:
        #     value = self.map_func(value)
        # return value

    def __len__(self):
        """
        file lines

        Returns:
            _type_: _description_
        """
        return len(self.offset_mapping)


class LazyFiles(object):
    def __init__(self, file_path_list: list, save_index=False, map_func=None, verbose=True) -> None:
        """
        merge multi file as one file, and read special line by key with less memory

        Args:
            file_path_list (list): _description_
            save_index (bool, optional): _description_. Defaults to False.
        """
        self.verbose = verbose
        self.file_path_list = sorted(
            [file_path for file_path in file_path_list if not file_path.endswith(".index.json")]
        )
        self.file_list = [
            LazyFile(file_path, save_index, map_func=map_func, verbose=verbose) for file_path in self.file_path_list
        ]
        self.idx_2_file_idx_map = self.get_idx_2_file_idx_map(self.file_list)
        self.file_length_intervels = self.get_file_length_intervels(self.file_list)

    def __len__(self):
        return len(self.idx_2_file_idx_map)

    def get_idx_2_file_idx_map(self, file_list):
        idx_2_file_idx_map = []
        for file_idx, file in enumerate(file_list):
            idx_2_file_idx_map.extend([file_idx] * len(file))
        return idx_2_file_idx_map

    def get_file_length_intervels(self, file_list):
        file_length_intervels = [0]
        start = 0
        for file in file_list:
            start += len(file)
            file_length_intervels.append(start)
        return file_length_intervels

    def __getitem__(self, idx):
        file_idx = self.idx_2_file_idx_map[idx]
        target_file = self.file_list[file_idx]
        idx_in_target_file = idx - self.file_length_intervels[file_idx]
        return target_file[idx_in_target_file]


if __name__ == "__main__":
    from glob import glob

    lazy_files = LazyFiles(glob("/public/changwanli/data/en_zh_debug/*"), save_index=False)
