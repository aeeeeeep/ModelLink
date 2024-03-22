# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
from base import model_test


class Bloom7BModelTest(model_test.ModelTest):
    def __init__(self, *args) -> None:
        super().__init__(*args)

    def get_model(self, hardware_type, model_type, data_type):
        pass

    def get_dataset_list(self):
        return ["CEval"]
    
    def get_chip_num(self):
        return 8


def main():
    Bloom7BModelTest.create_instance()

if __name__ == "__main__":
    main()
