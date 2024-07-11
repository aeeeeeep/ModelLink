# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
from mindspeed.patch_utils import Patch


class PatchesManager:
    _patches_info = {}

    @staticmethod
    def register_patch(orig_func_name, new_func=None, force_patch=False, create_dummy=False):
        if orig_func_name not in PatchesManager._patches_info:
            PatchesManager._patches_info[orig_func_name] = Patch(orig_func_name, new_func, create_dummy)
        else:
            PatchesManager._patches_info.get(orig_func_name).set_patch_func(new_func, force_patch)

    @staticmethod
    def apply_patches():
        for patch in PatchesManager._patches_info.values():
            patch.apply_patch()
