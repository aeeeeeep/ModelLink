from atb_speed.common.config import atb_speed_config

from .base import CEVALPrecisionTest, MMLUPrecisionTest


def get_precision_test_cls(mode=""):
    """

    :return:
    """
    cls_map = {
        "mmlu": MMLUPrecisionTest,
        "ceval": CEVALPrecisionTest
    }
    return cls_map.get(mode or atb_speed_config.precision.mode.lower(), CEVALPrecisionTest)
