import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional


@dataclass
class DatasetAttr:
    r"""
    Dataset attributes.
    """

    """ basic configs """
    load_from: Literal["file"]
    dataset_name: str
    """ extra configs """
    subset: Optional[str] = None
    folder: Optional[str] = None
    ranking: bool = False
    formatting: Literal["alpaca", "sharegpt"] = "alpaca"
    """ columns """
    system: Optional[str] = None
    images: Optional[str] = None
    """ columns for the alpaca format """
    prompt: Optional[str] = "instruction"
    query: Optional[str] = "input"
    response: Optional[str] = "output"
    history: Optional[str] = None
    """ columns for the sharegpt format """
    messages: Optional[str] = "conversations"
    tools: Optional[str] = None
    """ tags for the sharegpt format """
    role_tag: Optional[str] = "from"
    content_tag: Optional[str] = "value"
    user_tag: Optional[str] = "human"
    assistant_tag: Optional[str] = "gpt"
    observation_tag: Optional[str] = "observation"
    function_tag: Optional[str] = "function_call"
    system_tag: Optional[str] = "system"

    def __repr__(self) -> str:
        return self.dataset_name

    def set_attr(self, key: str, obj: Dict[str, Any], default: Optional[Any] = None) -> None:
        setattr(self, key, obj.get(key, default))