"""
This file contains the different "dictionaries" (metas) of available actions.
"""
from abc import ABC
from typing import Dict, Callable, List

from torchtyping import TensorType as TT
import torch


class ActionMeta(ABC):
    OPERATORS: Dict[str, Callable]  # OPERATORS := unary funcs
    FUNCTIONS: Dict[str, Callable]  # FUNCTIONS := binary funcs

    def __init__(self, num_features: int, has_constant: bool = True):
        # make sure we have at least 1 unary, binary, and feature elements respectively
        assert len(self.OPERATORS) > 0 and len(self.FUNCTIONS) > 0 and num_features > 0

        # saving some metadata
        self.has_constant = has_constant
        self.feat_num = num_features + 1 if has_constant else num_features
        self.op_num, self.fn_num = len(self.OPERATORS), len(self.FUNCTIONS)

        # include the constant token if needed
        self.action_dict = {"c": None} if self.has_constant else {}
        self.action_dict.update({
            **{f'x{idx + 1}': idx for idx in range(num_features)},  # features
            **self.OPERATORS,  # operators
            **self.FUNCTIONS,  # functions
        })

        # different types of masks
        self.fm = torch.ones(len(self.action_dict), dtype=torch.bool)
        self.fm[self.feat_num:] = False

        self.nbm = torch.ones(len(self.action_dict), dtype=torch.bool)
        self.nbm[self.feat_num + self.op_num:] = False

    @property
    def feature_only_mask(self):
        return self.fm

    @property
    def no_binary_fn_mask(self):
        return self.nbm

    @property
    def operator_num(self) -> int:
        return self.op_num

    @property
    def function_num(self) -> int:
        return self.fn_num

    @property
    def feature_num(self) -> int:
        return self.feat_num

    @property
    def action_names(self) -> List[str]:
        return list(self.action_dict.keys())

    @property
    def action_fns(self) -> List[Callable]:
        return list(self.action_dict.values())

    @property
    def action_arities(self) -> List[int]:
        """
        A list containing how many arguments (arities) that each action would take
        """
        return self.feat_num * [0] + self.op_num * [1] + self.fn_num * [2]

    def calculate_action_arities(self, action_tensor: TT["batch_size", torch.long])\
            -> TT["batch_size", "action_space", torch.long]:
        assert (action_tensor < len(self.action_dict)).all()  # make sure exit action is not involved
        arity_tensor = torch.zeros_like(action_tensor, dtype=torch.long)
        arity_tensor[action_tensor >= self.feat_num] += 1  # add one for unary operators
        arity_tensor[action_tensor >= self.feat_num + self.op_num] += 1  # another add for binary functions
        return arity_tensor


class DummyActionMeta(ActionMeta):
    OPERATORS = {
        'square': torch.square,
        'cos': torch.cos,
        'sin': torch.sin,
    }

    FUNCTIONS = {
        '*': torch.mul,
        '+': torch.add,
        '/': torch.div,
        '-': torch.sub
    }


class DefaultActionMeta(ActionMeta):
    OPERATORS = {
        'square': torch.square,
        'sqrt': torch.sqrt,
        'log': torch.log,
        'cos': torch.cos,
        'sin': torch.sin,
        'exp': torch.exp,
    }

    FUNCTIONS = {
        '*': torch.mul,
        '+': torch.add,
        '/': torch.div,
        '-': torch.sub
    }
