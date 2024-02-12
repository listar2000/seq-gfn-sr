from typing import Optional, ClassVar, Tuple, cast

import torch
from gfn.actions import Actions
from gfn.env import Env, DiscreteEnv
from gfn.states import States, DiscreteStates
from torchtyping import TensorType as TT

from actions.action_meta import ActionMeta
from envs.preprocess import OneHotLSTMPreprocessor
from envs.states import make_pre_order_states


class PreOrderEnv(DiscreteEnv):
    def __init__(
        self,
        max_token_length: int,
        action_meta: ActionMeta,
        placeholder: int = -2,
        device_str: Optional[str] = None,
    ):
        assert placeholder != -1, "placeholder cannot use the same index as empty token"
        # state dimension equals 1 + maximum token allowed
        # the first entry of state tensor is reserved for holding # of tokens to insert next
        self.state_dim = max_token_length + 1
        self.action_meta = action_meta
        s0 = -torch.ones(self.state_dim, dtype=torch.int8)  # fill in empty token
        s0[0] = 1

        n_actions = len(self.action_meta.action_dict) + 1  # plus 1 to account for exit action
        preprocessor = OneHotLSTMPreprocessor()  # TODO: not implemented yet

        super().__init__(
            n_actions=n_actions,
            s0=s0,
            device_str=device_str,
            preprocessor=preprocessor
        )

    def make_States_class(self) -> type[States]:
        # decouple the state generation logic from here
        return make_pre_order_states(self)

    def maskless_step(self, states: States, actions: Actions) -> TT["batch_shape", "state_shape", torch.int8]:
        pass

    def maskless_backward_step(self, states: States, actions: Actions) -> TT["batch_shape", "state_shape", torch.int8]:
        pass

    def is_action_valid(self, states: States, actions: Actions, backward: bool = False) -> bool:
        pass

    def log_reward(self, final_states: States) -> TT["batch_shape", torch.int8]:
        pass
