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
        device_str: Optional[str] = None,
    ):
        # the first entry of state tensor is reserved for the # token unfilled
        # the second entry of the state tensor is reserved for the # token need to be filled for valid tree
        # [# unfilled, # to fill, ...max token length...]
        self.state_dim = max_token_length + 2
        self.action_meta = action_meta
        s0 = -torch.ones(self.state_dim, dtype=torch.long)  # fill in empty token
        s0[0], s0[1] = max_token_length, 1

        sf = -torch.ones(self.state_dim, dtype=torch.long)  # sink state should be LongTensor

        n_actions = len(self.action_meta.action_dict) + 1  # plus 1 to account for exit action
        preprocessor = OneHotLSTMPreprocessor(output_dim=0)  # TODO: not implemented yet

        super().__init__(
            n_actions=n_actions,
            s0=s0,
            sf=sf,
            device_str=device_str,
            preprocessor=preprocessor
        )

    def make_States_class(self) -> type[States]:
        # decouple the state generation logic from here
        return make_pre_order_states(self)

    def maskless_step(self, states: States, actions: Actions) -> TT["batch_shape", "state_shape", torch.long]:
        pass

    def maskless_backward_step(self, states: States, actions: Actions) -> TT["batch_shape", "state_shape", torch.long]:
        pass

    def is_action_valid(self, states: States, actions: Actions, backward: bool = False) -> bool:
        pass

    def log_reward(self, final_states: States) -> TT["batch_shape", torch.long]:
        pass
