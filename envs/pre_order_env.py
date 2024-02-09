from typing import Optional, ClassVar, Tuple, cast

import torch
from gfn.actions import Actions
from gfn.env import Env, DiscreteEnv
from gfn.states import States, DiscreteStates
from torchtyping import TensorType as TT

from actions.action_meta import ActionMeta
from envs.preprocess import OneHotLSTMPreProcessor
from envs.states import make_pre_order_states


class PreOrderEnv(DiscreteEnv):
    def __init__(
        self,
        max_depth: int,
        action_meta: ActionMeta,
        placeholder: int = -2,
        device_str: Optional[str] = None,
    ):
        assert placeholder != -1, "placeholder cannot use the same index as empty token"
        # state dimension equals the size of full binary tree with depth `max_depth`
        self.state_dim = 2 ** max_depth - 1
        s0 = -torch.ones(self.state_dim)  # fill in empty token

        n_actions = len(action_meta.actions_dict) + 1  # plus 1 to account for exit action
        preprocessor = OneHotLSTMPreProcessor()  # TODO: not implemented yet

        super().__init__(
            n_actions=n_actions,
            s0=s0,
            device_str=device_str,
            preprocessor=preprocessor
        )

    def make_States_class(self) -> type[States]:
        # decouple the state generation logic from here
        return make_pre_order_states(self)

    def maskless_step(self, states: States, actions: Actions) -> TT["batch_shape", "state_shape", torch.float]:
        pass

    def maskless_backward_step(self, states: States, actions: Actions) -> TT["batch_shape", "state_shape", torch.float]:
        pass

    def is_action_valid(self, states: States, actions: Actions, backward: bool = False) -> bool:
        pass

    def log_reward(self, final_states: States) -> TT["batch_shape", torch.float]:
        pass
