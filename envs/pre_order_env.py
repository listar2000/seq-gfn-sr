from typing import Optional, ClassVar, Tuple, cast

import torch
from gfn.actions import Actions
from gfn.env import Env, DiscreteEnv
from gfn.preprocessors import IdentityPreprocessor
from gfn.states import States, DiscreteStates
from torchtyping import TensorType as TT

from actions.action_meta import ActionMeta
# from envs.preprocess import OneHotLSTMPreprocessor
from envs.states import make_pre_order_states


class PreOrderEnv(DiscreteEnv):
    def __init__(
            self,
            max_token_length: int,
            action_meta: ActionMeta,
            placeholder: int = -1,
            device_str: Optional[str] = "cpu",
    ):
        assert placeholder < 0, "placeholder must be negative"
        self.placeholder = placeholder
        # the first entry of state tensor is reserved for the # token unfilled
        # the second entry of the state tensor is reserved for the # token need to be filled for valid tree
        # [# unfilled, # to fill, ...max token length...]
        self.state_dim = max_token_length + 2
        self.action_meta = action_meta

        device = torch.device(device_str)
        s0 = self.placeholder * torch.ones(self.state_dim, dtype=torch.long, device=device)  # fill in empty token
        s0[0], s0[1] = max_token_length, 1

        sf = -torch.ones(self.state_dim, dtype=torch.long, device=device)  # sink state should be LongTensor
        sf[0:2] = 0

        n_actions = len(self.action_meta.action_dict) + 1  # plus 1 to account for exit action
        # preprocessor = OneHotLSTMPreprocessor(output_dim=0)  # TODO: not implemented yet
        preprocessor = IdentityPreprocessor(output_dim=self.state_dim)

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
        """
        The few things we handle in the maskless forward step
        1. find states that are not finished yet.
        2. Apply the actions to these states (insert to first vacant space in state tensor)
        3. Update the # of remaining spaces and # of remaining tokens, based on action arities
        """
        assert len(states.tensor.shape) <= 2
        action_tensor = actions.tensor
        state_tensor = states.tensor
        undone_mask = state_tensor[:, 1] > 0
        # make sure that the actions on done states is only the exit option
        assert (action_tensor[~undone_mask] == (self.n_actions - 1)).all(), "Done state can only exit!"

        undone_state_tensor = state_tensor[undone_mask]
        undone_action_tensor = action_tensor[undone_mask]
        # the next index to insert equals state dimension (i.e. tensor length) minus # of space left
        new_idx = self.state_dim - undone_state_tensor[:, 0]
        # fill the next empty spot (decided by `new_idx`) with the corresponding action
        new_undone_state_tensor = undone_state_tensor.scatter(-1, new_idx.unsqueeze(1), undone_action_tensor)
        # decrement # of empty space and # of token filled after applying action
        new_undone_state_tensor[:, 0:2] -= 1
        # depending on what action is inserted, increase # of token to be filled
        new_undone_state_tensor[:, 1] += self.action_meta.calculate_action_arities(undone_action_tensor).squeeze(1)

        new_state_tensor = state_tensor.clone()
        new_state_tensor[undone_mask] = new_undone_state_tensor
        return new_state_tensor

    def maskless_backward_step(self, states: States, actions: Actions) -> TT["batch_shape", "state_shape", torch.long]:
        assert len(states.tensor.shape) <= 2
        # make sure no state is in initial state
        action_tensor, state_tensor = actions.tensor, states.tensor
        assert not torch.eq(state_tensor, self.s0).all(dim=-1).any(), "State tensor cannot be initial state"

        # the most recently modified part of state tensor has index state_dim - place_left - 1
        recent_idx = self.state_dim - state_tensor[:, 0] - 1
        # modify the state tensor back by filling placeholder value (negative value default to be -1)
        new_state_tensor = state_tensor.scatter(-1, recent_idx.unsqueeze(1), self.placeholder)
        return new_state_tensor

    def log_reward(self, final_states: States) -> TT["batch_shape", torch.long]:
        # TODO: implement the actual probabilistic reward (based on MSE/RMSE)
        # currently implementing a uniform reward
        return torch.log(torch.ones(final_states.tensor.size(0)))

