from typing import ClassVar, Tuple, cast

import torch
from gfn.env import DiscreteEnv
from torchtyping import TensorType as TT
from gfn.states import DiscreteStates


def make_pre_order_states(env: DiscreteEnv) -> DiscreteStates:
    class PreOrderStates(DiscreteStates):
        state_shape: ClassVar[tuple[int, ...]] = (env.state_dim,)
        s0 = env.s0
        sf = env.sf
        n_actions = env.n_actions
        action_meta = env.action_meta
        device = env.device

        @classmethod
        def make_random_states_tensor(
                cls, batch_shape: Tuple[int, ...]
        ) -> TT["batch_shape", "state_shape", torch.float]:
            raise NotImplementedError("Random states not supported in pre-order GFN-SR")

        def update_masks(self) -> None:
            """Update the masks based on the current states."""
            # reset forward_masks to all true, backward_masks to all zero
            self.forward_masks[:], self.backward_masks[:] = True, False
            # The following two lines are for typing only.
            self.forward_masks = cast(
                TT["batch_shape", "n_actions", torch.bool],
                self.forward_masks,
            )
            self.backward_masks = cast(
                TT["batch_shape", "n_actions - 1", torch.bool],
                self.backward_masks,
            )
            remain_space, remain_token = self.tensor[..., 0], self.tensor[..., 1]
            assert (remain_space >= remain_token).all(), \
                "# of remaining token should at least be bigger than # of space in tensor state"

            # Part I: we only mask for states whose construction has not finished yet
            undone_mask = remain_token > 0

            undone_forward_masks = self.forward_masks[undone_mask]
            done_forward_masks = self.forward_masks[~undone_mask]

            undone_forward_masks[..., -1] = False
            done_forward_masks[..., :-1] = False

            # compare remain_space against remain_token for undone tensor
            diff_token = remain_space[undone_mask] - remain_token[undone_mask]

            # case 1: if remain_space == remain_token != 0, then we can only fill feature/constant
            undone_forward_masks[diff_token == 0, :-1] = self.action_meta.feature_only_mask

            # case 2: if remain_space - remain_token == 1, then we can only fill feature/constant/unary op
            undone_forward_masks[diff_token == 1, :-1] = self.action_meta.no_binary_fn_mask

            # case 3: in other cases, every token is possible
            self.forward_masks[undone_mask] = undone_forward_masks
            self.forward_masks[~undone_mask] = done_forward_masks

            # Part II: backward mask for pre-order env is trivial as the state space is a tree
            # we only consider backward mask for those who has taken at least 1 step & non-sink
            is_valid_state = torch.logical_and(~self.is_sink_state, ~self.is_initial_state)

            if not is_valid_state.any():
                return

            valid_backward_masks = self.backward_masks[is_valid_state, :]

            # calculate the indices of the most recently added token
            recent_idx = (self.state_shape[0] - 1 - remain_space[is_valid_state]).long()

            recent_val = torch.gather(self.tensor[is_valid_state],
                                      -1, recent_idx.unsqueeze(-1)).squeeze(-1)

            valid_backward_masks[torch.arange(valid_backward_masks.shape[0]), recent_val] = True
            # idx = torch.meshgrid([torch.arange(recent_val.size(dim)) for dim in range(len(self.batch_shape))])
            # valid_backward_masks[(idx + (recent_val,))] = True
            # there might be terminal states so we need to set all actions in `backward_masks` to False
            self.backward_masks[is_valid_state, :] = valid_backward_masks

    return PreOrderStates
