from typing import ClassVar, Tuple, cast

import torch
from torchtyping import TensorType as TT
from gfn.states import DiscreteStates


def make_pre_order_states(env):
    class PreOrderStates(DiscreteStates):
        state_shape: ClassVar[tuple[int, ...]] = (env.state_dim,)
        s0 = env.s0
        sf = env.sf
        n_actions = env.n_actions
        device = env.device

        @classmethod
        def make_random_states_tensor(
                cls, batch_shape: Tuple[int, ...]
        ) -> TT["batch_shape", "state_shape", torch.float]:
            raise NotImplementedError("Random states not supported in pre-order GFN-SR")

        def update_masks(self) -> None:
            """Update the masks based on the current states."""
            # The following two lines are for typing only.
            self.forward_masks = cast(
                TT["batch_shape", "n_actions", torch.bool],
                self.forward_masks,
            )
            self.backward_masks = cast(
                TT["batch_shape", "n_actions - 1", torch.bool],
                self.backward_masks,
            )

            # TODO: implement actual logic for forward masking (now there's no mask)
            self.backward_masks.zero_()

            recent_mask = (self.tensor >= 0).long()
            recent_updated_idx = torch.argmax(recent_mask, dim=1).unsqueeze(1)

            recent_updated_vals = torch.gather(self.tensor, 1, recent_updated_idx)
            self.backward_masks[torch.arange(self.batch_shape), recent_updated_vals] = True

    return PreOrderStates
