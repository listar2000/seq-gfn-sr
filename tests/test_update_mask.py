import torch

from actions.action_meta import DummyActionMeta
from envs.pre_order_env import PreOrderEnv
from envs.states import make_pre_order_states


# TODO: more comprehensive test architecture
def _get_pre_order_env():
    action_meta = DummyActionMeta(1, 1)
    pre_order_env = PreOrderEnv(4, action_meta)
    return pre_order_env


def test_update_mask_1():
    pre_order_env = _get_pre_order_env()
    test_tensor = pre_order_env.s0.unsqueeze(0)
    state = make_pre_order_states(pre_order_env)(tensor=test_tensor)
    print(state.forward_masks)
    print(state.backward_masks)


def test_update_mask_2():
    pre_order_env = _get_pre_order_env()
    test_tensor = torch.tensor([[3, 1, 3, -1, -1, -1]], dtype=torch.long)
    state = make_pre_order_states(pre_order_env)(tensor=test_tensor)
    print(state.forward_masks)
    print(state.backward_masks)


def test_update_mask_3():
    pre_order_env = _get_pre_order_env()
    test_tensor = torch.tensor([[2, 2, 3, 5, -1, -1]], dtype=torch.long)
    state = make_pre_order_states(pre_order_env)(tensor=test_tensor)
    print(state.forward_masks)
    print(state.backward_masks)


def test_update_mask_4():
    pre_order_env = _get_pre_order_env()
    test_tensor = torch.tensor([[0, 0, 3, 5, 0, 0]], dtype=torch.long)
    state = make_pre_order_states(pre_order_env)(tensor=test_tensor)
    print(state.forward_masks)
    print(state.backward_masks)


if __name__ == '__main__':
    test_update_mask_4()
