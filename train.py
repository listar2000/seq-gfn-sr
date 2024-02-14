import torch
from tqdm import tqdm

from gfn.gflownet import TBGFlowNet  # We use a GFlowNet with the Trajectory Balance (TB) loss
from gfn.gym import HyperGrid  # We use the hyper grid environment
from gfn.modules import DiscretePolicyEstimator
from gfn.samplers import Sampler
from gfn.utils import NeuralNet  # NeuralNet is a simple multi-layer perceptron (MLP)

from actions.action_meta import DummyActionMeta
from envs.pre_order_env import PreOrderEnv

if __name__ == "__main__":

    # 1 - We define the environment

    action_meta = DummyActionMeta(num_features=1, has_constant=1)
    env = PreOrderEnv(max_token_length=4, action_meta=action_meta)

    # 2 - We define the needed modules (neural networks)

    module_PF = NeuralNet(
        input_dim=env.preprocessor.output_dim,
        output_dim=env.n_actions
    )  # Neural network for the forward policy, with as many outputs as there are actions
    module_PB = NeuralNet(
        input_dim=env.preprocessor.output_dim,
        output_dim=env.n_actions - 1,
        torso=module_PF.torso  # We share all the parameters of P_F and P_B, except for the last layer
    )

    # 3 - We define the estimators

    pf_estimator = DiscretePolicyEstimator(module_PF, env.n_actions, is_backward=False, preprocessor=env.preprocessor)
    pb_estimator = DiscretePolicyEstimator(module_PB, env.n_actions, is_backward=True, preprocessor=env.preprocessor)

    # 4 - We define the GFlowNet

    gfn = TBGFlowNet(init_logZ=0., pf=pf_estimator, pb=pb_estimator)  # We initialize logZ to 0

    # 5 - We define the sampler and the optimizer

    sampler = Sampler(estimator=pf_estimator)  # We use an on-policy sampler, based on the forward policy

    # Policy parameters have their own LR.
    non_logz_params = [v for k, v in dict(gfn.named_parameters()).items() if k != "logZ"]
    optimizer = torch.optim.Adam(non_logz_params, lr=1e-3)

    # Log Z gets dedicated learning rate (typically higher).
    logz_params = [dict(gfn.named_parameters())["logZ"]]
    optimizer.add_param_group({"params": logz_params, "lr": 1e-1})

    # 6 - We train the GFlowNet for 1000 iterations, with 16 trajectories per iteration

    for i in (pbar := tqdm(range(1000))):
        trajectories = sampler.sample_trajectories(env=env, n_trajectories=5)
        optimizer.zero_grad()
        loss = gfn.loss(env, trajectories)
        loss.backward()
        optimizer.step()
        if i % 25 == 0:
            pbar.set_postfix({"loss": loss.item()})