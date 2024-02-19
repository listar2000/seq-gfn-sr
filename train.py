import argparse
import json

import torch
from gfn.utils.modules import DiscreteUniform
from tqdm import tqdm

from gfn.gflownet import TBGFlowNet, SubTBGFlowNet  # We use a GFlowNet with the Trajectory Balance (TB) loss
from gfn.modules import DiscretePolicyEstimator
from gfn.samplers import Sampler
from gfn.utils import NeuralNet  # NeuralNet is a simple multi-layer perceptron (MLP)

from actions.action_meta import DummyActionMeta
from envs.pre_order_env import PreOrderEnv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model using a configuration file.")
    parser.add_argument("--config", type=str, help="Path to the configuration file.", default="config.json")
    args = parser.parse_args()

    # Load the configuration file
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)

    # 0 - simulate some data
    X = 2 * torch.rand(20, 2)
    y = (torch.sin(X[:, 0]) + X[:, 1]) * X[:, 0]

    # 1 - We define the environment

    # Define the environment
    env_config = config["env"]
    action_meta = DummyActionMeta(num_features=env_config['num_features'],
                                  has_constant=env_config['has_constant'])
    env = PreOrderEnv(max_token_length=env_config['max_token_length'], action_meta=action_meta,
                      reward_eps=env_config['reward_eps'], X=X, y=y)

    # 2 - We define the needed modules (neural networks)

    module_PF = NeuralNet(
        input_dim=env.preprocessor.output_dim,
        output_dim=env.n_actions
    )  # Neural network for the forward policy, with as many outputs as there are actions

    if config["use_dummy"]:
        # we simply use a non-trainable uniform output, which will be masked out anyway
        module_PB = DiscreteUniform(output_dim=env.n_actions - 1)
    else:
        module_PB = NeuralNet(
            input_dim=env.preprocessor.output_dim,
            output_dim=env.n_actions - 1,
            torso=module_PF.torso  # We share all the parameters of P_F and P_B, except for the last layer
        )

    # 3 - We define the estimators

    pf_estimator = DiscretePolicyEstimator(module_PF, env.n_actions, is_backward=False, preprocessor=env.preprocessor)
    pb_estimator = DiscretePolicyEstimator(module_PB, env.n_actions, is_backward=True, preprocessor=env.preprocessor)

    # 4 - We define the GFlowNet

    gfn = TBGFlowNet(init_logZ=0., pf=pf_estimator, pb=pb_estimator, on_policy=True)  # We initialize logZ to 0

    # 5 - We define the sampler and the optimizer

    sampler = Sampler(estimator=pf_estimator)  # We use an on-policy sampler, based on the forward policy

    # Policy parameters have their own LR.
    non_logz_params = [v for k, v in dict(gfn.named_parameters()).items() if k != "logZ"]
    optimizer = torch.optim.Adam(non_logz_params, lr=1e-3)

    # Log Z gets dedicated learning rate (typically higher).
    logz_params = [dict(gfn.named_parameters())["logZ"]]
    optimizer.add_param_group({"params": logz_params, "lr": 1e-1})

    train_cfg = config["training"]
    eval_metric = train_cfg["eval_metric"]
    assert eval_metric in ["mse", "reward"]

    losses, metrics = [], []
    for i in (pbar := tqdm(range(train_cfg["iterations"]))):
        trajectories = sampler.sample_trajectories(env=env, n_trajectories=train_cfg["n_traj"])

        optimizer.zero_grad()
        loss = gfn.loss(env, trajectories)
        loss.backward()

        optimizer.step()
        if i % train_cfg["eval_interval"] == 0:
            with torch.no_grad():
                # measure rewards
                trajectories = sampler.sample_trajectories(env=env, n_trajectories=train_cfg["eval_traj"])
                final_states = trajectories.last_states

                if eval_metric == 'mse':
                    mses = env._evaluate_final_states(final_states, metric="mse")
                    metric = mses[torch.isfinite(mses)].mean()
                else:
                    log_rewards = env.log_reward(final_states)
                    metric = torch.exp(log_rewards).mean()

            pbar.set_postfix({"loss": loss.item(), eval_metric: metric.item()})
            losses.append(loss.item())
            metrics.append(metric.item())

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Loss vs. Iteration
    axs[0].plot(losses, label='Loss')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Loss vs. Iteration')
    axs[0].legend()
    axs[1].plot(metrics, label=eval_metric, color='orange')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel(eval_metric)
    axs[1].set_title('Metric vs. Iteration')
    axs[1].legend()

    plt.tight_layout()
    plt.show()
