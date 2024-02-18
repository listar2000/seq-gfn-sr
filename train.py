import torch
from gfn.utils.modules import DiscreteUniform
from tqdm import tqdm

from gfn.gflownet import TBGFlowNet  # We use a GFlowNet with the Trajectory Balance (TB) loss
from gfn.gym import HyperGrid  # We use the hyper grid environment
from gfn.modules import DiscretePolicyEstimator
from gfn.samplers import Sampler
from gfn.utils import NeuralNet  # NeuralNet is a simple multi-layer perceptron (MLP)

from actions.action_meta import DummyActionMeta
from envs.pre_order_env import PreOrderEnv

if __name__ == "__main__":
    # -1 - some hyper-params and arguments
    USE_DUMMY = False
    PLOTTING = True

    # 0 - simulate some data
    X = 2 * torch.rand(20, 2)
    y = torch.square(X[:, 0]) + X[:, 1]

    # 1 - We define the environment

    action_meta = DummyActionMeta(num_features=2, has_constant=False)
    env = PreOrderEnv(max_token_length=4, action_meta=action_meta, reward_eps=1, X=X, y=y)

    # 2 - We define the needed modules (neural networks)

    module_PF = NeuralNet(
        input_dim=env.preprocessor.output_dim,
        output_dim=env.n_actions
    )  # Neural network for the forward policy, with as many outputs as there are actions

    if USE_DUMMY:
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

    # 6 - We train the GFlowNet for 1000 iterations, with 16 trajectories per iteration
    logz_values = []
    loss_values = []
    reward_values = []
    eval_iters = []

    for i in (pbar := tqdm(range(5000))):
        trajectories = sampler.sample_trajectories(env=env, n_trajectories=16)
        optimizer.zero_grad()
        loss = gfn.loss(env, trajectories)
        loss.backward()

        loss_values.append(loss.item())
        logz_values.append(gfn.logZ.item())

        optimizer.step()
        if i % 50 == 0:
            with torch.no_grad():
                # measure rewards
                eval_iters.append(int(i))
                trajectories = sampler.sample_trajectories(env=env, n_trajectories=100)
                final_states = trajectories.last_states
                log_rewards = env.log_reward(final_states)
                reward_values.append(log_rewards.mean().item())

            pbar.set_postfix({"loss": loss.item()})

    if PLOTTING:
        import matplotlib.pyplot as plt
        # Plotting
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        # Loss vs. Iteration
        axs[0].plot(loss_values, label='Loss')
        axs[0].set_xlabel('Iteration')
        axs[0].set_ylabel('Loss')
        axs[0].set_title('Loss vs. Iteration')
        axs[0].legend()

        # logZ vs. Iteration or reward vs. Iteration
        # axs[1].plot(logz_values, label='logZ', color='orange')
        # axs[1].set_xlabel('Iteration')
        # axs[1].set_ylabel('logZ')
        # axs[1].set_title('logZ vs. Iteration')
        # axs[1].legend()
        axs[1].plot(eval_iters, reward_values, label='log rewards', color='orange')
        axs[1].set_xlabel('Iteration')
        axs[1].set_ylabel('log rewards')
        axs[1].set_title('Rewards with eps = 1')
        axs[1].legend()

        plt.tight_layout()
        plt.show()
