import tensorflow as tf
tf.logging.set_verbosity(tf.logging.FATAL)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward

from examples.utils import load_pilco

import gym
import numpy as np

import pickle
import utils


import warnings
warnings.filterwarnings("ignore")

import continuous_cartpole

# ENV_NAME_T = "CartPole-v1"
# ENV_NAME_T = "CartPole-v99"


from score_logger import ScoreLogger

# ENV_NAME = "Acrobot-v1"


GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01

np.random.seed(0)

ITERATION_TO_LOAD = 7

SUBS=3
bf = 10
maxiter=50
max_action=1.0
target = np.array([0., 0., 0., 0.]) # TODO review if correct
weights = np.diag([0.5, 0.1, 0.5, 0.25])
# m_init = np.reshape([-1.0, -1.0, 0., 0.0], (1,4))
# S_init = np.diag([0.01, 0.05, 0.01, 0.05])
T = 400
T_sim = T
# J = 4
# N = 8
# restarts = 2

state_dim = 4
control_dim = 1

N = 6


def loader(name):
    env = gym.make('continuous-cartpole-v99')
    env.seed(73)
    controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf,
                               max_action=max_action)
    R = ExponentialReward(state_dim=state_dim, t=target, W=weights)
    pilco = load_pilco('saved/pilco-continuous-cartpole-{:s}'.format(name), controller=controller, reward=R,
                       sparse=False)

    with open('10_pi_adj.pkl', 'rb') as inp2:
        pi_adjust = pickle.load(inp2)


    score_logger = ScoreLogger('PI ADJUST ANALYSISSSSSSS')
    run = 0
    while True:
        run += 1
        state = env.reset()
        # print(state)
        # input()
        step = 0
        while True:
            step += 1
            env.render()



            #TODO RUN PI ADJUST
            u_action =  utils.policy(env, pilco, state, False)
            state_copy = state

            a = np.ndarray.tolist(state_copy)
            a.extend(np.ndarray.tolist(u_action))
            action = pi_adjust.predict(np.array(a).reshape(1, -1))
            action = action[0]
            # TODO RUN PI ADJUST COMMENT THE NEXT LINE
            print(u_action,action)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state = state_next
            if terminal:
                print(
                    "Run: "  + ", score: " + str(step))
                score_logger.add_score(step, run)
                break

    env.env.close()


def true_loader(name):
    env = gym.make('continuous-cartpole-v99')
    env.seed(73)
    controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf,
                               max_action=max_action)
    R = ExponentialReward(state_dim=state_dim, t=target, W=weights)
    pilco = load_pilco('saved/pilco-continuous-cartpole-{:s}'.format(name), controller=controller, reward=R,
                       sparse=False)

    with open('9true_dyn_pi_adj.pkl', 'rb') as inp2:
        pi_adjust = pickle.load(inp2)

    # with open('10_pi_adj.pkl', 'rb') as inp2:
    #     good_pi = pickle.load(inp2)

    score_logger = ScoreLogger('PI ADJUST ANALYSISSSSSSS')
    run = 0
    while True:
        run += 1
        state = env.reset()
        # print(state)
        # input()
        step = 0
        while True:
            step += 1
            env.render()

            u_action = utils.policy(env, pilco, state, False)
            state_copy = state

            a = np.ndarray.tolist(state_copy)
            a.extend(np.ndarray.tolist(u_action))
            action = pi_adjust.predict(np.array(a).reshape(1, -1))[0]

            state_next, reward, terminal, info = env.step(action + u_action)
            reward = reward if not terminal else -reward
            state = state_next

            if terminal:
                print(
                    "Run: "  + ", score: " + str(step))
                score_logger.add_score(step, run)
                break

    env.env.close()


def see_progression(pilco_name='saved/pilco-continuous-cartpole-5', transfer_name='{:d}true_dyn_pi_adj.pkl', adjust=True):
    env = gym.make('continuous-cartpole-v99')
    env.seed(1)
    controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf, max_action=max_action)
    R = ExponentialReward(state_dim=state_dim, t=target, W=weights)
    pilco = load_pilco(pilco_name, controller=controller, reward=R, sparse=False)

    rewards = []

    for i in range(10):
        print('Running {:s}'.format(transfer_name.format(i)))
        if adjust:
            with open(transfer_name.format(i), 'rb') as inp2:
                pi_adjust = pickle.load(inp2)

        score_logger = ScoreLogger('Score for Model {:d}'.format(i))
        state = env.reset()
        step = 0
        while True:
            step += 1

            env.render()

            u_action = utils.policy(env, pilco, state, False)
            state_copy = state

            a = np.ndarray.tolist(state_copy)
            a.extend(np.ndarray.tolist(u_action))

            if adjust:
                pi_adjust_action = pi_adjust.predict(np.array(a).reshape(1, -1))[0]
            else:
                pi_adjust_action = 0 # ENABLE THIS TO SEE IT RUN WITHOUT THE ADJUSTMENT

            state_next, reward, terminal, info = env.step(u_action + pi_adjust_action)
            reward = reward if not terminal else -reward
            state = state_next

            if terminal:
                print('Run: {:d}, score: {:d}'.format(i, step))
                score_logger.add_score(step, i)
                break

        rewards.append(step)

    env.close()
    return rewards


def all_progressions():
    pilcos = ['initial'] + [str(i) for i in range(6)]
    all_rewards = {}
    for i, p in enumerate(pilcos):
        print('Getting progression for {:s}'.format(p))
        rewards = see_progression('saved/pilco-continuous-cartpole-{:s}'.format(p), 'transfer-save/pilco-{:s}-transfer-{{:d}}.pkl'.format(p))
        all_rewards[i] = rewards

    with open('rewards', 'wb') as output:  # Overwrites any existing file.
        pickle.dump(all_rewards, output, pickle.HIGHEST_PROTOCOL)

    plot_transfer_learning_curves()

    return all_rewards


def plot_transfer_learning_curves():
    with open('rewards', 'rb') as f:
        rewards = pickle.load(f)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, r in enumerate(rewards.keys()):
        x = [i for _ in rewards[r]]
        y = [i for i, _ in enumerate(rewards[r])]
        z = [r for r in rewards[r]]
        ax.plot3D(x, y, z)
        ax.scatter3D(x, y, z)
    ax.set_xlabel('PILCO Iteration')
    ax.set_ylabel('Transfer Learning Iteration')
    ax.set_zlabel('Reward')
    plt.title('Transfer Learning Curves per PILCO Iteration')

    plt.tight_layout()
    plt.savefig('rewards_plot')
    plt.close()


def plot_pilco_source_learning_curve():
    env = gym.make('continuous-cartpole-v0')
    env.seed(73)

    pilcos = ['initial'] + [str(i) for i in range(6)]


    rewards = []
    for i, p in enumerate(pilcos):
        controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf, max_action=max_action)
        R = ExponentialReward(state_dim=state_dim, t=target, W=weights)
        pilco = load_pilco('saved/pilco-continuous-cartpole-{:s}'.format(p), controller=controller, reward=R, sparse=False)

        score_logger = ScoreLogger('Score for Model {:d}'.format(i))
        state = env.reset()
        step = 0

        xs = []
        angles = []

        while True:
            xs.append(state[0])
            angles.append(state[2])
            step += 1

            env.render()

            u_action = utils.policy(env, pilco, state, False)
            state_copy = state

            a = np.ndarray.tolist(state_copy)
            a.extend(np.ndarray.tolist(u_action))


            state_next, reward, terminal, info = env.step(u_action)
            reward = reward if not terminal else -reward
            state = state_next

            if terminal:
                print('Run: {:d}, score: {:d}'.format(i, step))
                score_logger.add_score(step, i)
                break

        rewards.append(step)

        plt.plot(xs, angles)
        plt.savefig('pilco-{:d}_states_plot'.format(i), bbox_inches="tight")
        plt.close()

    env.close()

    plt.plot([i for i, _ in enumerate(pilcos)], rewards)
    plt.savefig('pilco_rewards_plot', bbox_inches="tight")
    plt.close()


    return rewards, xs, angles


    # with open('plot_pilco_source_learning_curve_dump', 'wb') as output:  # Overwrites any existing file.
    #     pickle.dump(rewards, output, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    # cartpole()
    # loader('5')
    # true_loader('5')

    # see_progression(pilco_name='saved/pilco-continuous-cartpole-0', transfer_name='transfer-save/pilco-0-transfer-{:d}.pkl', adjust=True)
    # see_progression(pilco_name='saved/pilco-continuous-cartpole-2', transfer_name='{:d}true_dyn_pi_adj.pkl', adjust=False)

    # rewards = all_progressions()
    # with open('rewards', 'wb') as output:  # Overwrites any existing file.
    #     pickle.dump(rewards, output, pickle.HIGHEST_PROTOCOL)

    plot_transfer_learning_curves()
    # plot_pilco_source_learning_curve()

# env.env.close()
