import tensorflow as tf
tf.logging.set_verbosity(tf.logging.FATAL)

import gym
import matplotlib.pyplot as plt

from cartpole_from_file import load_and_run_model

import matplotlib.pyplot as plt
import matplotlib
params = {'axes.titlesize': 9}
matplotlib.rcParams.update(params)
from mpl_toolkits.mplot3d import Axes3D

from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward

from utils import load_pilco, policy

import gym
import numpy as np

import pickle
import utils

import warnings
warnings.filterwarnings("ignore")

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


def run_transfer_model_and_plot_pos(env_name, pilco_name, fig_file_name, fig_title, transfer_name=None, save=True):
    env = gym.make(env_name)
    env.seed(1)
    controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf, max_action=max_action)
    R = ExponentialReward(state_dim=state_dim, t=target, W=weights)
    pilco = load_pilco('saved/pilco-continuous-cartpole-{:s}'.format(pilco_name), controller=controller, reward=R, sparse=False)

    if transfer_name is not None:
        with open(transfer_name, 'rb') as inp2:
            pi_adjust = pickle.load(inp2)

    xs = []
    angles = []

    state = env.reset()
    for _ in range(1000):
        xs.append(state[0])
        angles.append(state[2])
        env.render()

        u_action = policy(env, pilco, state, False)
        state_copy = state

        a = np.ndarray.tolist(state_copy)
        a.extend(np.ndarray.tolist(u_action))

        if transfer_name is not None:
            pi_adjust_action = pi_adjust.predict(np.array(a).reshape(1, -1))[0]
        else:
            pi_adjust_action = 0

        state_next, reward, terminal, info = env.step(u_action + pi_adjust_action)
        state = state_next

        if terminal:
            break

    env.close()

    xs = np.array(xs)
    angles = np.array(angles)

    plt.plot(xs, angles)
    plt.quiver(xs[:-1], angles[:-1], xs[1:] - xs[:-1], angles[1:] - angles[:-1], scale_units='xy', angles='xy', scale=1, color='blue', width=1e-2)
    plt.xlabel('position')
    plt.ylabel('angle')
    plt.title(fig_title)
    plt.xlim(-0.2, 0.2)
    plt.ylim(-0.2, 0.2)

    if save:
        plt.savefig(fig_file_name, bbox_inches="tight")
        plt.close()


def run_and_plot_pos(env_name, pilco_version, fig_file_name, fig_title):
    env = gym.make(env_name)
    env.seed(1)
    vals = load_and_run_model(env, pilco_version)[0]
    env.close()

    xs = vals[:, 0]
    angles = vals[:, 2]

    ax = plt.plot(xs, angles, label=fig_title)
    # plt.quiver(xs[:-1], angles[:-1], xs[1:] - xs[:-1], angles[1:] - angles[:-1], scale_units='xy', angles='xy', scale=1, color='blue')
    plt.xlabel('position')
    plt.ylabel('angle')
    # plt.title(fig_title)
    # ax.set_title(fig_title, fontsize=12)
    plt.xlim(-0.05, 0.2)
    plt.ylim(-0.05, 0.05)
    # plt.savefig(fig_file_name, bbox_inches="tight")
    # plt.close()


if __name__ == '__main__':

    fig = plt.figure(1)

    run_and_plot_pos('continuous-cartpole-v0', '0', 'oleg–plot-pilco-0-v0', 'Pilco Iteration 1')
    # run_and_plot_pos('continuous-cartpole-v0', '1', 'oleg–plot-pilco-1-v0', 'Pilco Iteration 2')
    run_and_plot_pos('continuous-cartpole-v0', '2', 'oleg–plot-pilco-2-v0', 'Pilco Iteration 3')
    # run_and_plot_pos('continuous-cartpole-v0', '3', 'oleg–plot-pilco-3-v0', 'Pilco Iteration 4')
    run_and_plot_pos('continuous-cartpole-v0', '4', 'oleg–plot-pilco-4-v0', 'Pilco Iteration 5')
    # run_and_plot_pos('continuous-cartpole-v0', '5', 'oleg–plot-pilco-5-v0', 'Pilco Iteration 6')
    run_and_plot_pos('continuous-cartpole-v0', 'initial', 'oleg–plot-pilco-0-v0', 'Random Controller')

    plt.legend()
    plt.tight_layout()
    plt.savefig('double-paths', bbox_inches="tight")

    plt.close()

    # run_and_plot_pos('continuous-cartpole-v99', '0', 'oleg–plot-pilco-0-v99', 'Pilco Iteration 1, Target Environment (no transfer)')
    # run_and_plot_pos('continuous-cartpole-v99', '5', 'oleg–plot-pilco-5-v99', 'Pilco Iteration 5, Target Environment (no transfer)')
    #
    # exit()

    # run_transfer_model_and_plot_pos('continuous-cartpole-v99', '0', 'oleg-plot-transfer-pilco-0-transfer-2',
    #                                 'Pilco Iteration 1, Target Environment Transfer Iteration 2',
    #                                 transfer_name='transfer-save/pilco-0-transfer-2.pkl', save=False)
    #
    exit()

    fig = plt.figure(1, (16, 4))

    plt.subplot(241)
    run_transfer_model_and_plot_pos('continuous-cartpole-v99', '0', 'oleg-plot-transfer-pilco-0-transfer-0', 'Pilco Iteration 1, Target Environment (no transfer)', transfer_name='transfer-save/pilco-0-transfer-0.pkl', save=False)
    plt.subplot(242)
    run_transfer_model_and_plot_pos('continuous-cartpole-v99', '0', 'oleg-plot-transfer-pilco-0-transfer-1', 'Pilco Iteration 1, Target Environment Transfer Iteration 2', transfer_name='transfer-save/pilco-0-transfer-1.pkl', save=False)
    plt.subplot(243)
    run_transfer_model_and_plot_pos('continuous-cartpole-v99', '0', 'oleg-plot-transfer-pilco-0-transfer-2', 'Pilco Iteration 1, Target Environment Transfer Iteration 3', transfer_name='transfer-save/pilco-0-transfer-2.pkl', save=False)
    plt.subplot(244)
    run_transfer_model_and_plot_pos('continuous-cartpole-v99', '0', 'oleg-plot-transfer-pilco-0-transfer-3', 'Pilco Iteration 1, Target Environment Transfer Iteration 4', transfer_name='transfer-save/pilco-0-transfer-3.pkl', save=False)

    plt.subplot(245)
    run_transfer_model_and_plot_pos('continuous-cartpole-v99', '5', 'oleg-plot-transfer-pilco-5-transfer-0', 'Pilco Iteration 5, Target Environment (no transfer)', transfer_name='transfer-save/pilco-5-transfer-0.pkl', save=False)
    plt.subplot(246)
    run_transfer_model_and_plot_pos('continuous-cartpole-v99', '5', 'oleg-plot-transfer-pilco-5-transfer-1', 'Pilco Iteration 5, Target Environment Transfer Iteration 2', transfer_name='transfer-save/pilco-5-transfer-1.pkl', save=False)
    plt.subplot(247)
    run_transfer_model_and_plot_pos('continuous-cartpole-v99', '5', 'oleg-plot-transfer-pilco-5-transfer-2', 'Pilco Iteration 5, Target Environment Transfer Iteration 3', transfer_name='transfer-save/pilco-5-transfer-2.pkl', save=False)
    plt.subplot(248)
    run_transfer_model_and_plot_pos('continuous-cartpole-v99', '5', 'oleg-plot-transfer-pilco-5-transfer-3', 'Pilco Iteration 5, Target Environment Transfer Iteration 4', transfer_name='transfer-save/pilco-5-transfer-3.pkl', save=False)

    # plt.subplots_adjust(wspace=5, hspace=0.6)
    plt.tight_layout()
    # fig.tight_layout()
    plt.savefig('pilco-transfer-big')
    plt.close()
