from pilco.controllers import RbfController
from pilco.rewards import ExponentialReward
import utils

from utils import rollout, load_pilco
from score_logger import ScoreLogger

import gym
import numpy as np
import pickle

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

import warnings
warnings.filterwarnings("ignore")

import continuous_cartpole

np.random.seed(0)

ITERATION_TO_LOAD = 7

SUBS = 3
bf = 10
maxiter = 50
max_action = 1.0
target = np.array([0., 0., 0., 0.]) # TODO review if correct
weights = np.diag([0.5, 0.1, 0.5, 0.25])

T = 400
T_sim = T

state_dim = 4
control_dim = 1

N = 6
SIGMA = 1e-10


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_and_run_model(env, name):
    controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf, max_action=max_action)
    R = ExponentialReward(state_dim=state_dim, t=target, W=weights)
    pilco = load_pilco('saved/pilco-continuous-cartpole-{:s}'.format(name), controller=controller, reward=R, sparse=False)

    print('Running {:s}'.format(name))
    rollout(env, pilco, timesteps=T_sim, verbose=False, SUBS=SUBS)


def loader(name):
    env = gym.make('continuous-cartpole-v99')
    env.seed(73)

    controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf, max_action=max_action)
    R = ExponentialReward(state_dim=state_dim, t=target, W=weights)
    pilco = load_pilco('saved/pilco-continuous-cartpole-{:s}'.format(name), controller=controller, reward=R, sparse=False)

    score_logger = ScoreLogger('PI ADJUST ANALYSISSSSSSS')
    # observation_space = env.observation_space.shape[0]

    run = 0
    while True:
        run += 1
        state = env.reset()
        step = 0
        while True:
            step += 1
            env.render()

            #TODO RUN PI ADJUST
            action = utils.policy(env, pilco, state, False)
            # TODO RUN PI ADJUST COMMENT THE NEXT LINE

            state_next, reward, terminal, info = env.step(action)
            # reward = reward if not terminal else -reward
            state = state_next
            if terminal:
                print("Run: " + str(run) + ", score: " + str(step))
                score_logger.add_score(step, run)
                break

    env.env.close()


def noiser(D,ind,mu=0,sigma=0.0000001):
    y = D
    noise = np.random.normal(mu, sigma, y.shape)
    y[:, ind] = y[:, ind] + noise[:, ind]

    return y


def sampler(pi, env, samples_n, trials=1, render=True):
    D = None

    for t in range(trials):
        state = env.reset()
        for i in range(samples_n):
            if render: env.render()

            action = utils.policy(env, pi, state, False) + np.random.normal(0, SIGMA)
            state_next, reward, terminal, info = env.step(action)

            if D is not None:
                D = np.vstack((D, [state, action, state_next]))
            elif D is None:
                D = np.array([state, action, state_next]).reshape(1, -1)

            state = state_next

            if terminal:
                break

    return D


def inverse_dyn(D_T):
    kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(0.0, 10.0)) \
             * RBF(length_scale=0.5, length_scale_bounds=(0.0, 10.0)) \
             + RBF(length_scale=2.0, length_scale_bounds=(0.0, 10.0))

    x = np.array([np.ndarray.tolist(xi) for xi in D_T[:, 0]])
    x2 = np.array([np.ndarray.tolist(xi) for xi in D_T[:, 2]])

    x3 = np.hstack((x, x2))

    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0) \
        .fit(x3, np.array(D_T[:, 1]).reshape(-1, 1))

    return gpr


def L3(D_adj):
    kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(0.0, 10.0)) \
             * RBF(length_scale=0.5, length_scale_bounds=(0.0, 10.0)) \
             + RBF(length_scale=2.0, length_scale_bounds=(0.0, 10.0))

    x = np.array([xi[0] for xi in D_adj])
    x2 = np.array([xi[1] for xi in D_adj])
    y = np.array([xi[2] for xi in D_adj])

    x3 = np.concatenate((x, x2.reshape(-1, 1)), axis=1)

    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0) \
        .fit(x3, y.reshape(-1, 1))

    return gpr


def sampler_adj(pi_adj, pi_s, env, samples_n, render=True):
    D = list()
    state = env.reset()

    for i in range(samples_n):
        if render: env.render()

        u_action = utils.policy(env, pi_s, state, False)
        state_copy = state

        a = np.ndarray.tolist(state_copy)
        a.extend(np.ndarray.tolist(u_action))
        pi_adj_action = pi_adj.predict(np.array(a).reshape(1, -1))[0]

        action = u_action + pi_adj_action + np.random.normal(0, SIGMA)

        state_next, reward, terminal, info = env.step(action)

        D.append([state, action, state_next])
        state = state_next

        if terminal:
            state = env.reset()
            break

    y = np.array([np.array(xi) for xi in D])

    return y


def piadjust(NT, name):
    controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf, max_action=max_action)
    R = ExponentialReward(state_dim=state_dim, t=target, W=weights)
    pilco = load_pilco('saved/pilco-continuous-cartpole-{:s}'.format(name), controller=controller, reward=R, sparse=False)

    env_S = gym.make('continuous-cartpole-v0')
    env_S.seed(73)

    env_T = gym.make('continuous-cartpole-v99')
    env_T.seed(73)

    D_S = sampler(pilco, env_S, samples_n=30, trials=50)
    # print('D_S sampling done')

    D_T = None
    pi_adj = pilco

    for i in range(NT):
        print('{:d}/{:d}'.format(i + 1, NT))

        D_adj = []

        if i == 0:
            D_i_T = sampler(pilco, env_T, samples_n=30)
        elif i != 0:
            D_i_T = sampler_adj(pi_adj, pilco, env_T, 30)

        if D_T is not None:
            D_T = np.concatenate((D_i_T, D_T))
        elif D_T is None:
            D_T = D_i_T

        # print('Going for inverse dyn')
        gpr = inverse_dyn(D_T)
        # print('inverse dyn done')

        for samp in D_S:
            x_s = list(samp[0])
            x_s1 = list(samp[2])
            u_t_S = samp[1]

            a = np.array(x_s + x_s1).reshape(1, 8)
            u_t_T = gpr.predict(a, return_std=False)

            D_adj.append((x_s, u_t_S, u_t_T - u_t_S))

        # print('Going for L3')
        pi_adj = L3(D_adj)
        # print('L3 Done')

        # i = i + 1
        # if (i % 1 == 0):
        save_object(pi_adj, 'transfer-save/pilco-{:s}-transfer-{:d}.pkl'.format(name, i))

    env_S.env.close()
    env_T.env.close()

    return pi_adj


def train_all_pilcos():
    pilcos = ['initial'] + [str(i) for i in range(6)]
    for p in pilcos:
        print('Training pilco version: {:s}'.format(p))
        piadjust(10, p)

if __name__ == "__main__":
    # cartpole()
    # loader('5')
    # piadjust(10, '5')
    train_all_pilcos()

# env.env.close()
