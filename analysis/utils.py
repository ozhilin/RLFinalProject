import numpy as np
from gpflow import autoflow
from gpflow import settings
from pilco.models.pilco import PILCO

from pilco.controllers import RbfController

float_type = settings.dtypes.float_type


def rollout(env, pilco, timesteps, verbose=True, random=False, SUBS=1, render=True):
    X = []; Y = []
    x = env.reset()
    for timestep in range(timesteps):
        if render: env.render()

        u = policy(env, pilco, x, random)

        for i in range(SUBS):
            x_new, _, done, _ = env.step(u)
            if done: break
            if render: env.render()

        if verbose:
            print("Action: ", u)
            print("State : ", x_new)
        X.append(np.hstack((x, u)))
        Y.append(x_new - x)
        x = x_new
        if done: break
    return np.stack(X), np.stack(Y)


def policy(env, pilco, x, random):
    if random:
        return env.action_space.sample()
    else:
        return pilco.compute_action(x[None, :])[0, :]


@autoflow((float_type,[None, None]), (float_type,[None, None]))
def predict_one_step_wrapper(mgpr, m, s):
    return mgpr.predict_on_noisy_inputs(m, s)


@autoflow((float_type,[None, None]), (float_type,[None, None]), (np.int32, []))
def predict_trajectory_wrapper(pilco, m, s, horizon):
    return pilco.predict(m, s, horizon)


@autoflow((float_type,[None, None]), (float_type,[None, None]))
def compute_action_wrapper(pilco, m, s):
    return pilco.controller.compute_action(m, s)


@autoflow((float_type, [None, None]), (float_type, [None, None]))
def reward_wrapper(reward, m, s):
    return reward.compute_reward(m, s)

def save_pilco(path, X, Y, pilco, sparse=False):
    np.savetxt(path + 'X.csv', X, delimiter=',')
    np.savetxt(path + 'Y.csv', Y, delimiter=',')
    if sparse:
        with open(path+ 'n_ind.txt', 'w') as f:
            f.write('%d' % pilco.mgpr.num_induced_points)
            f.close()

    np.save(path + 'pilco_values.npy', pilco.read_values())
    for i,m in enumerate(pilco.mgpr.models):
        np.save(path + "model_" + str(i) + ".npy", m.read_values())

def load_pilco(path, controller=None, reward=None, sparse=False):
    X = np.loadtxt(path + 'X.csv', delimiter=',')
    Y = np.loadtxt(path + 'Y.csv', delimiter=',')

    if not sparse:
        pilco = PILCO(X, Y, controller=controller, reward=reward)
    else:
        with open(path+ 'n_ind.txt', 'r') as f:
            n_ind = int(f.readline())
            f.close()
        pilco = PILCO(X, Y, controller=controller, reward=reward, num_induced_points=n_ind)

    params = np.load(path + "pilco_values.npy").item()
    pilco.assign(params)

    for i,m in enumerate(pilco.mgpr.models):
        values = np.load(path + "model_" + str(i) + ".npy").item()
        m.assign(values)

    return pilco