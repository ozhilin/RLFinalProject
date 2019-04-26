import tensorflow as tf
tf.logging.set_verbosity(tf.logging.FATAL)

import numpy as np
import gym
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward
import tensorflow as tf

from utils import rollout, load_pilco
import continuous_cartpole

np.random.seed(0)

ITERATION_TO_LOAD = 7

SUBS=3
bf = 10
maxiter=50
max_action=1.0
target = np.array([0., 0., 0., 0.])
weights = np.diag([0.5, 0.1, 0.5, 0.25])
T = 400
T_sim = T

state_dim = 4
control_dim = 1

N = 6


def load_and_run_model(env, name):
    controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf, max_action=max_action)
    R = ExponentialReward(state_dim=state_dim, t=target, W=weights)
    pilco = load_pilco('saved/pilco-continuous-cartpole-{:s}'.format(name), controller=controller, reward=R, sparse=False)

    print('Running {:s}'.format(name))
    return rollout(env, pilco, timesteps=T_sim, verbose=True, SUBS=SUBS)


if __name__ == '__main__':
    with tf.Session() as sess:
        env = gym.make('continuous-cartpole-v0')

        load_and_run_model(env, 'initial')
        for i in range(N):
            pass

        load_and_run_model(env, str(i))
