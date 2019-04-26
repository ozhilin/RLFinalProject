import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dense, Dropout
import pickle

import  sklearn

import warnings
warnings.filterwarnings("ignore")

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF


from test import *

ENV_NAME = "CartPole-v1"
ENV_NAMET = "CartPole-v99"

from score_logger import ScoreLogger

from test import *

# ENV_NAME = "Acrobot-v1"


GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995


def noiser(D,ind,mu=0,sigma=0.0000001):
    y = D
    noise = np.random.normal(mu,sigma,y.shape)
    # print(noise)
    y[:,ind]= y[:,ind] + noise[:,ind]

    return(y)


def sampler(pi, env, samples_n):
    D = list()
    observation_space = env.observation_space.shape[0]

    state = env.reset()
    state = np.reshape(state, [1, observation_space])


    for i in range(samples_n):

        action = pi.act(state)

        state_next, reward, terminal, info = env.step(action)
        state_next = np.reshape(state_next, [1, observation_space])

        D.append([state,action,state_next])
        state = state_next

        if terminal:
            state = env.reset()
            state = np.reshape(state, [1, observation_space])


    y = np.array([np.array(xi) for xi in D])

    return(y)


def inverse_dyn(D_T):
    kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(0.0, 10.0)) * RBF(length_scale=0.5,
                                                                                         length_scale_bounds=(0.0, 10.0))\
             + RBF(length_scale=2.0, length_scale_bounds=(0.0, 10.0))


    # print(np.concatenate(( D_T[:,0],D_T[:,2]),axis=0))
    # print(D_T[:,1])

    x =  np.array([ np.ndarray.tolist(xi)[0] for xi in D_T[:, 0] ])
    x2 = np.array([ np.ndarray.tolist(xi)[0] for xi in D_T[:, 2] ])
    x3 = np.concatenate((x,x2),axis=1)
    # print(x3.shape)


    gpr = GaussianProcessRegressor(kernel=kernel,random_state = 0).\
        fit(x3
            , D_T[:,1])
    # print(x3)
    return(gpr)

def L3(D_adj):
    kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(0.0, 10.0)) * RBF(length_scale=0.5,
                                                                                         length_scale_bounds=(0.0, 10.0))\
             + RBF(length_scale=2.0, length_scale_bounds=(0.0, 10.0))


    # print(np.concatenate(( D_T[:,0],D_T[:,2]),axis=0))
    # print(D_T[:,1])

    x =  np.array([ xi[0] for xi in D_adj ])
    x2 = np.array([ xi[1] for xi in D_adj ])
    y = np.array([ xi[2] for xi in D_adj ])

    # print(x.shape,'\n\n')
    # print(x2.reshape(-1,1).shape)

    x3 = np.concatenate((x,x2.reshape(-1,1)),axis=1)
    # print(x3.shape)
    # print(x3)
    # input()
    print('L3 gpr')
    gpr = GaussianProcessRegressor(kernel=kernel,random_state = 0).\
        fit(x3
            , y.reshape(-1,1))
    # print(x3)
    return(gpr)

def sampler_adj(pi_adj, pi_s, env, samples_n):
    D = list()
    observation_space = env.observation_space.shape[0]

    state = env.reset()
    state = np.reshape(state, [1, observation_space])


    for i in range(samples_n):

        u_action = pi_s.act(state)

        state_copy = state
        fuck = np.append(state_copy[0], u_action)
        action = pi_adj.predict(np.array(fuck).reshape(1,-1))
        action=action[0][0]
        # action=int(round(action))
        # if action!=u_action:
        #     print('DIIFFFFF')
        # print(action, u_action)
        # input()

        state_next, reward, terminal, info = env.step(action)
        state_next = np.reshape(state_next, [1, observation_space])

        D.append([state,action,state_next])
        state = state_next

        if terminal:
            state = env.reset()
            state = np.reshape(state, [1, observation_space])


    y = np.array([np.array(xi) for xi in D])

    return(y)


def piadjust(NT):
    with open('GOODv1.pkl ', 'rb') as inp:
        dqn_solver = pickle.load(inp)

    env_S = gym.make(ENV_NAME)
    env_S.seed(73)
    score_logger_S = ScoreLogger(ENV_NAME)
    observation_space_S = env_S.observation_space.shape[0]


    env_T = gym.make(ENV_NAMET)
    env_T.seed(73)
    score_logger_T = ScoreLogger(ENV_NAMET)
    observation_space_T = env_T.observation_space.shape[0]


    #TODO IMplement Pi adjust
    D_S = sampler(dqn_solver,env_S,1000)
    D_S = noiser(D_S, [0,2])
    print('D_S sampling done')

    D_T = None
    i = 0
    pi_adj = dqn_solver

    while i< NT:
        D_adj = []

        if i ==0:
            D_i_T = sampler(dqn_solver, env_T,1000)

        elif i!= 0:
            D_i_T = sampler_adj(pi_adj,dqn_solver, env_T, 1000)

        if D_T is not None:
            # print(D_i_T.shape, D_T.shape)
            D_T = np.concatenate((D_i_T,D_T))
        elif  D_T is  None:
            D_T = D_i_T
        print('Goin for inverse dyn')
        gpr = inverse_dyn(D_T)
        print('inverse dyn done')

        for samp in D_S:


            x_s = np.ndarray.tolist(samp[0])[0]
            x_s1 = np.ndarray.tolist(samp[2])[0]
            u_t_S = samp[1]
            # print(u_t_S)

            a=np.ndarray.tolist(samp[0])[0]
            a.extend( np.ndarray.tolist(samp[2])[0])
            # print( np.array(a).reshape(1, 8)  )

            u_t_T = gpr.predict( np.array(a).reshape(1, 8), return_std=False)

            if u_t_T > 0:
                u_t_T =1

            elif u_t_T <0:
                u_t_T = 0
            # print('\n\n', dqn_solver.act(  np.array(a[0:4]).reshape([1,4] ) ))
            # print( np.array(a[0:4]).reshape([1,4] )
            # print(i, '    ', D_adj)
            D_adj.append((x_s, u_t_S, u_t_T))


        # print(i, '    ',x_s, u_t_S, u_t_T)
        print('Goin for L3')
        pi_adj = L3(D_adj)
        print('L3 Done')
        # x_s.append(u_t_S)
        # print(pi_adj.predict(np.array(x_s).reshape(1,-1)))
        print(i)
        i = i + 1
        if (i%1==0):
            save_object(pi_adj, str(i)+'_pi_adj.pkl')



    env_S.env.close()
    env_T.env.close()

    return(pi_adj)

if __name__ == "__main__":
    pi_adj = piadjust(10)
    # save_object(pi_adj,  'pi_adj.pkl')

