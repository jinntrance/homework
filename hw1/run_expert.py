#!/usr/bin/env python3

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import roboschool
import gym
import load_policy

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        def run_exp(func, returns, observations, actions, bc = False):
            for i in range(args.num_rollouts):
                print('iter', i)
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    action = func(obs[None,:])
                    if steps % 1000 == 0:
                        print(type(action))
                    observations.append(obs)
                    obs, r, done, _ = env.step(action)
                    if bc:
                        action = policy_fn(obs[None,:])
                    actions.append(action)
                    totalr += r
                    steps += 1
                    if args.render:
                        env.render()
                    # if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                    if steps >= max_steps:
                        break
                returns.append(totalr)

            print('returns', returns)
            print('mean return', np.mean(returns))
            print('std of return', np.std(returns))

        returns = []
        observations = []
        actions = []
        run_exp(policy_fn, returns, observations, actions)

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}
        actions_size = expert_data['actions'].shape[0]
        expert_data['actions'] = np.reshape(expert_data['actions'], (actions_size, expert_data['actions'].shape[1] * expert_data['actions'].shape[2]) )

        #
        print(expert_data['observations'].shape, expert_data['actions'].shape)
        inputs = tf_util.get_placeholder('inputs', tf.float32, [None, expert_data['observations'].shape[1]])
        labels = tf_util.get_placeholder('labels', tf.float32, [None, expert_data['actions'].shape[1]])

        # models
        name = args.envname
        d1 = tf_util.dense(inputs, 8, 'd1')
        d2 = tf_util.dropout(d1, 0.8)
        d3 = tf_util.wndense(d2, 8 , 'd2')
        pred = tf_util.densenobias(d3, expert_data['actions'].shape[1], 'output')

        print(type(actions), type(pred))
        loss_func = tf.losses.mean_squared_error(actions, pred)
        loss = tf.reduce_mean(loss_func)
        optimizer = tf.train.RMSPropOptimizer(0.1).minimize(loss)

        # evaluations
        tf_util.initialize()

        # grid search parameters
        for i in range(args.num_rollouts):
            ls = 0
            batch_size = int(actions_size / 20)
            batch_num = int(actions_size / batch_size)
            for j in range(batch_num):
                start = batch_num * j
                end = start + batch_size
                op_eval, ls_current = tf_util.eval([optimizer, loss], {inputs: expert_data['observations'][start:end],labels:
            expert_data['actions'][start:end] })
                # print('batch ', j, ls_current)
                ls += ls_current
            print('iter ', i, ls/batch_num)
        def model_eval(obs):
            p = tf_util.eval([pred], {inputs: obs})
            return np.array(p)

        run_exp(model_eval, [], [], [])

        # run_exp(model_eval, [], observations, actions, True)


if __name__ == '__main__':
    main()
