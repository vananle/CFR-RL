from __future__ import print_function

import numpy as np
import tensorflow as tf
from absl import app
from absl import flags

from config import get_config
from env import Environment
from game import CFRRL_Game
from model import Network

FLAGS = flags.FLAGS
flags.DEFINE_string('ckpt', '', 'apply a specific checkpoint')
flags.DEFINE_boolean('eval_delay', False, 'evaluate delay or not')


def count_rc_cfr_rl(solutions, crit_pairs, lp_links):
    rc = []
    for i in range(len(solutions) - 1):
        count = 0
        solution_0 = solutions[i][0]
        solution_1 = solutions[i + 1][0]
        pairs_0 = crit_pairs[i]
        pairs_1 = crit_pairs[i + 1]

        intersec_flows = np.intersect1d(pairs_0, pairs_1)

        # counting the number of flows added to the list
        new_cflows = np.setdiff1d(pairs_1, intersec_flows)
        count += new_cflows.shape[0]

        # counting the number of flows that have path changed
        for flow_idx in intersec_flows:
            for e in lp_links:
                if solution_1[flow_idx, e[0], e[1]] - solution_0[flow_idx, e[0], e[1]] != 0.0:
                    count += 1
                    break

        rc.append(count)
    return np.asarray(rc)


def sim(config, network, game):
    mlus = []
    solutions = []
    crit_pairs = []
    for tm_idx in game.tm_indexes:
        if tm_idx % 10 == 0 and tm_idx != 0:
            print('t       opt_mlu             norm_mlu              mlu          norm_crit_mlu         crit_mlu     '
                  '       norm_topk_mlu          topk_mlu         norm_ecmp_mlu           ecmp_mlu')
        state = game.get_state(tm_idx)
        if config.method == 'actor_critic':
            policy = network.actor_predict(np.expand_dims(state, 0)).numpy()[0]
        elif config.method == 'pure_policy':
            policy = network.policy_predict(np.expand_dims(state, 0)).numpy()[0]
        actions = policy.argsort()[-game.max_moves:]

        u, solution, crit_pair = game.evaluate(tm_idx, actions, eval_delay=FLAGS.eval_delay)
        mlus.append(u)
        solutions.append(solution)
        crit_pairs.append(crit_pair)

    mlus = np.asarray(mlus)
    rc = count_rc_cfr_rl(solutions, crit_pairs=crit_pairs, lp_links=game.lp_links)
    print('----------------------------------- OVERALL RESULTS -------------------------------------------------------')
    print('MLU   Critical MLU   Topk MLU      Optimal MLU\n', np.mean(mlus, axis=0))
    print('RC CFR_RL: Total: {}  -  Avg: {}'.format(np.sum(rc), np.mean(rc)))


def main(_):
    # Using cpu for testing
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.get_logger().setLevel('INFO')

    config = get_config(FLAGS) or FLAGS
    env = Environment(config, is_training=False)
    game = CFRRL_Game(config, env)
    network = Network(config, game.state_dims, game.action_dim, game.max_moves)

    step = network.restore_ckpt(FLAGS.ckpt)
    if config.method == 'actor_critic':
        learning_rate = network.lr_schedule(network.actor_optimizer.iterations.numpy()).numpy()
    elif config.method == 'pure_policy':
        learning_rate = network.lr_schedule(network.optimizer.iterations.numpy()).numpy()
    print('\nstep %d, learning rate: %f\n' % (step, learning_rate))

    sim(config, network, game)


if __name__ == '__main__':
    app.run(main)
