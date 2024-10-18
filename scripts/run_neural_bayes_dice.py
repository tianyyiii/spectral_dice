# Copyright 2020 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Script for running BayesDICE with function approximation.

Make sure to generate the datasets prior to running this script (see
`scripts/create_dataset.py`). The default parameters here should reproduce
the published reacher results.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import numpy as np
import os
import sys
import tensorflow.compat.v2 as tf
tf.compat.v1.enable_v2_behavior()
import tensorflow_probability as tfp
import pickle

from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment

from dice_rl.environments.env_policies import get_target_policy
import dice_rl.environments.gridworld.navigation as navigation
import dice_rl.environments.gridworld.tree as tree
import dice_rl.environments.gridworld.taxi as taxi
from dice_rl.estimators.neural_bayes_dice import NeuralBayesDice
from dice_rl.estimators import estimator as estimator_lib
from dice_rl.networks.value_network import ValueNetwork
import dice_rl.utils.common as common_utils
from dice_rl.data.dataset import Dataset, EnvStep, StepType
from dice_rl.data.tf_offpolicy_dataset import TFOffpolicyDataset


FLAGS = flags.FLAGS

flags.DEFINE_string('load_dir', None, 'Directory to load dataset from.')
flags.DEFINE_string('save_dir', None,
                    'Directory to save the model and estimation results.')
flags.DEFINE_string('env_name', 'reacher', 'Environment name.')
flags.DEFINE_integer('seed', 0, 'Initial random seed.')
flags.DEFINE_bool('tabular_obs', False, 'Whether to use tabular observations.')
flags.DEFINE_integer('num_trajectory', 25, 'Number of trajectories to collect.')
flags.DEFINE_integer('max_trajectory_length', 100,
                     'Cutoff trajectory at this step.')
flags.DEFINE_float('alpha', 0.0, 'How close to target policy.')
flags.DEFINE_float('alpha_target', 1.0,
                   'How close is the target policy to optimal policy.')
flags.DEFINE_float('gamma', 0.99, 'Discount factor.')
flags.DEFINE_float('nu_learning_rate', 0.001, 'Learning rate for nu.')
flags.DEFINE_float('zeta_learning_rate', 0.0001, 'Learning rate for zeta.')
flags.DEFINE_float('nu_regularizer', 0.0, 'Ortho regularization on nu.')
flags.DEFINE_float('zeta_regularizer', 0.0, 'Ortho regularization on zeta.')
flags.DEFINE_integer('num_steps', 100000, 'Number of training steps.')
flags.DEFINE_integer('batch_size', 2048, 'Batch size.')

flags.DEFINE_float('f_exponent', 2., 'Exponent for f function.')
flags.DEFINE_bool('primal_form', True,
                  'Whether to use primal form of loss for nu.')

flags.DEFINE_float('primal_regularizer', 0.,
                   'LP regularizer of primal variables.')
flags.DEFINE_float('dual_regularizer', 0.001,
                   'LP regularizer of dual variables.')
flags.DEFINE_bool('zero_reward', True,
                  'Whether to ignore reward in optimization.')
flags.DEFINE_float('norm_regularizer', 0.,
                   'Weight of normalization constraint.')
flags.DEFINE_bool('zeta_pos', True, 'Whether to enforce positivity constraint.')

flags.DEFINE_float('scale_reward', 1., 'Reward scaling factor.')
flags.DEFINE_float('shift_reward', 0., 'Reward shift factor.')
flags.DEFINE_string(
    'transform_reward', None, 'Non-linear reward transformation'
    'One of [exp, cuberoot, None]')

# BayesDICE hyper-parameters
flags.DEFINE_float('kl_regularizer', 0.001, 'LP regularizer of kl(q||p).')
flags.DEFINE_float('eps_std', 1., 'Eps std for reparametrization.')


def main(argv):
  load_dir = FLAGS.load_dir
  save_dir = FLAGS.save_dir
  env_name = FLAGS.env_name
  seed = FLAGS.seed
  tabular_obs = FLAGS.tabular_obs
  num_trajectory = FLAGS.num_trajectory
  max_trajectory_length = FLAGS.max_trajectory_length
  alpha = FLAGS.alpha
  alpha_target = FLAGS.alpha_target
  gamma = FLAGS.gamma
  nu_learning_rate = FLAGS.nu_learning_rate
  zeta_learning_rate = FLAGS.zeta_learning_rate
  nu_regularizer = FLAGS.nu_regularizer
  zeta_regularizer = FLAGS.zeta_regularizer
  num_steps = FLAGS.num_steps
  batch_size = FLAGS.batch_size

  f_exponent = FLAGS.f_exponent
  primal_form = FLAGS.primal_form

  primal_regularizer = FLAGS.primal_regularizer
  dual_regularizer = FLAGS.dual_regularizer
  kl_regularizer = FLAGS.kl_regularizer
  zero_reward = FLAGS.zero_reward
  norm_regularizer = FLAGS.norm_regularizer
  zeta_pos = FLAGS.zeta_pos

  scale_reward = FLAGS.scale_reward
  shift_reward = FLAGS.shift_reward
  transform_reward = FLAGS.transform_reward

  kl_regularizer = FLAGS.kl_regularizer
  eps_std = FLAGS.eps_std

  def reward_fn(env_step):
    reward = env_step.reward * scale_reward + shift_reward
    if transform_reward is None:
      return reward
    if transform_reward == 'exp':
      reward = tf.math.exp(reward)
    elif transform_reward == 'cuberoot':
      reward = tf.sign(reward) * tf.math.pow(tf.abs(reward), 1.0 / 3.0)
    else:
      raise ValueError('Reward {} not implemented.'.format(transform_reward))
    return reward

  hparam_str = ('{ENV_NAME}_tabular{TAB}_alpha{ALPHA}_seed{SEED}_'
                'numtraj{NUM_TRAJ}_maxtraj{MAX_TRAJ}').format(
                    ENV_NAME=env_name,
                    TAB=tabular_obs,
                    ALPHA=alpha,
                    SEED=seed,
                    NUM_TRAJ=num_trajectory,
                    MAX_TRAJ=max_trajectory_length)
  train_hparam_str = (
      'nlr{NLR}_zlr{ZLR}_zeror{ZEROR}_preg{PREG}_dreg{DREG}_kreg{KREG}_nreg{NREG}_'
      'pform{PFORM}_fexp{FEXP}_zpos{ZPOS}_'
      'scaler{SCALER}_shiftr{SHIFTR}_transr{TRANSR}').format(
          NLR=nu_learning_rate,
          ZLR=zeta_learning_rate,
          ZEROR=zero_reward,
          PREG=primal_regularizer,
          DREG=dual_regularizer,
          KREG=kl_regularizer,
          NREG=norm_regularizer,
          PFORM=primal_form,
          FEXP=f_exponent,
          ZPOS=zeta_pos,
          SCALER=scale_reward,
          SHIFTR=shift_reward,
          TRANSR=transform_reward,
      )

  train_hparam_str = ('eps{EPS}_kl{KL}').format(EPS=eps_std, KL=kl_regularizer)

  if save_dir is not None:
    target_hparam_str = hparam_str.replace(
        'alpha{}'.format(alpha), 'alpha{}_alphat{}'.format(alpha, alpha_target))
    save_dir = os.path.join(save_dir, target_hparam_str, train_hparam_str)
    summary_writer = tf.summary.create_file_writer(logdir=save_dir)
    summary_writer.set_as_default()
  else:
    tf.summary.create_noop_writer()

  directory = os.path.join(load_dir, hparam_str)
  print('Loading dataset from', directory)
  dataset = Dataset.load(directory)
  #dataset = Dataset.load(directory.replace('alpha{}'.format(alpha), 'alpha0.0'))

  all_steps = dataset.get_all_steps()
  max_reward = tf.reduce_max(all_steps.reward)
  min_reward = tf.reduce_min(all_steps.reward)
  print('num loaded steps', dataset.num_steps)
  print('num loaded total steps', dataset.num_total_steps)
  print('num loaded episodes', dataset.num_episodes)
  print('num loaded total episodes', dataset.num_total_episodes)
  print('min reward', min_reward, 'max reward', max_reward)
  print('behavior per-step',
        estimator_lib.get_fullbatch_average(dataset, gamma=gamma))

  activation_fn = tf.nn.relu
  hidden_dims = (64, 64)
  input_spec = (dataset.spec.observation, dataset.spec.action)
  nu_network = ValueNetwork(
      input_spec,
      output_dim=2,
      fc_layer_params=hidden_dims,
      activation_fn=activation_fn,
      kernel_initializer=tf.keras.initializers.GlorotUniform(),
      last_kernel_initializer=tf.keras.initializers.GlorotUniform())
  output_activation_fn = tf.math.square if zeta_pos else tf.identity
  zeta_network = ValueNetwork(
      input_spec,
      output_dim=2,
      fc_layer_params=hidden_dims,
      activation_fn=activation_fn,
      output_activation_fn=output_activation_fn,
      kernel_initializer=tf.keras.initializers.GlorotUniform(),
      last_kernel_initializer=tf.keras.initializers.GlorotUniform())

  nu_optimizer = tf.keras.optimizers.Adam(nu_learning_rate)
  zeta_optimizer = tf.keras.optimizers.Adam(zeta_learning_rate)
  lam_optimizer = tf.keras.optimizers.Adam(nu_learning_rate)

  estimator = NeuralBayesDice(
      dataset.spec,
      nu_network,
      zeta_network,
      nu_optimizer,
      zeta_optimizer,
      lam_optimizer,
      gamma,
      zero_reward=zero_reward,
      f_exponent=f_exponent,
      primal_form=primal_form,
      reward_fn=reward_fn,
      primal_regularizer=primal_regularizer,
      dual_regularizer=dual_regularizer,
      kl_regularizer=kl_regularizer,
      eps_std=FLAGS.eps_std,
      norm_regularizer=norm_regularizer,
      nu_regularizer=nu_regularizer,
      zeta_regularizer=zeta_regularizer)

  global_step = tf.Variable(0, dtype=tf.int64)
  tf.summary.experimental.set_step(global_step)

  target_policy = get_target_policy(load_dir, env_name, tabular_obs,
                                    alpha_target)
  running_losses = []
  all_dual = []
  for step in range(num_steps):
    transitions_batch = dataset.get_step(batch_size, num_steps=2)
    initial_steps_batch, _ = dataset.get_episode(
        batch_size, truncate_episode_at=1)
    initial_steps_batch = tf.nest.map_structure(lambda t: t[:, 0, ...],
                                                initial_steps_batch)
    losses = estimator.train_step(initial_steps_batch, transitions_batch,
                                  target_policy)
    running_losses.append(losses)
    if step % 500 == 0 or step == num_steps - 1:
      num_samples = 100
      dual_ests = []
      for i in range(num_samples):
        dual_est = estimator.estimate_average_reward(
            dataset, target_policy, write_summary=(i == 0))
        dual_ests.append(dual_est)
      tf.summary.scalar('dual/mean', tf.math.reduce_mean(dual_ests))
      tf.summary.scalar('dual/std', tf.math.reduce_std(dual_ests))

      tf.print('dual/mean =', tf.math.reduce_mean(dual_ests), 'dual/std =',
               tf.math.reduce_std(dual_ests))

      all_dual.append(dual_ests)
      running_losses = []
    global_step.assign_add(1)

  if save_dir is not None:
    np.save(
        tf.io.gfile.GFile(os.path.join(save_dir, 'results.npy'), 'w'), all_dual)

  print('Done!')


if __name__ == '__main__':
  app.run(main)
