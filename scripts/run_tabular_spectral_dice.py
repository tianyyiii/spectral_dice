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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import numpy as np
import os
import sys
import gym
import tensorflow.compat.v2 as tf
tf.compat.v1.enable_v2_behavior()
from tensorflow.keras import layers
import pickle

from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment
from tf_agents.specs import tensor_spec
from tf_agents.networks import network

import sys
sys.path.append("/storage/coda1/p-kwang692/0/tchen667/tianyi/spectral-dice")
from dice_rl.environments.env_policies import get_target_policy
import dice_rl.environments.gridworld.navigation as navigation
import dice_rl.environments.gridworld.tree as tree
import dice_rl.environments.gridworld.taxi as taxi
from dice_rl.estimators.neural_dice import NeuralDice
from dice_rl.estimators.tabular_spectral_dice import SpectralNeuralDice
import dice_rl.data.dataset as dataset_lib
from dice_rl.estimators import estimator as estimator_lib
from dice_rl.networks.value_network import ValueNetwork
import dice_rl.utils.common as common_utils
from dice_rl.data.dataset import Dataset, EnvStep, StepType
from dice_rl.data.tf_offpolicy_dataset import TFOffpolicyDataset
import dice_rl.environments.env_policies as env_policies

from dice_rl.agent.speder.speder_agent_tabular import TabularSpederAgent
from dice_rl.estimators.tabular_spectral_dice_non_stochastic import SpectralTabularDice


FLAGS = flags.FLAGS

flags.DEFINE_string('load_dir', None, 'Directory to load dataset from.')
flags.DEFINE_string('save_dir', None,
                    'Directory to save the model and estimation results.')
flags.DEFINE_string('env_name', 'grid', 'Environment name.')
flags.DEFINE_integer('seed', 0, 'Initial random seed.')
flags.DEFINE_bool('tabular_obs', False, 'Whether to use tabular observations.')
flags.DEFINE_integer('num_trajectory', 1000,
                     'Number of trajectories to collect.')
flags.DEFINE_integer('max_trajectory_length', 40,
                     'Cutoff trajectory at this step.')
flags.DEFINE_float('alpha', 0.0, 'How close to target policy.')
flags.DEFINE_float('gamma', 0.99, 'Discount factor.')
flags.DEFINE_float('nu_learning_rate', 0.0001, 'Learning rate for nu.')
flags.DEFINE_float('zeta_learning_rate', 0.0001, 'Learning rate for zeta.')
flags.DEFINE_float('nu_regularizer', 0.0, 'Ortho regularization on nu.')
flags.DEFINE_float('zeta_regularizer', 0.0, 'Ortho regularization on zeta.')
flags.DEFINE_integer('num_steps', 50000, 'Number of training steps.')
flags.DEFINE_integer('batch_size', 2048, 'Batch size.')

flags.DEFINE_float('f_exponent', 2, 'Exponent for f function.')
flags.DEFINE_bool('primal_form', False,
                  'Whether to use primal form of loss for nu.')

flags.DEFINE_float('primal_regularizer', 0.,
                   'LP regularizer of primal variables.')
flags.DEFINE_float('dual_regularizer', 1., 'LP regularizer of dual variables.')
flags.DEFINE_bool('zero_reward', False,
                  'Whether to ignore reward in optimization.')
flags.DEFINE_float('norm_regularizer', 1.,
                   'Weight of normalization constraint.')
flags.DEFINE_bool('zeta_pos', True, 'Whether to enforce positivity constraint.')

flags.DEFINE_float('scale_reward', 1., 'Reward scaling factor.')
flags.DEFINE_float('shift_reward', 0., 'Reward shift factor.')
flags.DEFINE_string(
    'transform_reward', None, 'Non-linear reward transformation'
    'One of [exp, cuberoot, None]')

flags.DEFINE_integer('stage', 0, 'Which stage of spectral-dice to run.')
flags.DEFINE_integer('feature_dim', 2048, 'Feature dim of phi and mu.')
flags.DEFINE_bool('sd_save', False, 'Whether to save the stage 1 models.')

flags.DEFINE_integer('decay_steps', 1000, 'Decay steps for stage 2')
flags.DEFINE_float('decay_rate', 0.9, 'Lr scheduler decay rate for stage 2')

flags.DEFINE_bool('use_ema', False, 'Whether to use ema for the second stage training')
flags.DEFINE_float('ema_momentum', 0.99, 'Ema momentum for second stage training')
flags.DEFINE_integer('ema_overwrite_frequency', 100, 'Ema overwrite frequency for second stage training')


class Theta(tf.keras.Model):
    def __init__(self, layer1, feature_dim=1024):
        super(Theta, self).__init__()
        self.layer1 = tf.constant(layer1)
        self.l = layers.Dense(1, input_shape=(feature_dim,), activation=None, use_bias=False, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.l.trainable = True

    def call(self, inputs):
        states = inputs[0]
        actions = inputs[1]
        #this is for four_rooms only
        sa_indices = states * 4 + actions
        feature = tf.gather(self.layer1, sa_indices)
        r = self.l(feature)
        r = tf.squeeze(r)
        return r, 0


def main(argv):
  load_dir = FLAGS.load_dir
  save_dir = FLAGS.save_dir
  env_name = FLAGS.env_name
  seed = FLAGS.seed
  tabular_obs = FLAGS.tabular_obs
  num_trajectory = FLAGS.num_trajectory
  max_trajectory_length = FLAGS.max_trajectory_length
  alpha = FLAGS.alpha
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
  zero_reward = FLAGS.zero_reward
  norm_regularizer = FLAGS.norm_regularizer
  zeta_pos = FLAGS.zeta_pos

  scale_reward = FLAGS.scale_reward
  shift_reward = FLAGS.shift_reward
  transform_reward = FLAGS.transform_reward

  feature_dim = FLAGS.feature_dim
  stage = FLAGS.stage

  decay_steps = FLAGS.decay_steps
  decay_rate = FLAGS.decay_rate

  use_ema = FLAGS.use_ema
  ema_momentum = FLAGS.ema_momentum
  ema_overwrite_frequency = FLAGS.ema_overwrite_frequency

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
      'nlr{NLR}_zlr{ZLR}_zeror{ZEROR}_preg{PREG}_dreg{DREG}_nreg{NREG}_'
      'pform{PFORM}_fexp{FEXP}_zpos{ZPOS}_'
      'scaler{SCALER}_shiftr{SHIFTR}_transr{TRANSR}').format(
          NLR=nu_learning_rate,
          ZLR=zeta_learning_rate,
          ZEROR=zero_reward,
          PREG=primal_regularizer,
          DREG=dual_regularizer,
          NREG=norm_regularizer,
          PFORM=primal_form,
          FEXP=f_exponent,
          ZPOS=zeta_pos,
          SCALER=scale_reward,
          SHIFTR=shift_reward,
          TRANSR=transform_reward)
  if save_dir is not None:
    save_dir = os.path.join(save_dir, hparam_str, train_hparam_str)
    summary_writer = tf.summary.create_file_writer(logdir=save_dir)
    summary_writer.set_as_default()
  else:
    tf.summary.create_noop_writer()

  directory = os.path.join(load_dir, hparam_str)
  print('Loading dataset from', directory)
  dataset = Dataset.load(directory)
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
  target_dataset = Dataset.load(
      directory.replace('alpha{}'.format(alpha), 'alpha1.0'))
  print('target per-step',
        estimator_lib.get_fullbatch_average(target_dataset, gamma=1.))
  target_policy = get_target_policy(load_dir, env_name, tabular_obs)

  if stage == 1 or stage == 3:
    speder = TabularSpederAgent(directory, env_name, load_dir, feature_dim=feature_dim)
    U, Vt, p_table, mu_0, reward, _ = speder.train()


  if stage == 2 or stage == 3:
    U = tf.convert_to_tensor(U, dtype=tf.float32)  
    Vt = tf.convert_to_tensor(Vt, dtype=tf.float32)  
    V = tf.transpose(Vt)  
    nu_network = Theta(layer1=U, feature_dim=feature_dim)
    zeta_network = Theta(layer1=V, feature_dim=feature_dim)

    nu_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
      initial_learning_rate=nu_learning_rate, decay_steps=decay_steps, decay_rate=decay_rate)
    zeta_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
      initial_learning_rate=zeta_learning_rate, decay_steps=decay_steps, decay_rate=decay_rate)
    if use_ema == True:
      nu_optimizer = tf.keras.optimizers.Adam(nu_schedule, clipvalue=1.0, use_ema=True, ema_momentum=ema_momentum, ema_overwrite_frequency=ema_overwrite_frequency)
      zeta_optimizer = tf.keras.optimizers.Adam(zeta_schedule, clipvalue=1.0, use_ema=True, ema_momentum=ema_momentum, ema_overwrite_frequency=ema_overwrite_frequency)
    else:
      nu_optimizer = tf.keras.optimizers.Adam(nu_schedule, clipvalue=10000.0)
      zeta_optimizer = tf.keras.optimizers.Adam(zeta_schedule, clipvalue=10000.0)
    lam_optimizer = tf.keras.optimizers.Adam(nu_learning_rate, clipvalue=1.0)

    estimator = SpectralNeuralDice(
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
        norm_regularizer=norm_regularizer,
        nu_regularizer=nu_regularizer,
        zeta_regularizer=zeta_regularizer)

    global_step = tf.Variable(0, dtype=tf.int64)
    tf.summary.experimental.set_step(global_step)

    running_losses = []
    running_estimates = []
    for step in range(num_steps):
      transitions_batch = dataset.get_step(batch_size, num_steps=2)
      if env_name == "reacher":
        transitions_batch = transitions_batch._replace(observation=tf.cast(transitions_batch.observation, dtype=tf.float32)) 
      initial_steps_batch, _ = dataset.get_episode(
          batch_size, truncate_episode_at=1)
      if env_name == "reacher":
        initial_steps_batch = initial_steps_batch._replace(observation = tf.cast(initial_steps_batch.observation, dtype=tf.float32))
      initial_steps_batch = tf.nest.map_structure(lambda t: t[:, 0, ...],
                                                  initial_steps_batch)
      losses = estimator.train_step(initial_steps_batch, transitions_batch,
                                    target_policy)
      running_losses.append(losses)
      if step % 500 == 0 or step == num_steps - 1:
        estimate = estimator.estimate_average_reward(dataset, target_policy)
        running_estimates.append(estimate)
        running_losses = []
      global_step.assign_add(1)

  print('Done!')


if __name__ == '__main__':
  app.run(main)
