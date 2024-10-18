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
import copy

from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment
from tf_agents.specs import tensor_spec
from tf_agents.networks import network

import sys
sys.path.append("/home/tianyi")
from dice_rl.environments.env_policies import get_target_policy
import dice_rl.environments.gridworld.navigation as navigation
import dice_rl.environments.gridworld.tree as tree
import dice_rl.environments.gridworld.taxi as taxi
from dice_rl.estimators.neural_dice import NeuralDice
from dice_rl.estimators.neural_spectral_dice import SpectralNeuralDice
import dice_rl.data.dataset as dataset_lib
from dice_rl.estimators import estimator as estimator_lib
from dice_rl.networks.value_network import ValueNetwork
import dice_rl.utils.common as common_utils
from dice_rl.data.dataset import Dataset, EnvStep, StepType
from dice_rl.data.tf_offpolicy_dataset import TFOffpolicyDataset

from dice_rl.agent.speder.speder_agent import SPEDERAgent


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
flags.DEFINE_integer('speder_d_step', 5000, 'Training steps for stage 1.')
flags.DEFINE_float('phi_and_mu_lr', 0.001, 'Learning rate for stage 1.')
flags.DEFINE_integer('phi_hidden_dim', 512, 'Hidden dim for phi network.')
flags.DEFINE_integer('phi_hidden_depth', 1, 'Hidden depth for phi network.')
flags.DEFINE_integer('mu_hidden_dim', 512, "Hidden dim for mu network.")
flags.DEFINE_integer('mu_hidden_depth', 1, 'Hidden depth for mu network.')
flags.DEFINE_integer('sd_batch_size', 1024, "Batch size for stage 1.")
flags.DEFINE_bool('sd_save', False, 'Whether to save the stage 1 models.')
flags.DEFINE_integer('decay_steps', 1000, 'Decay steps for stage 1')
flags.DEFINE_float('decay_rate', 0.9, 'Lr scheduler decay rate for stage 1')

flags.DEFINE_integer('decay_steps_2', 1000, 'Decay steps for stage 2')
flags.DEFINE_float('decay_rate_2', 0.9, 'Lr scheduler decay rate for stage 2')
flags.DEFINE_bool('use_ema', False, 'Whether to use ema for the second stage training')
flags.DEFINE_float('ema_momentum', 0.99, 'Ema momentum for second stage training')
flags.DEFINE_integer('ema_overwrite_frequency', 100, 'Ema overwrite frequency for second stage training')

flags.DEFINE_bool('test_set', False, 'Whether to use test set for stage 1')


class Theta(tf.keras.Model):
    def __init__(self, layer1, load_from_pretrained, feature_dim=1024):
        super(Theta, self).__init__()
        if not load_from_pretrained:
          self.layer1 = tf.keras.models.clone_model(layer1)
          self.layer1.set_weights(layer1.get_weights())
        else:
          self.layer1 = layer1
        self.layer1.trainable = False
        self.l = layers.Dense(1, input_shape=(feature_dim,), activation=None, use_bias=False, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.l.trainable = True

    def call(self, inputs):
        states = inputs[0]
        actions = inputs[1]
        if len(actions.shape) < len(states.shape):
          actions = tf.expand_dims(actions, axis=-1)
          actions = tf.cast(actions, tf.float32)
        state_action = tf.concat([states, actions], axis=-1)
        feature = self.layer1(state_action)
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

  stage = FLAGS.stage

  feature_dim = FLAGS.feature_dim
  speder_d_step = FLAGS.speder_d_step
  phi_and_mu_lr = FLAGS.phi_and_mu_lr
  phi_hidden_dim = FLAGS.phi_hidden_dim
  phi_hidden_depth = FLAGS.phi_hidden_depth
  mu_hidden_dim = FLAGS.mu_hidden_dim
  mu_hidden_depth = FLAGS.mu_hidden_depth
  sd_batch_size =FLAGS.sd_batch_size
  sd_save = FLAGS.sd_save

  decay_steps = FLAGS.decay_steps
  decay_rate = FLAGS.decay_rate

  use_ema = FLAGS.use_ema
  ema_momentum = FLAGS.ema_momentum
  ema_overwrite_frequency = FLAGS.ema_overwrite_frequency

  test_set = FLAGS.test_set

  decay_steps_2 = FLAGS.decay_steps_2
  decay_rate_2 = FLAGS.decay_rate_2

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
      directory.replace('alpha{}'.format(alpha), 'alpha1.0').replace('seed{}'.format(seed), 'seed0'))
  print('target per-step',
        estimator_lib.get_fullbatch_average(target_dataset, gamma=1.))
  target_policy = get_target_policy(load_dir, env_name, tabular_obs)

  if stage == 1 or stage == 3:
    if env_name == "cartpole":
      state_dim = 4
      action_dim = 1 
    else:
      if env_name == "reacher":
        env_temp_name = "Reacher-v2"
      else:
        env_temp_name = env_name
      temp_env = gym.make(env_temp_name)
      state_dim = temp_env.observation_space.shape[0]
      action_dim = temp_env.action_space.shape[0]

    feature_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=phi_and_mu_lr, 
                                                                 decay_steps=decay_steps, decay_rate=decay_rate)
    feature_optimizer = tf.keras.optimizers.Adam(feature_schedule, clipvalue=1.0)

    speder = SPEDERAgent(state_dim=state_dim, 
                        action_dim=action_dim, 
                        feature_optimizer=feature_optimizer,
                        phi_hidden_dim=phi_hidden_dim,
                        phi_hidden_depth=phi_hidden_depth, 
                        mu_hidden_dim=mu_hidden_dim, 
                        mu_hidden_depth=mu_hidden_depth, 
                        feature_dim=feature_dim)

    best_loss = 10000
    stage_loss = 0
    model_loss = 0
    for step in range(speder_d_step):
      step_batch = dataset.get_step(sd_batch_size, num_steps=2)
      if env_name == "cartpole":
        states = step_batch.observation[:, 0, :]
        actions = step_batch.action[:, 0]
        actions = np.expand_dims(actions, axis=-1)
        next_states = step_batch.observation[:, 1, :]
        rewards = step_batch.reward[:, 0]
        tfagents_step = dataset_lib.convert_to_tfagents_timestep(step_batch)
        next_actions = target_policy.action(tfagents_step).action[:, 1]
        next_actions = next_actions.numpy()  
        next_actions = np.expand_dims(next_actions, axis=-1)   

      #for reacher and other continuous envs
      else:
        states = step_batch.observation[:, 0, :]
        actions = step_batch.action[:, 0, :]
        next_states = step_batch.observation[:, 1, :]
        rewards = step_batch.reward[:, 0]
        step_batch_one = copy.deepcopy(step_batch)
        step_batch_one = step_batch_one._replace(observation=step_batch_one.observation[:, 1, :])
        step_batch_one = step_batch_one._replace(reward=step_batch_one.reward[:, 1])
        step_batch_one = step_batch_one._replace(step_type=step_batch_one.step_type[:, 1])
        step_batch_one = step_batch_one._replace(discount=step_batch_one.discount[:, 1])
        tfagents_step = dataset_lib.convert_to_tfagents_timestep(step_batch_one)
        next_actions = target_policy.action(tfagents_step).action
        next_actions = next_actions.numpy()

      step_batch2 = dataset.get_step(sd_batch_size)
      states2 = step_batch2.observation
      actions2 = step_batch2.action
      step_batch3 = dataset.get_step(sd_batch_size)
      states3 = step_batch3.observation
      actions3 = step_batch3.action    
      if env_name == "cartpole":
        actions2 = np.expand_dims(actions2, axis=-1)
        actions3 = np.expand_dims(actions3, axis=-1)
      feature_info = speder.feature_step(states, actions, next_states, next_actions, states2, actions2, states3, actions3, rewards)
      stage_loss += feature_info["total_loss"]
      model_loss += feature_info["model_loss"]
      
      if step % 100 == 0:
        if step != 0:
          stage_loss = stage_loss / 100
          model_loss = model_loss / 100
        print(f"ERM total loss at step {step}: {stage_loss} Model loss: {model_loss}")
        if model_loss < best_loss and sd_save == True:
          phi_network, mu_network = speder.final_info()
          phi_network.save("checkpoints/phi" + hparam_str + "_" + str(decay_rate) +"_" + str(decay_steps) + "_" + str(phi_hidden_dim) + "_" + str(phi_and_mu_lr) + "_" + str(zeta_learning_rate) + train_hparam_str+ ".keras")
          mu_network.save("checkpoints/mu" + hparam_str + "_" + str(decay_rate) +"_" + str(decay_steps) + "_" + str(phi_hidden_dim) + "_" + str(phi_and_mu_lr)  + "_" + str(zeta_learning_rate) + train_hparam_str + ".keras")
          best_loss = model_loss
        stage_loss = 0
        model_loss = 0

  if stage == 2 or stage == 3:
    phi_network = tf.keras.models.load_model("checkpoints/phi" + hparam_str + "_" + str(decay_rate) +"_" + str(decay_steps) + "_" + str(phi_hidden_dim) + "_" + str(phi_and_mu_lr) + "_" + str(zeta_learning_rate) + train_hparam_str + ".keras")
    mu_network = tf.keras.models.load_model("checkpoints/mu" + hparam_str + "_" + str(decay_rate) +"_" + str(decay_steps) + "_" + str(phi_hidden_dim) + "_" + str(phi_and_mu_lr)  + "_" + str(zeta_learning_rate) + train_hparam_str + ".keras")
    # phi_network = tf.keras.models.load_model("checkpoints/phi" + hparam_str + "_" + str(decay_rate) +"_" + str(decay_steps) + "_" + str(phi_hidden_dim) + ".keras")
    # mu_network = tf.keras.models.load_model("checkpoints/mu" + hparam_str + "_" + str(decay_rate) +"_" + str(decay_steps) + "_" + str(phi_hidden_dim) + ".keras")    
    nu_network = Theta(layer1=phi_network, feature_dim=feature_dim, load_from_pretrained=True)
    zeta_network = Theta(layer1=mu_network, feature_dim=feature_dim, load_from_pretrained=True)

    nu_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
      initial_learning_rate=nu_learning_rate, decay_steps=decay_steps_2, decay_rate=decay_rate_2)
    zeta_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
      initial_learning_rate=zeta_learning_rate, decay_steps=decay_steps_2, decay_rate=decay_rate_2)
    if use_ema == True:
      nu_optimizer = tf.keras.optimizers.Adam(nu_schedule, clipvalue=1.0, use_ema=True, ema_momentum=ema_momentum, ema_overwrite_frequency=ema_overwrite_frequency)
      zeta_optimizer = tf.keras.optimizers.Adam(zeta_schedule, clipvalue=1.0, use_ema=True, ema_momentum=ema_momentum, ema_overwrite_frequency=ema_overwrite_frequency)
    else:
      nu_optimizer = tf.keras.optimizers.Adam(nu_schedule, clipvalue=1.0)
      zeta_optimizer = tf.keras.optimizers.Adam(zeta_schedule, clipvalue=1.0)
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
      elif env_name == "cartpole":
        transitions_batch = transitions_batch._replace(action = tf.cast(transitions_batch.action, dtype=tf.float32))
        transitions_batch = transitions_batch._replace(action = tf.expand_dims(transitions_batch.action, axis=-1))
      initial_steps_batch, _ = dataset.get_episode(
          batch_size, truncate_episode_at=1)
      if env_name == "reacher":
        initial_steps_batch = initial_steps_batch._replace(observation = tf.cast(initial_steps_batch.observation, dtype=tf.float32))
      elif env_name == "cartpole":
        initial_steps_batch = initial_steps_batch._replace(action = tf.cast(initial_steps_batch.action, dtype=tf.float32))
        initial_steps_batch = initial_steps_batch._replace(action = tf.expand_dims(initial_steps_batch.action, axis=-1))
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
