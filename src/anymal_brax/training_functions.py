import functools
import jax
import os
import string
import brax
import flax
import matplotlib.pyplot as plt

from datetime import datetime
from jax import numpy as jp
from IPython.display import HTML, clear_output
from etils import epath
from brax import envs
from brax.io import model
from brax.io import json
from brax.io import html
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import train as sac
from brax.training.agents.apg import train as apg

import mlflow

# Global Variables -------------------------------------------------------------
data_container = {}


# Set up environment ----------------------------------------------------------
def setup_env(config: dict):
  env = envs.get_environment(env_name=config["env_name"],
                            backend=config["backend"])
  state = jax.jit(env.reset)(rng=jax.random.PRNGKey(seed=2))
  
  return env

# Training function -----------------------------------------------------------
def setup_train_fn(config: dict):
  if config["training_agent"] == ppo or config["training_agent"] == sac:
    train_fn = {
      ## Params changed for for super fast training (rapid debugging of the infrastructure)
      'anymal_c': functools.partial(config["training_agent"].train, num_timesteps=20_000_000, num_evals=10, reward_scaling=0.1, episode_length=config["episode_length"], normalize_observations=True, action_repeat=1, unroll_length=5, num_minibatches=32, num_updates_per_batch=8, discounting=0.97, learning_rate=2*config["learning_rate"], entropy_cost=1e-2, num_envs=config["num_envs"], batch_size=1024, seed=1),
      'a1': functools.partial(config["training_agent"].train, num_timesteps=500_000, num_evals=10, reward_scaling=10, episode_length=config["episode_length"], normalize_observations=True, action_repeat=1, unroll_length=5, num_minibatches=32, num_updates_per_batch=8, discounting=0.97, learning_rate=config["learning_rate"], entropy_cost=1e-2, num_envs=2048, batch_size=1024, seed=1),
      'inverted_pendulum': functools.partial(config["training_agent"].train, num_timesteps=2_000_000, num_evals=20, reward_scaling=10, episode_length=config["episode_length"], normalize_observations=True, action_repeat=1, unroll_length=5, num_minibatches=32, num_updates_per_batch=4, discounting=0.97, learning_rate=config["learning_rate"], entropy_cost=1e-2, num_envs=2048, batch_size=1024, seed=1),
      'inverted_double_pendulum': functools.partial(config["training_agent"].train, num_timesteps=20_000_000, num_evals=20, reward_scaling=10, episode_length=config["episode_length"], normalize_observations=True, action_repeat=1, unroll_length=5, num_minibatches=32, num_updates_per_batch=4, discounting=0.97, learning_rate=config["learning_rate"], entropy_cost=1e-2, num_envs=2048, batch_size=1024, seed=1),
      'ant': functools.partial(config["training_agent"].train,  num_timesteps=1_000_000, num_evals=10, reward_scaling=10, episode_length=config["episode_length"], normalize_observations=True, action_repeat=1, unroll_length=5, num_minibatches=32, num_updates_per_batch=4, discounting=0.97, learning_rate=config["learning_rate"], entropy_cost=1e-2, num_envs=4096, batch_size=2048, seed=1),
      'humanoid': functools.partial(config["training_agent"].train,  num_timesteps=50_000_000, num_evals=10, reward_scaling=0.1, episode_length=config["episode_length"], normalize_observations=True, action_repeat=1, unroll_length=10, num_minibatches=32, num_updates_per_batch=8, discounting=0.97, learning_rate=config["learning_rate"], entropy_cost=1e-3, num_envs=2048, batch_size=1024, seed=1),
      'reacher': functools.partial(config["training_agent"].train, num_timesteps=50_000_000, num_evals=20, reward_scaling=5, episode_length=config["episode_length"], normalize_observations=True, action_repeat=4, unroll_length=50, num_minibatches=32, num_updates_per_batch=8, discounting=0.95, learning_rate=config["learning_rate"], entropy_cost=1e-3, num_envs=2048, batch_size=256, max_devices_per_host=8, seed=1),
      'humanoidstandup': functools.partial(config["training_agent"].train, num_timesteps=100_000_000, num_evals=20, reward_scaling=0.1, episode_length=config["episode_length"], normalize_observations=True, action_repeat=1, unroll_length=15, num_minibatches=32, num_updates_per_batch=8, discounting=0.97, learning_rate=2*config["learning_rate"], entropy_cost=1e-2, num_envs=2048, batch_size=1024, seed=1),
      'hopper': functools.partial(config["training_agent"].train, num_timesteps=6_553_600, num_evals=20, reward_scaling=30, episode_length=config["episode_length"], normalize_observations=True, action_repeat=1, discounting=0.997, learning_rate=2*config["learning_rate"], num_envs=128, batch_size=512, grad_updates_per_step=64, max_devices_per_host=1, max_replay_size=1048576, min_replay_size=8192, seed=1),
      'walker2d': functools.partial(config["training_agent"].train, num_timesteps=7_864_320, num_evals=20, reward_scaling=5, episode_length=config["episode_length"], normalize_observations=True, action_repeat=1, discounting=0.997, learning_rate=2*config["learning_rate"], num_envs=128, batch_size=128, grad_updates_per_step=32, max_devices_per_host=1, max_replay_size=1048576, min_replay_size=8192, seed=1),
      'halfcheetah': functools.partial(config["training_agent"].train, num_timesteps=50_000_000, num_evals=20, reward_scaling=1, episode_length=config["episode_length"], normalize_observations=True, action_repeat=1, unroll_length=20, num_minibatches=32, num_updates_per_batch=8, discounting=0.95, learning_rate=config["learning_rate"], entropy_cost=0.001, num_envs=2048, batch_size=512, seed=3),
      'pusher': functools.partial(config["training_agent"].train, num_timesteps=50_000_000, num_evals=20, reward_scaling=5, episode_length=config["episode_length"], normalize_observations=True, action_repeat=1, unroll_length=30, num_minibatches=16, num_updates_per_batch=8, discounting=0.95, learning_rate=config["learning_rate"], entropy_cost=1e-2, num_envs=2048, batch_size=512, seed=3),
    }[config["env_name"]]
    config["training_agent_name"] = 'ppo_sac'
  elif config["training_agent"] == apg:
    train_fn = {
      'anymal_c': functools.partial(config["training_agent"].train, num_timesteps=20_000_000, num_evals=10, reward_scaling=0.1, episode_length=config["episode_length"], normalize_observations=True, action_repeat=1, unroll_length=5, num_minibatches=32, num_updates_per_batch=8, discounting=0.97, learning_rate=2*config["learning_rate"], entropy_cost=1e-2, num_envs=config["num_envs"], batch_size=1024, seed=1),
      'ant': functools.partial(config["training_agent"].train, num_evals=config["num_episodes"], episode_length=config["episode_length"], truncation_length=config["truncation_length"], normalize_observations=True, action_repeat=config["action_repeat"], learning_rate=config["learning_rate"], num_envs=config["num_envs"], num_eval_envs=config["num_envs"], max_devices_per_host=8, seed=1),
      'reacher': functools.partial(config["training_agent"].train, num_evals=config["num_episodes"], episode_length=config["episode_length"], truncation_length=config["truncation_length"], normalize_observations=True, action_repeat=config["action_repeat"], learning_rate=config["learning_rate"], num_envs=config["num_envs"], num_eval_envs=config["num_envs"], deterministic_eval=config["deterministic_eval"], max_devices_per_host=8, seed=1),
    }[config["env_name"]]
    config["training_agent_name"] = 'apg'
  else:
    raise NotImplementedError(f'unknown training agent: {config["training_agent"]}')
  
  return train_fn

# Global data containers ------------------------------------------------------
def init_data_containers():
  data_container = {
    "times": [datetime.now()],
    "xdata": [],
    "ydata": [],
    "training_steps": [],
    "eval_rewards": [],
    "episode_model_params": [],
    "training_sps": [],
    "eval_sps": [],
    "train_grad_norms": []
  }
  return data_container

# Progress function -----------------------------------------------------------
def progress(num_steps, metrics):
  data_container["times"].append(datetime.now())
  data_container["xdata"].append(num_steps)
  data_container["ydata"].append(metrics['eval/episode_reward'])
  if "training/sps" in metrics:
    # Metrics
    print(f'steps: {num_steps} eval reward: {metrics["eval/episode_reward"]}')
    print(f'steps: {num_steps} eval control reward: {metrics["eval/episode_reward_ctrl"]}')
    print(f'steps: {num_steps} eval distance reward: {metrics["eval/episode_reward_dist"]}')
    data_container["training_steps"].append(num_steps)
    data_container["eval_rewards"].append(metrics['eval/episode_reward'])
    data_container["training_sps"].append(metrics['training/sps'])
    data_container["eval_sps"].append(metrics['eval/sps'])
    # Mlflow metric logging
    mlflow.log_metric("eval_reward", metrics['eval/episode_reward'], step=num_steps)
    mlflow.log_metric("eval_reward_ctrl", metrics['eval/episode_reward_ctrl'], step=num_steps)
    mlflow.log_metric("eval_reward_dist", metrics['eval/episode_reward_dist'], step=num_steps)
    mlflow.log_metric("training_sps", metrics['training/sps'], step=num_steps)
    mlflow.log_metric("eval_sps", metrics['eval/sps'], step=num_steps)
    if "training/grad_norm" in metrics:
      print(f'steps: {num_steps} training grad norm: {metrics["training/grad_norm"]}')
      data_container["train_grad_norms"].append(metrics['training/grad_norm'])
      mlflow.log_metric("training_grad_norm", metrics['training/grad_norm'], step=num_steps)
    if "episode_params" in metrics:
      # Artifacts
      data_container["episode_model_params"].append(metrics['episode_params'])
    # Mlflow artifact logging
    # TODO

# Test rollout ----------------------------------------------------------------
def run_rollout(params, config, make_policy, env):
  inference_fn = make_policy(params, deterministic=config["deterministic_eval"])
  jit_env_reset = jax.jit(env.reset)
  jit_env_step = jax.jit(env.step)
  jit_inference_fn = jax.jit(inference_fn)
  # Run rollout
  rollout = []
  rng = jax.random.PRNGKey(10)
  state = jit_env_reset(rng=rng)
  for _ in range(config["episode_length"]):
    rollout.append(state.pipeline_state)
    act_rng, rng = jax.random.split(rng)
    act, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_env_step(state, act)
  return rollout

# Check whether directory exists and create it if not -------------------------
def check_dir_exists(dir_path):
  isExist = os.path.exists(dir_path)
  if not isExist:
    os.makedirs(dir_path)
    print("The new directory is created!")


# Print summary -------------------------------------------------
def print_training_summary(data_container, training_metrics):
  print(f'time to JIT: {data_container["times"][1] - data_container["times"][0]}')
  print(f'time to train: {data_container["times"][-1] - data_container["times"][1]}')
  print('Average training steps per second:', jp.mean(jp.asarray(data_container["training_sps"])))
  print('Average eval steps per second:', jp.mean(jp.asarray(data_container["eval_sps"])))
  print('Average eval reward:', jp.mean(jp.asarray(data_container["eval_rewards"])))
  best_episode_index = jp.argmax(jp.asarray(data_container["eval_rewards"]))
  print('Best episode params achieved in episode ', best_episode_index)
  data_container["best_episode_index"] = best_episode_index
  print('Best episode reward:', data_container["eval_rewards"][best_episode_index])
  if len(data_container["episode_model_params"]) > 0:
    data_container["best_episode_model_params"] = data_container["episode_model_params"][best_episode_index]
  print('Final episode reward:', training_metrics['eval/episode_reward'])

def save_final_figure(data_container: dict, config: dict):
  fig, axis = plt.subplots(2, 1)
  fig.set_size_inches(10, 10)
  # Validation
  axis[0].plot(data_container["xdata"], data_container["ydata"])
  axis[0].set_title('Validation')
  axis[0].set_xlim([0, config["num_episodes"] if config["training_agent"] == ppo else config["num_evals"]])
  axis[0].set_ylim([min(data_container["ydata"]), max(data_container["ydata"])])
  axis[0].set_xlabel('# environment steps')
  axis[0].set_ylabel('reward per episode')
  # Training
  axis[1].plot(data_container["training_steps"], data_container["train_grad_norms"])
  axis[1].set_title('Training')
  axis[1].set_xlabel('# environment steps')
  axis[1].set_ylabel('gradient norm')
  # Save to disk
  dir_path = str(epath.resource_path('brax')) +'/../results/'
  check_dir_exists(dir_path)
  plt.savefig(dir_path + str(datetime.now()) + '_' + config["env_name"] + '_' + config["backend"] + "_" + config["training_agent_name"] + '.png')

# Create HTML ------------------------------------------------------------------
def create_html_data(rollout: list, env):
  html_object = HTML(html.render(env.sys.replace(dt=env.dt), rollout))
  return html_object.data

# Write HTML to disk -----------------------------------------------------------
def write_html_to_disk(html_data, html_filepath: str):
  import builtins
  with builtins.open(html_filepath, 'w') as f:
    f.write(html_data)

# MLFLOW Commodities -----------------------------------------------------------------
## Log config
def log_config(config):
  for dict_entry in config:
    mlflow.log_param(dict_entry, config[dict_entry])

## Log HTML
def log_html(html_path: str):
  mlflow.log_artifact(html_path)

def log_best_model_params(data_container: dict):
  pkl_file_path = '/tmp/best_model_params_episode' + str(data_container["best_episode_index"]) + ".pkl"
  model.save_params(pkl_file_path, data_container["best_episode_model_params"])
  mlflow.log_artifact(pkl_file_path)

# Training function
def train(config: dict):

  # Setup -----------------------------------------------------------------------
  # Get environment
  env = setup_env(config)
  # Get training function
  train_fn = setup_train_fn(config)
  # Setup Data containers
  global data_container 
  data_container= init_data_containers()

  # Setup mlflow experiment -----------------------------------------------------
  mlflow.set_experiment(f'{config["training_agent_name"]}_{config["env_name"]}')
  
  # Start experiment run --------------------------------------------------------
  with mlflow.start_run(run_name=str(datetime.now())):
    log_config(config)

    # Train -----------------------------------------------------------------------
    print("Training...")
    make_inference_fn, params, training_metrics = train_fn(environment=env, progress_fn=progress)

    # Print summary -------------------------------------------------
    print_training_summary(data_container, training_metrics)

    # Logging ---------------------------------------------------------------------
    model.save_params('/tmp/params', params)
    last_params = model.load_params('/tmp/params')

    # Log video of last ----------------------------------------------------------------
    rollout_last = run_rollout(last_params, config=config, make_policy=make_inference_fn, env=env)
    html_data_last = create_html_data(rollout_last, env)
    dir_path = str(epath.resource_path('brax')) +'/../results/'
    check_dir_exists(dir_path)
    html_path_last = "".join(dir_path + str(datetime.now()) + '_last_' + config["env_name"] + '_' + config["backend"] + "_" + config["training_agent_name"] + '.html')
    write_html_to_disk(html_data_last, html_path_last)
    log_html(html_path_last)

    # Log video of best ----------------------------------------------------------------
    if len(data_container["episode_model_params"]) > 0:
      rollout_best = run_rollout(data_container["best_episode_model_params"], config=config, make_policy=make_inference_fn, env=env)
      html_data_best = create_html_data(rollout_best, env)
      html_path_best = "".join(dir_path + str(datetime.now()) + '_best_' + config["env_name"] + '_' + config["backend"] + "_" + config["training_agent_name"] + '.html')
      write_html_to_disk(html_data_best, html_path_best)
      log_html(html_path_best)

      # Log model params ----------------------------------------------------------------
      log_best_model_params(data_container)
