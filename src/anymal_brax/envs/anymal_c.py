# Copyright 2023 The Brax Authors.
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

# pylint:disable=g-multiple-import

# THE URDF FOR THIS MODEL IS TAKEN FROM: https://github.com/ANYbotics/anymal_c_simple_description

from brax import actuator
from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp


class Anymal_c(PipelineEnv):

  def __init__(
      self,
      standing_height_goal=0.55, # This correspondes to the default height of the body, NOT the CoM
      standing_reward_weight=20,
      forward_reward_weight=0.7,
      ctrl_cost_weight=0.5,
      healthy_reward=1.0,
      terminate_when_unhealthy=True,
      healthy_z_range=(0.4, 0.7),
      reset_noise_scale=1e-2,
      exclude_current_positions_from_observation=True,
      backend='generalized',
      **kwargs,
  ):
    path = epath.resource_path('brax') / 'envs/assets/anymal_c.xml'
    sys = mjcf.load(path)

    n_frames = 5

    if backend in ['spring', 'positional']:
      sys = sys.replace(dt=0.0015)
      n_frames = 10
    else:
      sys = sys.replace(dt=0.0015)
      #gear = jp.array([
      #    350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0,
      #    350.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])  # pyformat: disable
      #sys = sys.replace(actuator=sys.actuator.replace(gear=gear))

    kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

    super().__init__(sys=sys, backend=backend, **kwargs)

    self._standing_height_goal = standing_height_goal
    self._standing_reward_weight = standing_reward_weight
    self._forward_reward_weight = forward_reward_weight
    self._ctrl_cost_weight = ctrl_cost_weight
    self._healthy_reward = healthy_reward
    self._terminate_when_unhealthy = terminate_when_unhealthy
    self._healthy_z_range = healthy_z_range
    self._reset_noise_scale = reset_noise_scale
    self._exclude_current_positions_from_observation = (
        exclude_current_positions_from_observation
    )

  def reset(self, rng: jp.ndarray) -> State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jax.random.split(rng, 3)

    low, hi = -self._reset_noise_scale, self._reset_noise_scale
    qpos = self.sys.init_q + jax.random.uniform(
        rng1, (self.sys.q_size(),), minval=low, maxval=hi
    )
    qvel = jax.random.uniform(
        rng2, (self.sys.qd_size(),), minval=low, maxval=hi
    )

    pipeline_state = self.pipeline_init(qpos, qvel)

    obs = self._get_obs(pipeline_state, jp.zeros(self.sys.act_size()))
    reward, done, zero = jp.zeros(3)
    metrics = {
        'forward_reward': zero,
        'reward_linvel': zero,
        'reward_alive': zero,
        'reward_quadctrl': zero,
        'reward_uph': zero,
        'x_position': zero,
        'y_position': zero,
        'distance_from_origin': zero,
        'x_velocity': zero,
        'y_velocity': zero,
    }
    return State(pipeline_state, obs, reward, done, metrics)

  def step(self, state: State, action: jp.ndarray) -> State:
    """Runs one timestep of the environment's dynamics."""
    pipeline_state_0 = state.pipeline_state
    pipeline_state = self.pipeline_step(pipeline_state_0, action)

    # Center of mass
    com_before, *_ = self._com(pipeline_state_0)
    com_after, *_ = self._com(pipeline_state)

    # Velocity reward
    velocity = (com_after - com_before) / self.dt
    forward_reward = self._forward_reward_weight * velocity[0]

    # Healthiness reward
    min_z, max_z = self._healthy_z_range
    is_healthy = jp.where(pipeline_state.x.pos[0, 2] < min_z, x=0.0, y=1.0)
    is_healthy = jp.where(
        pipeline_state.x.pos[0, 2] > max_z, x=0.0, y=is_healthy
    )
    if self._terminate_when_unhealthy:
      healthy_reward = self._healthy_reward
    else:
      healthy_reward = self._healthy_reward * is_healthy

    # Action Penalty
    ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

    #with jax.disable_jit():
    #  jax.debug.print("🤯 {x} 🤯", x=jp.sum(jp.square(action)))

    # Standing penalty
    pos_after = pipeline_state.x.pos[0, 2]  # z coordinate of torso
    uph_cost = self._standing_reward_weight * jp.abs(pos_after - self._standing_height_goal)

    # Final Reward
    reward =  healthy_reward + forward_reward + - uph_cost - ctrl_cost

    # Termination
    done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

    # Update metrics
    state.metrics.update(
        forward_reward=forward_reward,
        reward_linvel=forward_reward,
        reward_alive=healthy_reward,
        reward_quadctrl=-ctrl_cost,
        reward_uph=-uph_cost,
        x_position=com_after[0],
        y_position=com_after[1],
        distance_from_origin=jp.linalg.norm(com_after),
        x_velocity=velocity[0],
        y_velocity=velocity[1],
    )

    # Observations
    obs = self._get_obs(pipeline_state, action)

    return state.replace(
        pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
    )

  def _get_obs(
      self, pipeline_state: base.State, action: jp.ndarray
  ) -> jp.ndarray:
    """Observes ANYmal body position, velocities, and angles."""
    position = pipeline_state.q
    velocity = pipeline_state.qd

    if self._exclude_current_positions_from_observation:
      position = position[2:]

    com, inertia, mass_sum, x_i = self._com(pipeline_state)
    cinr = x_i.replace(pos=x_i.pos - com).vmap().do(inertia)
    com_inertia = jp.hstack(
        [cinr.i.reshape((cinr.i.shape[0], -1)), inertia.mass[:, None]]
    )

    xd_i = (
        base.Transform.create(pos=x_i.pos - pipeline_state.x.pos)
        .vmap()
        .do(pipeline_state.xd)
    )
    com_vel = inertia.mass[:, None] * xd_i.vel / mass_sum
    com_ang = xd_i.ang
    com_velocity = jp.hstack([com_vel, com_ang])

    qfrc_actuator = actuator.to_tau(self.sys, action, pipeline_state.q)

    # external_contact_forces are excluded
    return jp.concatenate([
        position,
        velocity,
        com_inertia.ravel(),
        com_velocity.ravel(),
        qfrc_actuator,
    ])

  def _com(self, pipeline_state: base.State) -> jp.ndarray:
    inertia = self.sys.link.inertia
    if self.backend in ['spring', 'positional']:
      inertia = inertia.replace(
          i=jax.vmap(jp.diag)(
              jax.vmap(jp.diagonal)(inertia.i)
              ** (1 - self.sys.spring_inertia_scale)
          ),
          mass=inertia.mass ** (1 - self.sys.spring_mass_scale),
      )
    mass_sum = jp.sum(inertia.mass)
    x_i = pipeline_state.x.vmap().do(inertia.transform)
    com = (
        jp.sum(jax.vmap(jp.multiply)(inertia.mass, x_i.pos), axis=0) / mass_sum
    )
    return com, inertia, mass_sum, x_i  # pytype: disable=bad-return-type  # jax-ndarray
