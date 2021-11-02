import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os


class SwimmerEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/fixed_swimmer.xml' % dir_path, 4)
        utils.EzPickle.__init__(self)

    def step(self, action):
        return self._step(action)

    def _step(self, a):
        ctrl_cost_coeff = 0.0001

        """
        xposbefore = self.model.data.qpos[0, 0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.model.data.qpos[0, 0]
        """
        xy_position_before = self.sim.data.qpos[0:2].copy()
        self.do_simulation(a, self.frame_skip)
        xy_position_after = self.sim.data.qpos[0:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        forward_reward = x_velocity

        ctrl_cost = ctrl_cost_coeff * np.sum(np.square(a))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        return observation, reward, False, dict(reward_fwd=forward_reward, reward_ctrl=ctrl_cost)

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat[2:], qvel.flat])

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        self.xposbefore = self.xposafter = self.sim.data.site_xpos[0][0]
        return self._get_obs()