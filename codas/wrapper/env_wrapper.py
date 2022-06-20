import gym
import numpy as np
import stable_baselines
import cv2
import os
from RLA.easy_log import logger
from stable_baselines.bench import Monitor
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env.vec_normalize import VecNormalize

def make_vec_env_unit(env_id, n_envs=1, seed=None, start_index=0,
                 monitor_dir=None, wrapper_class=None,
                 env_kwargs=None, vec_env_cls=None, vec_env_kwargs=None):
    """
    Create a wrapped, monitored `VecEnv`.
    By default it uses a `DummyVecEnv` which is usually faster
    than a `SubprocVecEnv`.

    :param env_id: (str or Type[gym.Env]) the environment ID or the environment class
    :param n_envs: (int) the number of environments you wish to have in parallel
    :param seed: (int) the initial seed for the random number generator
    :param start_index: (int) start rank index
    :param monitor_dir: (str) Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param wrapper_class: (gym.Wrapper or callable) Additional wrapper to use on the environment.
        This can also be a function with single argument that wraps the environment in many things.
    :param env_kwargs: (dict) Optional keyword argument to pass to the env constructor
    :param vec_env_cls: (Type[VecEnv]) A custom `VecEnv` class constructor. Default: None.
    :param vec_env_kwargs: (dict) Keyword arguments to pass to the `VecEnv` class constructor.
    :return: (VecEnv) The wrapped environment
    """
    env_kwargs = {} if env_kwargs is None else env_kwargs
    vec_env_kwargs = {} if vec_env_kwargs is None else vec_env_kwargs

    def make_env(rank):
        def _init():
            if isinstance(env_id, str):
                env = gym.make(env_id, **env_kwargs)

            else:
                env = env_id(**env_kwargs)
            if seed is not None:
                env.seed(seed + rank)
                env.action_space.seed(seed + rank)
            # Wrap the env in a Monitor wrapper
            # to have additional training information
            monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            # Create the monitor folder if needed
            if monitor_path is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path)
            # Optionally, wrap the environment with the provided wrapper
            if wrapper_class is not None:
                env = wrapper_class(env)
            return env
        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnv

    return vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)


def make_vec_env(env_id, num_env, dynamic_param=None, stoc_init_range=None):
    if env_id == "Ant-v3":
        env = make_vec_env_unit(env_id, n_envs=num_env,
                                env_kwargs={"exclude_current_positions_from_observation": False})
    elif env_id == 'Hopper-v4':
        env = make_vec_env_unit(env_id, n_envs=num_env, env_kwargs={'friction': dynamic_param,
                                                                    "stoc_init_range": stoc_init_range})
    else:
        env = make_vec_env_unit(env_id, n_envs=num_env)
    return env


class GeneratorWrapper(object):
    def __init__(self, env, use_image_noise=False, alpha=1.0, normalize=True):
        self.use_image_noise = use_image_noise
        self.elaspsed_time = 0
        if self.use_image_noise:
            noise_img = np.zeros([64, 64, 3], dtype=np.uint8)
            color = (255, 0, 0)
            noise_img = cv2.circle(noise_img, (10, 10), 3, color, -1)
            self.noise_img, self.x_mv_fn, self.y_mv_fn = (noise_img, lambda _: 0.1*(_%10), lambda _: 0.5*(_%20))
            self.alpha = alpha
        else:
            self.noise_img = self.alpha = self.x_mv_fn = self.y_mv_fn = None

        self.normalize = normalize
        if isinstance(env, VecNormalize):
            self.env = env.venv.envs[0]
            if env.norm_obs:
                self.mean = np.copy(env.obs_rms.mean)
                self.std = np.sqrt(env.obs_rms.var + env.epsilon)
            else:
                self.mean = 0
                self.std = 1
            logger.info("inner mean {}, std {}".format(self.mean, self.std))
        else:
            self.env = env
            self.mean = 0
            self.std = 1
        self.full_state_to_obs_fn = getattr(self.env.env, 'full_state_to_obs', lambda x: x)
        self.full_state_to_state_fn = getattr(self.env.env, 'full_state_to_state', lambda x: x)

        self.state_space = getattr(self.env.env, 'state_space', self.env.env.observation_space)
        self.full_state_space = getattr(self.env.env, 'full_state_space', self.env.env.observation_space)

        from mj_envs.hand_manipulation_suite.door_v0 import DoorEnvV0
        from mj_envs.hand_manipulation_suite.relocate_v0 import RelocateEnvV0
        from mj_envs.hand_manipulation_suite.pen_v0 import PenEnvV0
        from mj_envs.hand_manipulation_suite.hammer_v0 import HammerEnvV0
        if isinstance(self.env.env, DoorEnvV0) or isinstance(self.env.env, RelocateEnvV0) \
           or isinstance(self.env.env, PenEnvV0) or isinstance(self.env.env, HammerEnvV0):
            self.is_dapg_env = True
        else:
            self.is_dapg_env = False

    def _set_ob(self, ob):
        raw_env = self.env
        from codas.reset_able_mj_env.swimmer_v4 import SwimmerEnv
        from codas.reset_able_mj_env.inverted_double_pendulum_v5 import InvertedDoublePendulumEnv as InvertedDoublePendulumEnvv5
        from codas.reset_able_mj_env.ant_v3 import AntEnv
        from gym.envs.mujoco import InvertedPendulumEnv

        # no VecNormalize for door-v0 and relocate-v0
        if self.is_dapg_env:
            self.env.env.set_env_full_state(ob)
            return
        elif isinstance(self.env.env.env, AntEnv):
            qpos = ob[:15]
            qvel = ob[15:29]
        elif isinstance(self.env.env.env, SwimmerEnv):
            qpos = ob[:raw_env.model.nq - 2]
            qpos = np.concatenate(([0.0, 0.0], qpos))
            qvel = ob[raw_env.model.nq - 2:-1]
        elif isinstance(self.env.env.env, InvertedPendulumEnv):
            qpos = ob[:raw_env.model.nq]
            qvel = ob[raw_env.model.nq:]
        elif isinstance(self.env.env.env, InvertedDoublePendulumEnvv5):
            qpos1 = ob[0:1]
            qpos2 = np.arcsin(ob[1:raw_env.model.nq])
            qpos = np.concatenate([qpos1, qpos2])
            qvel = ob[raw_env.model.nq + raw_env.model.nq - 1:raw_env.model.nq + raw_env.model.nq - 1 + raw_env.model.nv]
        else:
            qpos = ob[:raw_env.model.nq - 1]
            qpos = np.concatenate(([0.0], qpos))
            qvel = ob[raw_env.model.nq - 1:]
        raw_env.set_state(qpos, qvel)

    def reset(self):
        self.elaspsed_time = 0
        raw_ob = self.env.reset()
        self.env.sim.data.qacc_warmstart[:] = 0
        if self.normalize:
            ob = (raw_ob - self.mean) / self.std
        else:
            ob = raw_ob
        return ob

    def state_to_full_state(self, state):
        if self.is_dapg_env:
            original_full_state = self.env.env.get_env_full_state()
            self.env.env.set_env_state(state)
            full_state = self.env.env.get_env_full_state()
            self.env.env.set_env_full_state(original_full_state)
            return full_state
        else:
            return state

    def step(self, acs):
        clipped_actions = np.clip(acs, self.env.action_space.low, self.env.action_space.high)
        self.env.sim.data.qacc_warmstart[:] = 0
        raw_ob, reward, done, info = self.env.step(clipped_actions)

        if self.normalize:
            ob = (raw_ob - self.mean) / self.std
        else:
            ob = raw_ob
        # ob = ob * self.std + self.mean
        self.elaspsed_time += 1
        return ob, reward, done, info

    def set_ob_and_step(self, ob, act, ret_full_state=False):
        self.reset()
        if self.normalize:
            raw_ob = ob * self.std + self.mean
        else:
            raw_ob = ob
        self._set_ob(raw_ob)
        if ret_full_state:
            full_state = self.env.env.get_env_full_state()
        else:
            full_state = None
        ob, reward, done, _ = self.step(act)
        return ob, reward, done, {"full_state": full_state}

    def set_ob(self, ob):
        self.reset()
        if self.normalize:
            raw_ob = ob * self.std + self.mean
        else:
            raw_ob = ob
        self._set_ob(raw_ob)

    def full_state_to_obs(self, full_states):
        ori_shape = list(full_states.shape)[0:-1]
        full_states = full_states.reshape([-1, full_states.shape[-1]])
        obs = self.full_state_to_obs_fn(full_states)
        return obs.reshape(ori_shape + [-1])

    def full_state_to_state(self, full_states):
        ori_shape = list(full_states.shape)[0:-1]
        full_states = full_states.reshape([-1, full_states.shape[-1]])
        states = self.full_state_to_state_fn(full_states)
        return states.reshape(ori_shape + [-1])

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def render(self, *args, **kwargs):
        if not self.use_image_noise:
            return self.env.render(*args, **kwargs)
        else:
            original_image = self.env.render(*args, **kwargs)
            rows, cols = original_image.shape[0], original_image.shape[1]
            noise_img = self.noise_img.copy()
            x_mv = int(cols * self.x_mv_fn(self.elaspsed_time))
            y_mv = int(rows * self.y_mv_fn(self.elaspsed_time))
            movement = np.float32([[1, 0, x_mv], [0, 1, y_mv]])
            noise = cv2.warpAffine(noise_img, movement, (cols, rows))
            image = original_image
            noise_idx = np.where(noise > 0)
            image[noise_idx[0],noise_idx[1]] = \
                noise[noise_idx[0],noise_idx[1]] * self.alpha + \
                image[noise_idx[0],noise_idx[1]] * (1 - self.alpha)
            return image.astype(np.uint8)


def is_dapg_env(env_id: str):
    dapg_env_prefixes = [
        'relocate',
        'door',
        'hammer',
        'pen'
        ]
    env_id_prefix = env_id.split("-")[0]
    if any([e == env_id_prefix for e in dapg_env_prefixes]):
        return True
    else:
        return False
