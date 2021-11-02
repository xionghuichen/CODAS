from codas.utils.config import ImgShape
from RLA.easy_log import logger
import numpy as np

EPSILON = 0.


class MappingHolder(object):
    """
    NOTE: you should create a new MappingHolder instance for each episode.
    """
    def __init__(self, stack_imgs, is_dynamics_emb, var_seq, default_rnn_state, obs_shape,
                 data_mean, data_std, adjust_allowed, init_first_state):
        self.stack_imgs = stack_imgs
        self.is_dynamics_emb = is_dynamics_emb
        self.var_seq = var_seq
        self.default_rnn_state = default_rnn_state
        self.cur_state = np.copy(self.default_rnn_state)
        self.obs_shape = obs_shape
        self.adjust_allowed = adjust_allowed
        self.last_record_idx = -1
        self.init_first_state = init_first_state
        self.data_mean = data_mean
        self.data_std = data_std
        pass

    def set_first_state(self, init_obs):
        if self.init_first_state:
            self.cur_state[:, -1 * self.obs_shape:] = init_obs

    def do_map(self, index, img_traj, ac_traj, stoc_infer):
        slice_idx = index + 1
        assert slice_idx > self.last_record_idx, "Mapping Holder should not be used in multiple episodes"
        self.last_record_idx = slice_idx
        if slice_idx < self.stack_imgs:
            pad_num = self.stack_imgs - slice_idx
            pad_zeros = np.zeros([1] + list(img_traj.shape[1:-1]) + [img_traj.shape[-1] * pad_num])
            img = np.concatenate(
                [np.expand_dims(np.concatenate(img_traj[list(reversed(range(0, slice_idx)))], axis=-1), axis=0),
                 pad_zeros], axis=-1)
        else:
            img = np.expand_dims(
                np.concatenate(img_traj[list(reversed(range(int(slice_idx - self.stack_imgs), slice_idx)))],
                               axis=-1), axis=0)
        if index == 0:
            prev_ac = np.expand_dims(np.zeros(ac_traj[0].shape), axis=0)
        else:
            prev_ac = np.expand_dims(ac_traj[index - 1], axis=0)
        r2s_ob, cur_state = self.var_seq.infer_step(img, prev_ac, prev_state=self.cur_state,
                                                    adjust_allowed=self.adjust_allowed, stoc_infer=stoc_infer)
        self.cur_state = cur_state
        # de_norm r2s
        return r2s_ob[0] * self.data_std + self.data_mean

class Runner(object):
    TOTAL_REW = 'total_rew'
    TRAJ_LEN = 'traj_len'
    OB_TRAJ = 'ob_traj'
    AC_TRAJ = 'ac_traj'
    R2S_OB_TRAJ = 'r2s_ob_traj'
    IMG_TRAJ = 'img_traj'

    def __init__(self, simulator_env, real_world_env, sim_policy, real_policy,
                 max_horizon, img_shape, clip_acs, exact_consist, action_noise_std=0.0):
        self.simulator_env = simulator_env
        self.real_world_env = real_world_env
        self.max_horizon = max_horizon
        self.img_shape = img_shape
        self.clip_acs = clip_acs
        self.sim_policy = sim_policy
        self.real_policy = real_policy
        self.exact_consist = exact_consist
        self.action_noise_std = action_noise_std
        pass

    def map_to_target(self):
        pass

    def run_traj(self, deter, mapping_holder, render_img, run_in_realworld):
        if run_in_realworld:
            select_env = self.real_world_env
            policy = self.real_policy
        else:
            select_env = self.simulator_env
            policy = self.sim_policy
        de_normalize_ob_traj = np.zeros((self.max_horizon, select_env.state_space.shape[0]), dtype=np.float64)
        de_normalize_r2s_ob_traj = np.zeros((self.max_horizon, select_env.state_space.shape[0]), dtype=np.float64)
        ac_traj = np.zeros((self.max_horizon, select_env.action_space.shape[0]), dtype=np.float32)
        img_traj = np.zeros((self.max_horizon, self.img_shape[ImgShape.WIDTH], self.img_shape[ImgShape.HEIGHT],
                             self.img_shape[ImgShape.CHANNEL]), dtype=np.float32)

        if mapping_holder is not None:
            do_mapping = True
            assert isinstance(mapping_holder, MappingHolder)
        else:
            do_mapping = False
        ob = select_env.reset()
        if do_mapping:
            mapping_holder.set_first_state(init_obs=ob)
        total_rew = 0
        i = 0
        for i in range(self.max_horizon):
            if render_img:
                img = select_env.render(mode='rgb_array', width=self.img_shape[ImgShape.WIDTH],
                                        height=self.img_shape[ImgShape.HEIGHT], camera_name='track')
                img = (img / 255.0).astype(np.float16)
                img_traj[i] = img
            de_normalize_ob_traj[i] = ob
            if do_mapping:
                assert render_img
                # do_map output a de-normalized state directly.
                r2s_state = mapping_holder.do_map(index=i, img_traj=img_traj, ac_traj=ac_traj,
                                               stoc_infer=not run_in_realworld)
                de_normalize_r2s_ob_traj[i] = r2s_state
            else:
                r2s_state = ob
            ac, *_ = policy.step(r2s_state[None, :], deterministic=deter)
            ac += np.random.normal(size=ac.shape) * self.action_noise_std
            if self.clip_acs:
                ac = np.clip(ac, select_env.action_space.low, select_env.action_space.high)
            ac_traj[i] = ac + EPSILON   # to avoid zero output.
            if self.exact_consist:
                ob, rew, done, _ = select_env.set_ob_and_step(ob, ac.squeeze())
            else:
                ob, rew, done, _ = select_env.step(ac.squeeze())

            total_rew += rew
            if done:
                break
        if i == 0:
            logger.warn("rollout a zero-step trajectory!")
        if i > 0 and np.where(np.all(de_normalize_ob_traj[np.all(ac_traj == 0, axis=-1)] != 0, axis=-1))[0].shape[0] != 0:
            nozero_idx = np.where(np.all(de_normalize_ob_traj[np.all(ac_traj == 0, axis=-1)] != 0, axis=-1))[0]
            for idx in nozero_idx:
                logger.info("----{} ---".format(idx))
                logger.info("obs:", de_normalize_ob_traj[idx])
                logger.info("acs:", ac_traj[idx])
                raise RuntimeError
        assert np.where(np.all(ac_traj[np.all(de_normalize_ob_traj == 0, axis=-1)] != 0, axis=-1))[0].shape[0] == 0
        return {self.TOTAL_REW: total_rew, self.TRAJ_LEN: i,
                self.OB_TRAJ: de_normalize_ob_traj, self.AC_TRAJ: ac_traj,
                self.R2S_OB_TRAJ: de_normalize_r2s_ob_traj, self.IMG_TRAJ: img_traj}

