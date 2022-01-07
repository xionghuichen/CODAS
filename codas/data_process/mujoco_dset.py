'''
Data structure of the input .npz:
the data is save in python dictionary format with keys: 'acs', 'ep_rets', 'rews', 'obs'
the values of each item is a list storing the expert trajectory sequentially
a transition can be: (data['obs'][t], data['acs'][t], data['obs'][t+1]) and get reward data['rews'][t]
'''

import os
import random

import numpy as np

from RLA.easy_log import logger
from codas.utils.functions import padding


class Dset(object):
    def __init__(self, inputs, inputs2, labels, labels2=None, randomize=True, label_percent=0, se_label=False, len_list=None):
        self.inputs = inputs
        self.inputs2 = inputs2
        self.labels = labels
        # added to store ac_means
        self.labels2 = labels2
        self.len_list = len_list
        assert (len(self.inputs) == len(self.labels))
        assert self.inputs2 is None or (len(self.inputs2) == len(self.labels))
        assert self.labels2 is None or (len(self.labels2) == len(self.labels))

        self.randomize = randomize
        self.num_pairs = len(inputs)
        self.mask_label_list = np.zeros([len(inputs)] + list(inputs[0].shape[0:-1]))
        if len(inputs[0].shape) == 1:
            label_idx = np.random.randint(0, len(inputs) - 1, int(len(inputs) * label_percent))
            self.mask_label_list[label_idx] = 1
        elif len(inputs[0].shape) == 2:
            for b in range(len(inputs)): # label_batch:
                label_sample = random.sample(list(range(len(inputs[0]) - 1)), int(len(inputs[0]) * label_percent))
                self.mask_label_list[b][label_sample] = 1
            if se_label:
                self.mask_label_list[:, 0] = 1
                self.mask_label_list[:, -1] = 1
        else:
            raise NotImplementedError
        self.init_pointer()

    def init_pointer(self):
        self.pointer = 0
        if self.randomize:
            idx = np.arange(self.num_pairs)
            np.random.shuffle(idx)
            self.inputs = self.inputs[idx, :]
            self.labels = self.labels[idx, :]
            if self.inputs2 is not None:
                self.inputs2 = self.inputs2[idx, :]
            if self.labels2 is not None:
                self.labels2 = self.labels2[idx, :]
            self.mask_label_list = self.mask_label_list[idx]

    def get_next_batch(self, batch_size, stack_img=False):
        # if batch_size is negative -> return all
        if batch_size < 0:
            return self.inputs, self.labels
        if self.pointer + batch_size >= self.num_pairs:
            self.init_pointer()
        end = self.pointer + batch_size
        inputs = self.inputs[self.pointer:end, :]
        len_batch = self.len_list[self.pointer:end]
        if self.inputs2 is not None:
            inputs2 = self.inputs2[self.pointer:end, :]
            if stack_img:
                sel_idx = np.random.randint(np.repeat([4], axis=0, repeats=batch_size), len_batch)
                np.concatenate([inputs2[sel_idx - 4], inputs2[sel_idx - 3], inputs2[sel_idx - 2], inputs2[sel_idx - 1]])
            inputs2 = (inputs2 / 255.0).astype(np.float16)
        else:
            inputs2 = None
        if self.labels2 is not None:
            labels2 = self.labels2[self.pointer:end, :]
            labels = self.labels[self.pointer:end, :]
            mask = self.mask_label_list[self.pointer:end]
            self.pointr = end
            return inputs, inputs2, labels, labels2, mask
        else:
            labels = self.labels[self.pointer:end, :]
            mask = self.mask_label_list[self.pointer:end]
            self.pointer = end
            return inputs, inputs2, labels, mask, len_batch

    def get_stacked_batch(self, batch_size):
        # if batch_size is negative -> return all
        if batch_size < 0:
            return self.inputs, self.labels
        if self.pointer + batch_size >= self.num_pairs:
            self.init_pointer()
        end = self.pointer + batch_size
        inputs = self.inputs[self.pointer:end, :]
        label = self.labels[self.pointer:end, :]
        inputs2 = self.inputs2[self.pointer:end, :]
        len_batch = self.len_list[self.pointer:end]
        sel_idx = np.random.randint(np.repeat([0], axis=0, repeats=batch_size), len_batch)
        row_num = np.arange(len(sel_idx))
        inputs2 = np.concatenate([inputs2[row_num, np.clip(sel_idx-3, 0, np.inf).astype(np.int32)],
                                  inputs2[row_num, np.clip(sel_idx-2, 0, np.inf).astype(np.int32)],
                                  inputs2[row_num, np.clip(sel_idx-1, 0, np.inf).astype(np.int32)],
                                  inputs2[row_num, np.clip(sel_idx, 0, np.inf).astype(np.int32)]], axis=-1)
        inputs2 = (inputs2 / 255.0).astype(np.float16)
        labels = label[row_num, sel_idx]
        if self.labels2 is not None:
            labels2 = self.labels2[row_num, sel_idx]
            mask = None
            self.pointr = end
            return inputs, inputs2, labels, labels2, mask
        else:
            mask = None
            self.pointer = end
            return inputs, inputs2, labels, mask

class Mujoco_Dset(object):
    def __init__(self, expert_path, max_sequence, env, sim_data, filter_traj,
                 data_used_fraction=1, train_fraction=0.7, traj_limitation=-1,
                 use_stack_img=False, clip_policy_bound=False, npmap_replace=False,
                 randomize=True, label_percent=0, use_trajectory=False,
                 se_label=False, clip_action=False, ):

        print('init dataset from {}\n\n'.format(expert_path))
        traj_data = np.load(expert_path, allow_pickle=True)
        self.use_stack_img = use_stack_img
        self.filter_traj = filter_traj

        if sim_data:
            acs = np.concatenate([traj_data['acs'], traj_data['train_acs']], axis=0)
            obs = np.concatenate([traj_data['obs'], traj_data['train_obs']], axis=0)
            ep_rets = np.append(traj_data['ep_rets'], traj_data['train_ep_rets'].squeeze())
        else:
            obs = traj_data['obs']
            acs = traj_data['acs']
            ep_rets = traj_data['ep_rets']

        if clip_action:
            acs = np.clip(acs, env.action_space.low, env.action_space.high)
        ac_means = None

        if 'imgs' in traj_data and not sim_data:
            self.has_img = True
        else:
            self.has_img = False
        if self.has_img:
            imgs = traj_data['imgs']
        else:
            imgs = None
        if traj_limitation > 0:
            obs = obs[:traj_limitation]
            acs = acs[:traj_limitation]
            ep_rets = ep_rets[:traj_limitation]
            if imgs is not None:
                imgs = imgs[:traj_limitation]
        self.num_traj = len(obs)
        print("rews in dataset {}".format(np.mean(ep_rets)))
        print("finished loading raw value")

        # sample sim data

        self.max_sequence = max_sequence

        # obs, acs: shape (N, L, ) + S where N = # episodes, L = episode length
        # and S is the environment observation/action space.
        # Flatten to (N * L, prod(S))
        self.len_list = []
        for ob in obs:
            nonzero_indices = ~np.all(ob == 0, axis=1)
            nonzero_indices[max_sequence + 1:] = False
            self.len_list.append(ob[nonzero_indices].shape[0] - 1)

        delta_obs = obs[:, 1:] - obs[:, :-1]
        delta_obs = delta_obs[:, :self.max_sequence - 1]
        delta_square_sum = np.square(delta_obs).sum(axis=(0, 1)) / np.sum(np.array(self.len_list) - 1)
        self.delta_obs_mean = delta_obs.sum(axis=(0, 1)) / np.sum(np.array(self.len_list) - 1)
        self.delta_obs_std = np.sqrt(delta_square_sum - np.square(self.delta_obs_mean))

        if use_trajectory:
            self.obs = padding(obs, max_length=self.max_sequence)
            self.acs = padding(acs, max_length=self.max_sequence)
            if self.has_img:
                self.imgs = padding(imgs, max_length=self.max_sequence)
            # Too-short trajectory might make the training process unstable.
            if self.filter_traj:
                len_arr = np.array(self.len_list) > int(self.max_sequence * 0.25)
                print("valid ratio of length filter: {}/{}".format(np.sum(len_arr), len(self.len_list)))
                self.len_list = list(np.array(self.len_list)[len_arr])
                self.obs = self.obs[len_arr]
                self.acs = self.acs[len_arr]
                self.imgs = self.imgs[len_arr]

            if self.use_stack_img:
                len_arr = np.array(self.len_list) > 4
                print(np.all(len_arr))
                self.len_list = list(np.array(self.len_list)[len_arr])
                self.obs = self.obs[len_arr]
                self.acs = self.acs[len_arr]
                self.imgs = self.imgs[len_arr]

            self.ac_means = padding(ac_means, max_length=self.max_sequence) if ac_means is not None else None
        else:
            # self.obs = np.concatenate(obs, axis=0)
            # self.acs = np.concatenate(acs, axis=0)
            self.obs = np.reshape(obs, [-1, int(np.prod(obs[0].shape[1:]))])
            nonzero_indices = ~np.all(self.obs==0, axis=1)
            print("nonzero data ratio: {:.2f}%".format(100*self.obs[nonzero_indices].shape[0]/self.obs.shape[0]))
            self.obs = self.obs[nonzero_indices]
            self.acs = np.reshape(acs, [-1, int(np.prod(acs[0].shape[1:]))])
            self.acs = self.acs[nonzero_indices]
            if self.has_img:
                self.imgs = np.reshape(imgs, [-1, imgs.shape[2], imgs.shape[3], imgs.shape[4]])
                self.imgs = self.imgs[nonzero_indices]
            if ac_means is not None:
                self.ac_means = np.reshape(acs, [-1, int(np.prod(acs[0].shape[1:]))])
                self.ac_means = self.ac_means[nonzero_indices]
            else:
                self.ac_means=None
        if use_trajectory:
            reduce_idx = (0,1)
        else:
            reduce_idx = (0)
        self.rets = ep_rets[:traj_limitation]
        self.avg_ret = sum(self.rets)/len(self.rets)
        self.std_ret = np.std(np.array(self.rets))
        # if len(self.acs) > 2:
        #     self.acs = np.squeeze(self.acs)
        assert len(self.obs) == len(self.acs)

        self.num_transition = len(self.obs)
        self.randomize = randomize

        def do_npmap_replace(data, name):
            if data is None:
                return data
            if sim_data:
                post_str = "-sim"
            else:
                post_str = ""

            if use_trajectory:
                npmap_filename = '.'.join(expert_path.split('.')[:-1]) + '-{}-{}{}.npmap'.format(name, self.max_sequence, post_str)
            else:
                npmap_filename = '.'.join(expert_path.split('.')[:-1]) + '-{}-{}-no-traj{}.npmap'.format(name, self.max_sequence, post_str)
            #write nmap file to /tmp and use local cache
            npmap_filename = npmap_filename.split("/")[-1]
            npmap_filename = os.path.join('/tmp', npmap_filename)
            if os.path.exists(npmap_filename):
                os.remove(npmap_filename)
            fp = np.memmap(npmap_filename, dtype=data.dtype, mode='w+', shape=data.shape)
            fp[:] = data
            return fp
        print("compelte padding")
        if npmap_replace:
            self.obs = do_npmap_replace(self.obs, 'obs')
            self.acs = do_npmap_replace(self.acs, 'acs')
        self.obs_max, self.obs_min = np.max(self.obs, axis=reduce_idx), np.min(self.obs, axis=reduce_idx)
        self.acs_max, self.acs_min = np.max(self.acs, axis=reduce_idx), np.min(self.acs, axis=reduce_idx)
        self.obs_mean, self.obs_std = np.mean(self.obs, axis=reduce_idx), np.std(self.obs, axis=reduce_idx)
        if self.has_img:
            if npmap_replace:
                self.imgs = do_npmap_replace(self.imgs, 'imgs')
            self.dset = Dset(self.obs[:int(self.num_transition * data_used_fraction), :],
                             self.imgs[:int(self.num_transition * data_used_fraction), :, :, :],
                             self.acs[:int(self.num_transition * data_used_fraction), :],
                             self.ac_means[:int(self.num_transition * data_used_fraction), :] if self.ac_means is not None else None,
                             self.randomize, label_percent, se_label,
                             np.array(self.len_list))
        else:
            self.dset = Dset(self.obs[:int(self.num_transition * data_used_fraction), :], None,
                             self.acs[:int(self.num_transition * data_used_fraction), :],
                             self.ac_means[:int(self.num_transition * data_used_fraction), :] if self.ac_means is not None else None,
                             self.randomize, label_percent, se_label,
                             np.array(self.len_list))

        self.log_info()

    def log_info(self):
        logger.log("Total trajectorues: %d" % self.num_traj)
        logger.log("Total transitions: %d" % self.num_transition)
        logger.log("Average returns: %f" % self.avg_ret)
        logger.log("Std for returns: %f" % self.std_ret)
        logger.log("obs in ({}, {})".format(self.obs_min, self.obs_max))
        logger.log("acs in ({}, {})".format(self.acs_min, self.acs_max))

    def get_next_batch(self, batch_size, stack_img=False, split=None):
        if split is None:
            if stack_img:
                return self.dset.get_stacked_batch(batch_size)
            else:
                return self.dset.get_next_batch(batch_size, stack_img)
        else:
            raise NotImplementedError

    def plot(self):
        import matplotlib.pyplot as plt
        plt.hist(self.rets)
        plt.savefig("histogram_rets.png")
        plt.close()


def test(expert_path, traj_limitation, plot, label_percent):
    dset = Mujoco_Dset(expert_path, traj_limitation=traj_limitation, label_percent=label_percent)
    if plot:
        dset.plot()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_path", type=str, default="../data/deterministic.trpo.Hopper.0.00.npz")
    parser.add_argument("--traj_limitation", type=int, default=-1)
    parser.add_argument("--label_percent", type=int, default=0.5)
    parser.add_argument("--plot", type=bool, default=False)
    args = parser.parse_args()
    test(args.expert_path, args.traj_limitation, args.plot, args.label_percent)
    print("end")
