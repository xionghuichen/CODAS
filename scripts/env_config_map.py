env_config_map = {
    "InvertedPendulum-v2": {
        "total_timesteps": 20000,
        "policy_timestep": 200000,
        "mapping_train_epoch": 2,
    },
    "Walker2d-v4": {
        "total_timesteps": 40000,
    },
    "Hopper-v4": {
        "total_timesteps": 40000,
        "rollout_step": 25,
    },
    "HalfCheetah-v2": {
        "total_timesteps": 10000,
        # "adjust_allowed": 2.5,
    },
    "Swimmer-v4": {
        "total_timesteps": 10000,
    },
    "InvertedDouble-v5": {
        "total_timesteps": 40000,
        "policy_timestep": 200000,
        "mapping_train_epoch": 2,
    },
    "door-v0": {
        "total_timesteps": 25000,
        "max_sequence": 200,
        "trajectory_batch": 20,
        "disc_emb_hid_dim": 2048,
        "rollout_step": 1,
        "lr_dis": 0.0002,
    	"dyn_l2_loss": 0.00000002,
    },
    "pen-v0": {
        "total_timesteps": 25000,
        "max_sequence": 100,
        "lr_rescale": 0.5,
        "rollout_step": 10,
        "trajectory_batch": 20,
        "disc_emb_hid_dim": 2048,
        "lr_dis": 0.0002,
    },
    "relocate-v0": {
        "total_timesteps": 25000,
        "sim_noise": 0.3,
        "max_sequence": 200,
        "trajectory_batch": 20,
        "disc_emb_hid_dim": 2048,
        "lr_dis": 0.0002,
    },
    "hammer-v0": {
        "total_timesteps": 25000,
        "max_sequence": 200,
        "lr_rescale": 0.5,
        "trajectory_batch": 20,
        "rollout_step": 20,
        "disc_emb_hid_dim": 2048,
        "lr_dis": 0.0002,
    }
}
