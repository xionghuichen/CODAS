class WrappedPolicy:
    def __init__(self, policy, env):
        self.policy = policy
        self.env = env

    def step(self, o, deterministic=False):
        o = o.squeeze()
        assert len(o.shape) == 1
        full_state = self.env.state_to_full_state(o)
        return self.full_state_step(full_state, deterministic)

    def full_state_step(self, full_state, deterministic=False):
        if len(full_state.shape) == 1:
            full_state = full_state[None]
        mapped_o = self.env.full_state_to_obs(full_state)
        if deterministic:
            a = self.policy.get_action(mapped_o)[1]['evaluation']
        else:
            a = self.policy.get_action(mapped_o)[0]
        return a,
