import torch
import math

def select_action(state, policy_net, m, action_rng, EPS_START, EPS_END, EPS_DECAY, STEPS_DONE):
    # sample = random.random()
    # instead of python's random we'll use numpy's to be able to have a generator with its own seed
    sample = action_rng.uniform()
    eps_threshold =EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * STEPS_DONE / EPS_DECAY)
    STEPS_DONE += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest value for column of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1), STEPS_DONE
    else:
        return torch.tensor([[action_rng.randint(m)]], dtype=torch.long), STEPS_DONE