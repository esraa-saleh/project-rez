import random
import math
import torch

#takes in a state, the network (called policy_net in Nidhi's code), and m (m = env.action_space.n)
def select_action(state, net, m):
    steps_done = 0
    eps_start = 0.9
    eps_end = 0.05
    eps_decay = 200
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * \
                    math.exp(-1. * steps_done/ eps_decay)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest value for column of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(m)]], dtype=torch.long)