import torch
import math

'''
This code is a modified version of Wang & Hegde [2019] and the PyTorch DQN tutorial found at:
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
'''

def select_action(state, policy_net, m, action_rng, EPS_START, EPS_END, EPS_DECAY, STEPS_DONE):
    sample = action_rng.uniform()
    eps_threshold =EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * STEPS_DONE / EPS_DECAY)
    STEPS_DONE += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1), STEPS_DONE
    else:
        return torch.tensor([[action_rng.randint(m)]], dtype=torch.long), STEPS_DONE