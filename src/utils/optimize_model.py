import torch
import torch.nn.functional as F
from src.utils.ReplayMemory import ReplayMemory
from src.utils.ReplayMemory import Transition

'''
This code is a modified version of Wang & Hegde [2019] and the PyTorch DQN tutorial found at:
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
'''

def optimize_model(memory, optimizer, policy_net, target_net, GAMMA, BATCH_SIZE, device="cpu"):

    if len(memory) < BATCH_SIZE:
        return policy_net
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return policy_net