from collections import namedtuple, deque
import numpy as np
import random
import torch

Experience = namedtuple("Experience", \
                ["state", "action", "reward", "next_state", "done"])

class ReplayBuffer(object):
    def __init__(self, buffer_size):

        self.capacity = buffer_size
        self.count = 0
        self.buffer = deque(maxlen = buffer_size)

    def add_experience(self, s, a, r, s_next, done):
        
        experience = Experience(s, [a], [r], s_next, [done])

        if self.count < self.capacity:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def sample_batch(self, batch_size = 64):

        # NOTE: we take samples from the buffer to compute
        # target part of TD-error. This is done via the target
        # networks defined for Actor and Critic. Once we have the
        # target, we can calculate the TD-error by predicting
        # the action-value for the current state via actual critic
        # network. 

        if self.count < batch_size:
            samples = random.sample(self.buffer, self.count)
        else:
            samples = random.sample(self.buffer, batch_size)

        #
        s, a, r, s_next, done = [], [], [], [], []
        for sample in samples:
            s.append(sample.state)
            a.append(sample.action)
            r.append(sample.reward)
            s_next.append(sample.next_state)
            done.append(sample.done)

        return Experience(torch.tensor(s).float(), 
                    torch.tensor(a).float(),
                    torch.tensor(r).float(), 
                    torch.tensor(s_next).float(), 
                    torch.tensor(done))


