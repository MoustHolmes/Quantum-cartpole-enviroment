import numpy as np

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discreate=False):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.discreate =discreate # this defines if the avaible actions are disctreate or continious
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discreate else np.float32
        
        self.action_memory = np.zeros((self.mem_size, n_actions),dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)# explain
        
    def store_transition(self,state, action, reward, state_, done):
        index =self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        
        self.terminal_memory[index] = 1-int(done)#explain
        
        if self.discreate:
            actions =  np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.mem_cntr += 1
        
    def sample_buffer( self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]
        
        return states, actions, rewards, states_, terminal
