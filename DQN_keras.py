from keras.layers import Dense, Activation 
from keras.models import Sequential, load_model, model_from_json
from keras.optimizers import Adam


import numpy as np

from ReplayBuffer import ReplayBuffer
from Build_DQN_net_keras import build_dqn

class DQN_Agent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size, input_dims,
                 activation='relu',
                 n_dense1=32, n_dense2=32,
                epsilon_dec = 0.9995, epsilon_end=0.05,
                mem_size =1000000, fname='dqn_model.h5'):
        
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon =epsilon
        self.activation=activation
        self.epsilon_dec =epsilon_dec
        self.epsilon_end = epsilon_end
        self.batch_size = batch_size 
        self.model_file = 'dqn_model_'+str(n_dense1)+'_'+str(n_dense2)+'.h5'
        self.alpha = alpha
        self.n_dense1 = n_dense1
        self.n_dense2 = n_dense2
        
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, discreate=True)
        
        self.q_eval = build_dqn(alpha, n_actions, input_dims, n_dense1, n_dense2, activation=self.activation)
        
    def remember(self, state, action, reward , new_state, done):
        self.memory.store_transition(state, action, reward , new_state, done)
        
    def choose_action(self, state):
        state = state [np.newaxis,:]
        rand = np.random.random()
        greedy = True
        
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
            greedy = False
            actions = self.q_eval.predict(state)#this is here only for a test
            
        else: 
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)
            
        return action, actions, greedy
    
    def choose_action_max(self, state):
        state = state [np.newaxis,:]
        actions = self.q_eval.predict(state)
        action = np.argmax(actions)
            
        return action
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done =self.memory.sample_buffer(self.batch_size)
        
        
        action_values = np.array(self.action_space, dtype=np.int8)
        action_indices = np.dot(action, action_values)
        
        q_eval = self.q_eval.predict(state)
        q_next = self.q_eval.predict(new_state)
    
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype =np.int32)
        
        q_target[batch_index, action_indices] = reward + self.gamma *np.max(q_next, axis=1) *done
        
        hist =self.q_eval.fit(state, q_target, verbose=0)
        
        loss = hist.history['loss'][0]
        self.epsilon = self.epsilon *self.epsilon_dec if self.epsilon > self.epsilon_end else self.epsilon_end
        
        return loss
        
    def learn_simple(self, state, action, reward , new_state, done):
        
        action_values = np.array(self.action_space, dtype=np.int8)
        action_indices = np.dot(action, action_values)
       
        q_eval = self.q_eval.predict(state[np.newaxis,:])
        q_next = self.q_eval.predict(new_state[np.newaxis,:])
    
        q_target = q_eval.copy()
        
        q_target[0, action_indices] = reward + self.gamma *np.max(q_next, axis=1) *done
        
        
        _ =self.q_eval.fit(state[np.newaxis,:], q_target, verbose=0)
        
        # decreace the epsilon value 
        self.epsilon = self.epsilon *self.epsilon_dec if self.epsilon > self.epsilon_end else self.epsilon_end
        
    
    def save_model(self,model_file):
        
        self.model_file = model_file
        self.q_eval.save(self.model_file)
        
    def save_model_JSON(self, model_file):
        self.model_file= model_file
        model_json = self.q_eval.to_json()
        with open(str(model_file)+'.json', "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.q_eval.save_weights(str(model_file)+'.h5')
        
    def load_model(self, model_file):
        self.model_file = model_file
        self.q_eval.load_model(self.model_file, compile=False)
        
    def load_model_JSON(self, model_file):
        self.model_file= model_file
        json_file = open(str(model_file)+'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.q_eval = model_from_json(loaded_model_json)
        # load weights into new model
        self.q_eval.load_weights(str(model_file)+'.h5')
        self.q_eval.compile(optimizer = Adam(lr=self.alpha), loss = 'mse')
    
    def print_model(self):
        print(self.q_eval.to_json(indent=4))
        
    def print_weights(self):
        print(self.q_eval.model.get_weights())
