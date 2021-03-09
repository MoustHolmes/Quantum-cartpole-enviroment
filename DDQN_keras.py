from keras.layers import Dense, Activation 
from keras.models import Sequential, load_model, model_from_json
from keras.optimizers import Adam

import numpy as np

class DDQN_Agent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_dec=0.996,  epsilon_end=0.01,
                 n_dense1=32, n_dense2=32,
                 mem_size=1000000, fname='ddqn_model.h5', replace_target=100):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        
        
        
        self.n_dense1 = n_dense1
        self.n_dense2 = n_dense2
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.replace_target = replace_target
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions,discreate=True)
        
        self.q_eval = build_dqn(alpha, n_actions, input_dims, n_dense1, n_dense2)
        self.q_target = build_dqn(alpha, n_actions, input_dims, n_dense1, n_dense2)

        self.model_file = 'ddqn_model_'+str(n_dense1)+'_'+str(n_dense2)+'.h5'
        
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)

        return action

    def learn(self):
        if self.memory.mem_cntr > self.batch_size:
            state, action, reward, new_state, done = \
                                          self.memory.sample_buffer(self.batch_size)

            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(action, action_values)

            q_next = self.q_target.predict(new_state)
            q_eval = self.q_eval.predict(new_state)
            q_pred = self.q_eval.predict(state)

            max_actions = np.argmax(q_eval, axis=1)

            q_target = q_pred

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            q_target[batch_index, action_indices] = reward + self.gamma*q_next[batch_index, max_actions.astype(int)]*done

            hist = self.q_eval.fit(state, q_target, verbose=0)
            loss = hist.history['loss'][0]
            
            
            self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min
            
            if self.memory.mem_cntr % self.replace_target == 0:
                self.update_network_parameters()
            
            return loss
        
    def update_network_parameters(self):
        self.q_target.model.set_weights(self.q_eval.model.get_weights())

    def save_model(self):
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