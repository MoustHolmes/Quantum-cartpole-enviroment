from keras.layers import Dense, Activation 
from keras.models import Sequential, load_model, model_from_json
from keras.optimizers import Adam

def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims, activation='relu'):
    model =Sequential([
            Dense(fc1_dims, input_shape=(input_dims,)), #had problems with input shape so its manualy assigned to 4 
            Activation(activation),
            Dense(fc2_dims),
            Activation(activation),
            Dense(n_actions)])
    model.compile(optimizer = Adam(lr=lr), loss = 'mse')
    return model