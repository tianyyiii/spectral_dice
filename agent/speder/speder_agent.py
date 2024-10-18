import tensorflow as tf
from tensorflow.keras import layers, models

class Theta(tf.keras.Model):
    def __init__(self, feature_dim=1024):
        super(Theta, self).__init__()
        self.l = layers.Dense(1, input_shape=(feature_dim,))

    def call(self, feature):
        r = self.l(feature)
        return r

class SPEDERAgent():
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 feature_optimizer,
                 phi_hidden_dim=-1, 
                 phi_hidden_depth=-1, 
                 mu_hidden_dim=-1, 
                 mu_hidden_depth=-1, 
                 feature_dim=2048):
        
        self.feature_dim = feature_dim

        activation_fn = tf.nn.relu
        kernel_initializer = tf.keras.initializers.GlorotUniform()

        self.phi = tf.keras.Sequential(name='mlp')
        if phi_hidden_depth == 0:
            self.phi.add(layers.Dense(
                  input_dim = state_dim + action_dim,
                  units=feature_dim, 
                  activation=activation_fn, 
                  kernel_initializer=kernel_initializer
                  ))      
        else:
            for i in range(phi_hidden_depth):
                if i == 0:
                    self.phi.add(layers.Dense(
                        input_dim=state_dim + action_dim,
                        units=phi_hidden_dim, 
                        activation=activation_fn, 
                        kernel_initializer=kernel_initializer
                        ))
                else:
                    self.phi.add(layers.Dense(
                        units=phi_hidden_dim, 
                        activation=activation_fn, 
                        kernel_initializer=kernel_initializer
                        ))                   
            self.phi.add(layers.Dense(
                units=feature_dim, 
                activation=activation_fn, 
                kernel_initializer=kernel_initializer
                ))
            
        self.mu = tf.keras.Sequential(name='mlp')
        if mu_hidden_depth == 0:
            self.mu.add(layers.Dense(
                  input_dim = state_dim + action_dim,
                  units=feature_dim, 
                  activation=activation_fn, 
                  kernel_initializer=kernel_initializer
                  ))      
        else:
            for i in range(mu_hidden_depth):
                if i == 0:
                    self.mu.add(layers.Dense(
                        input_dim=state_dim + action_dim,
                        units=mu_hidden_dim, 
                        activation=activation_fn, 
                        kernel_initializer=kernel_initializer
                        ))
                else:
                    self.mu.add(layers.Dense(
                        units=mu_hidden_dim, 
                        activation=activation_fn, 
                        kernel_initializer=kernel_initializer
                        ))                   
            self.mu.add(layers.Dense(
                units=feature_dim, 
                activation=activation_fn, 
                kernel_initializer=kernel_initializer
                ))


        self.theta = Theta(feature_dim=feature_dim)

        self.feature_optimizer = feature_optimizer

    def feature_step(self, state, action, next_state, next_action, s_random, a_random, s_prime_random, a_prime_random, reward, update=True):
        state = tf.cast(state, tf.float32)
        action = tf.cast(action, tf.float32)
        next_state = tf.cast(next_state, tf.float32)
        s_random = tf.cast(s_random, tf.float32)
        a_random = tf.cast(a_random, tf.float32)
        s_prime_random = tf.cast(s_prime_random, tf.float32)
        next_action = tf.cast(next_action, tf.float32)
        a_prime_random = tf.cast(a_prime_random, tf.float32)


        with tf.GradientTape() as tape:
            z_phi = self.phi(tf.concat([state, action], axis=-1))
            z_phi_random = self.phi(tf.concat([s_random, a_random], axis=-1))

            z_mu_next = self.mu(tf.concat([next_state, next_action], axis=-1))
            z_mu_next_random = self.mu(tf.concat([s_prime_random, a_prime_random], axis=-1))

            assert z_phi.shape[-1] == self.feature_dim
            assert z_mu_next.shape[-1] == self.feature_dim

            model_loss_pt1 = -2 * tf.reduce_sum(tf.multiply(z_phi, z_mu_next), axis=1, keepdims=True)
            model_loss_pt2_a= tf.reduce_sum(tf.multiply(z_phi_random, z_mu_next_random), axis=1, keepdims=True)
            model_loss_pt2 = tf.multiply(model_loss_pt2_a, model_loss_pt2_a)

            model_loss_pt1_summed = tf.reduce_sum(model_loss_pt1) / tf.size(model_loss_pt1, out_type=tf.float32)
            model_loss_pt2_summed = tf.reduce_sum(model_loss_pt2) / tf.size(model_loss_pt2, out_type=tf.float32)

            model_loss = model_loss_pt1_summed + model_loss_pt2_summed

            r_loss = 0.5 * tf.reduce_mean(tf.square(self.theta(z_phi) - reward))

            loss = model_loss + r_loss
        
        if update:
            grads = tape.gradient(loss, self.phi.trainable_variables + self.mu.trainable_variables + self.theta.trainable_variables)
            self.feature_optimizer.apply_gradients(zip(grads, self.phi.trainable_variables + self.mu.trainable_variables + self.theta.trainable_variables))

        return {
            'total_loss': loss.numpy(),
            'model_loss': model_loss.numpy(),
        }
       
    def final_info(self):
        return self.phi, self.mu

