import copy
import zinc_grammar as G
import tensorflow as tf


masks_K      = tf.Variable(G.masks)
ind_of_ind_K = tf.Variable(G.ind_of_ind)

MAX_LEN = 277
DIM = G.D


class MoleculeVAE():

    autoencoder = None
    
    def create(self,
               charset,
               max_length = MAX_LEN,
               latent_rep_size = 2,
               hypers = {'hidden': 501, 'dense': 435, 'conv1': 9, 'conv2': 9, 'conv3': 10},
               weights_file = None):
        charset_length = len(charset)
        #charset_length = charset
        self.hypers = hypers
        
        x = tf.keras.Input(shape=(max_length, charset_length))
        _, z = self._buildEncoder(x, latent_rep_size, max_length)
        self.encoder = tf.keras.Model(x, z)

        encoded_input = tf.keras.Input(shape=(latent_rep_size,))
        self.decoder = tf.keras.Model(
            encoded_input,
            self._buildDecoder(
                encoded_input,
                latent_rep_size,
                max_length,
                charset_length
            )
        )

        x1 = tf.keras.Input(shape=(max_length, charset_length))
        vae_loss, z1 = self._buildEncoder(x1, latent_rep_size, max_length)
        self.autoencoder = tf.keras.Model(
            x1,
            self._buildDecoder(
                z1,
                latent_rep_size,
                max_length,
                charset_length
            )
        )


        x2 = tf.keras.Input(shape=(max_length, charset_length))
        (z_m, z_l_v) = self._encoderMeanVar(x2, latent_rep_size, max_length)
        self.encoderMV = tf.keras.Model(inputs=x2, outputs=[z_m, z_l_v])

        if weights_file:
            self.autoencoder.load_weights(weights_file)
            self.encoder.load_weights(weights_file, by_name = True)
            self.decoder.load_weights(weights_file, by_name = True)
            self.encoderMV.load_weights(weights_file, by_name = True)


        self.autoencoder.compile(optimizer = 'Adam',
                                 loss = vae_loss,
                                 metrics = ['accuracy'])


    def _encoderMeanVar(self, x, latent_rep_size, max_length, epsilon_std = 0.01):
        
        
        
        h = tf.keras.layers.Conv1D(filters=self.hypers['conv1'], kernel_size=self.hypers['conv1'], strides=1, activation = 'relu', use_bias=True, padding='same')(x)
        h = tf.keras.layers.BatchNormalization(momentum=0.997, epsilon=1e-5, trainable=True)(h)
        h = tf.keras.layers.Conv1D(filters=self.hypers['conv2'], kernel_size=self.hypers['conv2'], strides=1, activation = 'relu', use_bias=True, padding='same')(h)
        h = tf.keras.layers.BatchNormalization(momentum=0.997, epsilon=1e-5, trainable=True)(h)
        h = tf.keras.layers.Conv1D( filters=self.hypers['conv3'], kernel_size=self.hypers['conv3']+1, strides=1, activation = 'relu', use_bias=True, padding='same')(h)
        h = tf.keras.layers.BatchNormalization(momentum=0.997, epsilon=1e-5, trainable=True)(h)
        h = tf.keras.layers.Flatten()(h)
        h = tf.keras.layers.Dense(units=self.hypers['dense'], activation = 'relu')(h)
        
        z_mean = tf.keras.layers.Dense(units=latent_rep_size, activation = 'linear')(h)
        z_log_var = tf.keras.layers.Dense(units=latent_rep_size, activation = 'linear')(h)
        

        return (z_mean, z_log_var) 

    def _buildEncoder(self, x, latent_rep_size, max_length, epsilon_std = 0.01):
        
        
        h = tf.keras.layers.Conv1D(filters=self.hypers['conv1'], kernel_size=self.hypers['conv1'], strides=1, activation = 'relu', use_bias=True, padding='same')(x)
        h = tf.keras.layers.BatchNormalization(momentum=0.997, epsilon=1e-5, trainable=True)(h)
        h = tf.keras.layers.Conv1D(filters=self.hypers['conv2'], kernel_size=self.hypers['conv2'], strides=1, activation = 'relu', use_bias=True, padding='same')(h)
        h = tf.keras.layers.BatchNormalization(momentum=0.997, epsilon=1e-5, trainable=True)(h)
        h = tf.keras.layers.Conv1D(filters=self.hypers['conv3'], kernel_size=self.hypers['conv3']+1, strides=1, activation = 'relu', use_bias=True, padding='same')(h)
        h = tf.keras.layers.BatchNormalization(momentum=0.997, epsilon=1e-5, trainable=True)(h)
        h = tf.keras.layers.Flatten()(h)
        h = tf.keras.layers.Dense(units=self.hypers['dense'], activation = 'relu')(h)
        

        def sampling(args):
            z_mean_, z_log_var_ = args
            
            batch_size = tf.shape(z_mean_)[0]
            epsilon = tf.random.normal(shape=(batch_size, latent_rep_size), mean=0., stddev = epsilon_std)
            
            return z_mean_ + tf.exp(z_log_var_ / 2) * epsilon

        z_mean = tf.keras.layers.Dense(units=latent_rep_size, activation = 'linear')(h)
        z_log_var = tf.keras.layers.Dense(units=latent_rep_size, activation = 'linear')(h)

        def conditional(x_true, x_pred):
            most_likely = tf.math.argmax(x_true)
            
            most_likely = tf.reshape(most_likely,[-1]) # flatten most_likely
            ix2 = tf.expand_dims(tf.gather(ind_of_ind_K, most_likely),1) # index ind_of_ind with res
            ix2 = tf.cast(ix2, tf.int32) # cast indices as ints 
            M2 = tf.gather_nd(masks_K, ix2) # get slices of masks_K with indices
            M3 = tf.reshape(M2, [-1,MAX_LEN,DIM]) # reshape them
            P2 = tf.math.multiply(tf.math.exp(x_pred),tf.cast(M3,tf.float32)) # apply them to the exp-predictions
            P2 = tf.math.divide(P2,tf.math.reduce_sum(P2,axis=-1,keepdims=True)) # normalize predictions
            return P2

        def vae_loss(x, x_decoded_mean):
            #x_decoded_mean = conditional(x, x_decoded_mean)
            conditional = ConditionalLayer()
            x_decoded_mean = conditional([x, x_decoded_mean])
            #print("test1")
            #print(tf.shape(x))
            x = tf.keras.layers.Flatten()(x)
            #print("test2")
            #print(tf.shape(x_decoded_mean))
            x_decoded_mean = tf.keras.layers.Flatten()(x_decoded_mean)
            
            xent_loss = max_length * tf.keras.losses.binary_crossentropy(x, x_decoded_mean)
            
            kl_loss = - 0.5 * tf.reduce_mean(1 + z_log_var - tf.math.square(z_mean) - tf.math.exp(z_log_var), axis = -1)
            kl_loss = - 0.5 * tf.reduce_mean(1 + z_log_var - tf.math.square(z_mean) - tf.math.exp(z_log_var), axis = -1)
            
            return xent_loss + kl_loss

        return (vae_loss, tf.keras.layers.Lambda(sampling, output_shape=(latent_rep_size,), name='lambda')([z_mean, z_log_var]))

    def _buildDecoder(self, z, latent_rep_size, max_length, charset_length):
        
        h = tf.keras.layers.BatchNormalization(momentum=0.997, epsilon=1e-5, trainable=True)(z)
        h = tf.keras.layers.Dense(units=latent_rep_size, activation = 'relu')(h)
        h = tf.keras.layers.RepeatVector(max_length)(h)
        gru1 = tf.keras.layers.GRU(units=self.hypers['hidden'], return_sequences = True)
        h = gru1(h)
        gru2 = tf.keras.layers.GRU(units=self.hypers['hidden'], return_sequences = True)
        h = gru2(h)
        gru3 = tf.keras.layers.GRU(units=self.hypers['hidden'], return_sequences = True)
        h = gru3(h)
        

        return tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=charset_length))(h)

    def save(self, filename):
        self.autoencoder.save_weights(filename)
    
    def load(self, charset, weights_file, latent_rep_size = 2, max_length=MAX_LEN, hypers = {'hidden': 501, 'dense': 435, 'conv1': 9, 'conv2': 9, 'conv3': 10}):
        self.create(charset, max_length = max_length, weights_file = weights_file, latent_rep_size = latent_rep_size, hypers = hypers)

class ConditionalLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(ConditionalLayer, self).__init__(name='conditional_layer')
        self.masks_K = tf.Variable(G.masks)
        self.ind_of_ind_K = tf.Variable(G.ind_of_ind)
    
    

    def call(self, inputs):
        x_true = inputs[0]
        x_pred = inputs[1]
        most_likely = tf.math.argmax(x_true,-1)
        most_likely = tf.reshape(most_likely,[-1])
        ix2 = tf.expand_dims(tf.gather(self.ind_of_ind_K, most_likely),1) # index ind_of_ind with res
        ix2 = tf.cast(ix2, tf.int32) # cast indices as ints 
        M2 = tf.gather_nd(self.masks_K, ix2) # get slices of masks_K with indices
        M3 = tf.reshape(M2, [-1,MAX_LEN,DIM])#+tf.cast(tf.convert_to_tensor(0.01),tf.float64) # reshape them
        P2 = tf.math.multiply(tf.math.exp(x_pred),tf.cast(M3,tf.float32)) # apply them to the exp-predictions
        #P2 = tf.math.exp(x_pred) # apply them to the exp-predictions (running this line instead of the above line, yielded better accuracy)
        P2 = tf.math.divide(P2,tf.math.reduce_sum(P2,axis=-1,keepdims=True)) # normalize predictions
        
        return P2        