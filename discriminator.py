import tensorflow as tf
# To import tensorflow_addons, I commented tensoflow_addons/activation/rrelu.py->Option[tf.random.Generator] away
# This only happens to tensoflow<=2.1
import tensorflow_addons as tfa


class Res(tf.keras.layers.Layer):
    def __init__(self,dim_in):
        super(Res,self).__init__()
        self.dim_in = dim_in
    def build(self, input_shape):
        self.layers = []
        init = tf.keras.initializers.RandomNormal(stddev=0.02)
        self.layers.append(  tf.keras.layers.Conv2D(self.dim_in, (3,3), padding='same', kernel_initializer=init))
        self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2) )
        self.layers.append( tfa.layers.InstanceNormalization(axis=-1) )

        self.layers.append(  tf.keras.layers.Conv2D(self.dim_in, (3,3), padding='same', kernel_initializer=init))
        self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2) )
        self.layers.append( tfa.layers.InstanceNormalization(axis=-1) )


    def call(self, inputs):
        ori = inputs
        for layer in self.layers:
            inputs = layer(inputs)
        return tf.concat([ori,inputs],axis=-1)

class LS_Discriminator(tf.keras.layers.Layer):
    def __init__(self):
        super(LS_Discriminator,self).__init__()
        pass

    def build(self, input_shape):
        dim = input_shape[1]
        init = tf.keras.initializers.RandomNormal(stddev=0.02)

        self.layers = []

        self.layers.append(  tf.keras.layers.Conv2D(32, (9,9), strides=(1,1), padding='same', kernel_initializer=init))
        self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2) )
        self.layers.append( tfa.layers.InstanceNormalization(axis=-1) )

        self.layers.append(  Res(32) )
        
        self.layers.append(  tf.keras.layers.Conv2D(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init))
        self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2) )
        self.layers.append( tfa.layers.InstanceNormalization(axis=-1) )

        self.layers.append(  Res(64) )

        self.layers.append(  tf.keras.layers.Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init))
        self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2) )
        self.layers.append( tfa.layers.InstanceNormalization(axis=-1) )


        self.layers.append(  Res(128) )

        self.layers.append(  tf.keras.layers.Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init))
        self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2) )
        self.layers.append( tfa.layers.InstanceNormalization(axis=-1) )

        self.layers.append(  tf.keras.layers.Conv2D(1, (3,3), padding='same', kernel_initializer=init))

        self.layers.append(tf.keras.layers.Activation("linear" , dtype='float32') )

    def call(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
            # tf.print("min_inputs",tf.keras.backend.min(inputs))
        return inputs