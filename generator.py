import tensorflow as tf
from resnet_block import ResNetBlockInstanceNorm
from suppixel import SubpixelConv2D
import tensorflow_addons as tfa

class PRes(tf.keras.layers.Layer):
    def __init__(self,dim_in):
        super(PRes,self).__init__()
        self.dim_in = dim_in
    def build(self, input_shape):
        self.layers = []
        init = tf.keras.initializers.RandomNormal(stddev=0.02)
        self.layers.append(  tf.keras.layers.Conv2D(self.dim_in, (3,3), padding='same', kernel_initializer=init))
        self.layers.append(  tf.keras.layers.PReLU() )
        self.layers.append(  tf.keras.layers.Conv2D(self.dim_in, (3,3), padding='same', kernel_initializer=init))
        self.layers.append(  tf.keras.layers.PReLU() )

    def call(self, inputs):
        ori = inputs
        for layer in self.layers:
            inputs = layer(inputs)
        return tf.concat([ori,inputs],axis=-1)

class GeneratorV2(tf.keras.layers.Layer):
    def __init__(self):
        super(GeneratorV2,self).__init__()
        pass
# SubpixelConv2d(scale=2, n_out_channels=None, act=tf.nn.relu)(n)
    def build(self, input_shape):
        # print("GeneratorV2 input,",input_shape)
        self.layers = []

        init = tf.keras.initializers.RandomNormal(stddev=0.02)
        layers = tf.keras.layers

        self.layers.append( layers.Conv2D(32,(3,3),padding = 'same',kernel_initializer=init ))
        self.layers.append( tfa.layers.InstanceNormalization(axis=-1) )
        self.layers.append( layers.LeakyReLU(alpha=0.2) )
        self.layers.append( layers.Conv2D(64,(5,5), strides=(2,2),padding = 'same',kernel_initializer=init )  )
        self.layers.append( tfa.layers.InstanceNormalization(axis=-1) )
        self.layers.append( layers.LeakyReLU(alpha=0.2) )
        self.layers.append( layers.Conv2D(128,(5,5), strides=(2,2),padding = 'same',kernel_initializer=init )  )
        self.layers.append( tfa.layers.InstanceNormalization(axis=-1) )
        self.layers.append( layers.LeakyReLU(alpha=0.2) )

        for _ in range(3):
            self.layers.append(ResNetBlockInstanceNorm(num_filter=128) )

        self.layers.append( layers.Conv2D(256,(3,3),padding = 'same',kernel_initializer=init )  )
        self.layers.append( SubpixelConv2D(input_shape,scale=2) )
        self.layers.append( layers.PReLU() )

        self.layers.append( PRes(64) )        

        self.layers.append( layers.Conv2D(256,(3,3),padding = 'same',kernel_initializer=init )  ) 
        self.layers.append( SubpixelConv2D(input_shape,scale=2) )
        self.layers.append( layers.PReLU() )

        self.layers.append( PRes(64) )

        self.layers.append( layers.Conv2D(3,(1,1), strides=(1,1),padding = 'same',kernel_initializer=init )  )
        self.layers.append(tf.keras.layers.Activation("tanh" , dtype='float32') )

    def call(self,input_image):
        
        for layer in self.layers:
            input_image = layer(input_image)

        # print("GeneratorV2 output,",input_image.shape)
        return input_image


class UpsampleGenerator(tf.keras.layers.Layer):
    def __init__(self):
        super(UpsampleGenerator,self).__init__()
        pass
# SubpixelConv2d(scale=2, n_out_channels=None, act=tf.nn.relu)(n)
    def build(self, input_shape):
        self.layers = []

        init = tf.keras.initializers.RandomNormal(stddev=0.02)
        layers = tf.keras.layers

        self.layers.append( layers.Conv2D(64,(3,3),padding = 'same',kernel_initializer=init ))
        self.layers.append( tfa.layers.InstanceNormalization(axis=-1) )
        self.layers.append( layers.LeakyReLU(alpha=0.2) )

        for _ in range(3):
            self.layers.append(ResNetBlockInstanceNorm(num_filter=64) )

        self.layers.append( layers.Conv2D(256,(3,3),padding = 'same',kernel_initializer=init )  )
        self.layers.append( SubpixelConv2D(input_shape,scale=2) )
        self.layers.append( layers.PReLU() )

        self.layers.append( PRes(64) )

        self.layers.append( layers.Conv2D(256,(3,3),padding = 'same',kernel_initializer=init )  )
        self.layers.append( SubpixelConv2D(input_shape,scale=2) )
        self.layers.append( layers.PReLU() )

        self.layers.append( PRes(64) )


        self.layers.append( layers.Conv2D(3,(1,1), strides=(1,1),padding = 'same',kernel_initializer=init )  )
        self.layers.append(tf.keras.layers.Activation("tanh" , dtype='float32') )

    def call(self,input_image):
        # print("UpsampleGenerator input, ",input_image.shape)
        for layer in self.layers:
            input_image = layer(input_image)
            # print(layer)
            # print(layer.count_params())
        # print("UpsampleGenerator output, ",input_image.shape)

        return input_image