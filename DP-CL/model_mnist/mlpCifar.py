'''
Multi-Layer Perceptron Class
'''
import numpy as np
import tensorflow as tf
#import input_data
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import candidate_sampling_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
#from logisticRegression import LogisticRegression
import math

AECODER_VARIABLES = 'AECODER_VARIABLES'
CONV_VARIABLES = 'CONV_VARIABLES'

class EncLayer(object):
    '''Typical convolutional layer of MLP'''
    def __init__(self, inpt, kernel_size, out_channels, stride, var_list, pad="SAME", name="autoencoder", collect=AECODER_VARIABLES):
        
        """self.W = W
        self.b = b
        self.input = inpt;"""
        self.input = inpt
        in_channels = inpt.get_shape().as_list()[-1]
        with tf.variable_scope(name):
            #n = kernel_size * kernel_size * out_channels
            n = kernel_size * in_channels
            stdv = 1.0 / math.sqrt(n)
            w = tf.get_variable('kernel', [kernel_size, kernel_size, in_channels, out_channels],
                            tf.float32, initializer=tf.random_uniform_initializer(-stdv, stdv))
            #initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))
                            
            # Append the variable to the trainable variables list
            tf.add_to_collections(collect, w)
            var_list.append(w)
            b = tf.get_variable('biases', [out_channels], tf.float32, initializer=tf.random_uniform_initializer(-stdv, stdv))
            tf.add_to_collections(collect, b)
            var_list.append(b)

        self.inputShape = inpt.get_shape().as_list()
        # params of the layers
        #self.params = [w, b]
            
    # define the Chebyshev Polinomial approximation
    def Chebyshev(self, x):
        return (-5*x**7 + 21*x**5 - 35*x**3 + 35*x + 16)/(2.0**5) # L = 7
        #return (-x**3 + 3*x + 2)/(2.0**2) # L = 3
        #return (5*x**5 + 2*x**4 - 14*x**3 - 4*x**2 + 17*x + 10)/(2.0) # L = 5
        #return (-x**2 + 2*x + 3)/(2.0**2) # L = 2
    
    def dpChebyshev(self, x, Delta, epsilon, batch_size):
        coefficients = [-5.0, 21.0, -35.0, 35.0, 16.0] # L = 7
        #coefficients = [-1.0, 3.0, 2.0] # L = 3
        #coefficients = [-1.0, 2.0, 3.0] # L = 2
        #coefficients = [5.0, 2.0, -14.0, -4.0, 17.0, 10.0] # L = 5
        for i in range(0, len(coefficients)):
            perturbFM = np.random.laplace(0.0, 1.0/(epsilon*batch_size), 1).astype(np.float32);
            perturbFM = tf.multiply(perturbFM, Delta);
            coefficients[i] += perturbFM;
        return (tf.multiply(coefficients[0], x**7) + tf.multiply(coefficients[1], x**5) + tf.multiply(coefficients[2], x**3) + tf.multiply(coefficients[3], x**1) + coefficients[4])/(2.0**5) # L = 7
        #return (tf.multiply(coefficients[0], x**3) + tf.multiply(coefficients[1], x) + coefficients[2])/(2.0**2) # L = 3
        #return (tf.multiply(coefficients[0], x**2) + tf.multiply(coefficients[1], x) + coefficients[2])/(2.0**2) # L = 2
        #return (tf.multiply(coefficients[0], x**5) + tf.multiply(coefficients[1], x**4) + tf.multiply(coefficients[2], x**3) + tf.multiply(coefficients[3], x**2) + tf.multiply(coefficients[4], x**1) + coefficients[5])/(2.0) # L = 5
    
    # sampling hidden neurons given visible neurons
    def propup(self, v, W, b, var_list, train_phase):
        '''Compute the sigmoid activation for hidden units given visible units'''
        h = tf.nn.relu(tf.nn.conv2d(v, W, strides=[1, 2, 2, 1], padding='SAME') + b);
        n_out = 64
        collect = AECODER_VARIABLES
        with tf.variable_scope('bn_auto'):
            beta = tf.get_variable('beta', shape=[n_out], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            gamma = tf.get_variable('gamma', shape=[n_out], dtype=tf.float32, initializer=tf.constant_initializer(1.0))
            var_list.append(beta)
            var_list.append(gamma)
            tf.add_to_collections(collect, beta)
            tf.add_to_collections(collect, gamma)
            batch_mean, batch_var = tf.nn.moments(h, [0,1,2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.9)
            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)
            mean, var = tf.cond(train_phase,
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(h, mean, var, beta, gamma, 1e-3)
        #h = tf.contrib.layers.batch_norm(h, scale=True, is_training=False, updates_collections=[AECODER_VARIABLES])
        
        
        return tf.clip_by_value(normed, -1.0, 1.0) # values of hidden neurons must be bounded [-1, 1]

    """def propup(self, v, W, b, training_switch):
        '''Compute the sigmoid activation for hidden units given visible units'''
        h = tf.nn.relu(tf.nn.conv2d(v, W, strides=[1, 2, 2, 1], padding='SAME') + b);
        h = tf.layers.batch_normalization(h,
                                          trainable=True,
                                          training=training_switch,
                                          name='propup_bn',
                                          reuse=tf.AUTO_REUSE)
        return tf.clip_by_value(h, -1, 1) # values of hidden neurons must be bounded [-1, 1]
    """
    # differentially private hidden terms given visible neurons
    def dp_propup(self, v, Delta, epsilon, batch_size):
        '''Compute the differentially private activation for hidden terms given visible units'''
        h = tf.add(tf.nn.conv2d(v, self.W, strides=[1, 2, 2, 1], padding='SAME'), self.b)
        max = tf.reduce_max(h)
        h = h/max;
        # hidden neurons have to be bounded in [0, 1] after the perturbation
        Chebyshev_h = tf.clip_by_value(self.dpChebyshev(h, Delta, epsilon, batch_size), 0.0, 1.0)

        return Chebyshev_h # return perturbed approximated polinomial coefficients h
    
    # transpose of convolutional RBM given hidden neurons, this is use for convolutional auto-encoder
    def decode(self, xShape, propup, activation=tf.nn.sigmoid):
        rc_input = activation(tf.add(tf.nn.conv2d_transpose(propup, self.W,
                    tf.stack([xShape, self.inputShape[1], self.inputShape[2], self.inputShape[3]]),
                    strides=[1, 2, 2, 1], padding='SAME'), self.b))
        return rc_input
    
    # reconstruct visible units from convolutional feature maps
    def decode2(self, xShape, propup, W, b, activation=tf.nn.sigmoid):
        # upsampling given hidden feature maps to obtain input's size feature maps. This step can be considered an actual deconvolution
        upsample3 = tf.image.resize_images(propup, size=(self.inputShape[1],self.inputShape[2]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # reconstruct the original inputs
        #rc_input = activation(tf.add(tf.nn.conv2d(input=upsample3, filter=tf.transpose(W, perm=[1, 0, 3, 2]), strides=[1, 1, 1, 1], padding='SAME'), b))
        rc_input = activation(tf.nn.conv2d(input=upsample3, filter=tf.transpose(W, perm=[1, 0, 3, 2]), strides=[1, 1, 1, 1], padding='SAME'))
        ###
        return rc_input

    # get pre-training objective function, this is use for convolutional auto-encoder
    def get_train_ops(self, xShape, learning_rate=0.1):
        propup = self.propup(self.input)
        rc_v = self.decode(xShape, propup, activation=tf.nn.sigmoid);
        self.cost = tf.reduce_sum(tf.square(rc_v - self.input))
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost, var_list=self.params)
        return optimizer
    
    # get differentially private pre-training energy function for convolutional RBM
    def get_train_ops2(self, xShape, W, b, perturbFMx, perturbFM_h, var_list, train_phase):
        # compute Laplace noise injected into coefficients h
        #Delta = 2*14*14*25;
        #perturbFM = np.random.laplace(0.0, 2*Delta/(epsilon*batch_size), 14*14*32)
        #perturbFM = np.reshape(perturbFM, [-1, 14, 14, 32]);
        
        # compute h terms
        propup = self.propup(self.input + perturbFMx, W, b, var_list, train_phase) + perturbFM_h
        # reconstruct v terms
        rc_v = self.decode2(xShape, propup, W, b, activation=tf.nn.relu);
        #rc_v = tf.reduce_sum(rc_v, 3, keepdims=True)/32
        
        #propup2 = self.propup(rc_v, W, b)
        #print(propup2)
        
        zeros = array_ops.zeros_like(rc_v, dtype=self.input.dtype)
        cond = (rc_v >= zeros)
        relu_logits = array_ops.where(cond, rc_v, zeros)
        neg_abs_logits = array_ops.where(cond, -rc_v, rc_v)
        #Taylor = math_ops.add(relu_logits - activation_v * self.input, math.log(2.0) + 0.5*neg_abs_logits + 1.0/8.0*neg_abs_logits**2)
        self.cost = tf.abs(math_ops.add(relu_logits - rc_v * (self.input + perturbFMx), math.log(2.0) + 0.5*neg_abs_logits)) #+ tf.nn.softmax_cross_entropy_with_logits(labels = propup, logits = propup2)#+ 1.0/8.0*neg_abs_logits**2
        
        # define AdamOptimizer optimization
        #optimizer = tf.train.AdamOptimizer().minimize(self.cost, var_list=[W, b])
        #return optimizer
        return self.cost






