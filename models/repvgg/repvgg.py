"""
RepVGG Network Architecture
reference : https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
"""
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Dense ,GlobalAveragePooling2D, ReLU


def conv_bn(filters, kernel_size, stride, padding='same', groups=1, is_training=True):
    """convolution -> batch normalization"""
    result = Sequential()
    result.add(
        Conv2D(filters=filters, kernel_size=kernel_size, 
               strides=stride, padding=padding, groups=groups, use_bias=False)
        )
    result.add(BatchNormalization(trainable=is_training))
    return result

class RepVGGBlock(Layer):
    """RepVGG Block class"""
    
    def __init__(self, filters, kernel_size, stride=1, dilation_rate=1, groups=1, deploy=False, is_internal=True):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        
        # padding_1x1 = padding - kernel_size // 2 # 이게 뭔 의미가 있는건지 모르겠음..
        self.nonlinearity = ReLU()
        
        if self.deploy:
            self.conv_infer = Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, 
                                     dilation_rate=dilation_rate, groups=groups, use_bias=True)
        else:
            self.conv_3x3 = conv_bn(filters=filters, kernel_size=kernel_size, stride=stride, padding='same', groups=groups, is_training=True)
            self.conv_1x1 = conv_bn(filters=filters, kernel_size=1, stride=stride, padding='valid', groups=1, is_training=True)
            self.identity = BatchNormalization(trainable=True) if is_internal is True and stride == 1 else None

    def call(self, inputs):
        if hasattr(self, 'conv_infer'):
            return self.nonlinearity(self.conv_infer(inputs))
        
        if self.identity is None:
            identity_out = 0
        else:
            identity_out = self.identity(inputs)
        
        outputs = tf.math.add_n([self.conv_3x3(inputs),
                                 self.conv_1x1(inputs),
                                 identity_out])
        return outputs
    
    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        return tf.pad(kernel1x1, [[1, 1,],[1, 1,]])
    
    def _fuse_bn_tensor(self, branch):
        if branch is None: # if self.identity is None
            return 0, 0
        if isinstance(branch, Sequential): # for branches using conv_bn
            # branch[0] : Conv2D, branch[1] : BatchNormalization
            kernel = tf.squeeze(
                tf.convert_to_tensor(branch[0].weights, tf.float32)
                )
            moving_mean = tf.squeeze(
                tf.convert_to_tensor(branch[1].moving_mean, tf.float32)
                )
            moving_variance = tf.squeeze(
                tf.convert_to_tensor(branch[1].moving_variance, tf.float32)
                )
            gamma = tf.squeeze(
                tf.convert_to_tensor(branch[1].gamma, tf.float32)
                )
            beta = tf.squeeze(
                tf.convert_to_tensor(branch[1].beta, tf.float32)
                )
        else: # for identity branches
            if not hasattr(self, 'identity_tensor'): # self.identity_tensor는 없는데, 이게 필요한가?
                

class RepVGG(Model):
    """RepVGG: Making VGG-style ConvNets Great Again (2020)"""

    def __init__(self):
        super(RepVGG, self).__init__()

    def call(self, inputs, training=False):
        pass
