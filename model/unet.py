# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow import keras

class UNet_model(keras.Model):
    def __init__(self):
        super(UNet_model, self).__init__(name='UNet_model')
        self.down_block_1 = Down_Block(16, 0.1)
        self.down_block_2 = Down_Block(32, 0.1)
        self.down_block_3 = Down_Block(64, 0.1)
        self.down_block_4 = Down_Block(128, 0.1)
        self.middle_block = Middle_Block(256, 0.1)
        self.up_block_1 = Up_Block(128, 0.1)
        self.up_block_2 = Up_Block(64, 0.1)
        self.up_block_3 = Up_Block(32, 0.1)
        self.up_block_4 = Up_Block(16, 0.1)
        self.out_block = Out_Block()

    def call(self, inputs, training=False, mask=None):
        temp, short_cut_1 = self.down_block_1(inputs, training=training)
        temp, short_cut_2 = self.down_block_2(temp, training=training)
        temp, short_cut_3 = self.down_block_3(temp, training=training)
        temp, short_cut_4 = self.down_block_4(temp, training=training)

        temp = self.middle_block(temp, training=training)

        temp = self.up_block_1(temp, shortcut_inputs=short_cut_4, training=training)
        temp = self.up_block_2(temp, shortcut_inputs=short_cut_3, training=training)
        temp = self.up_block_3(temp, shortcut_inputs=short_cut_2, training=training)
        temp = self.up_block_4(temp, shortcut_inputs=short_cut_1, training=training)

        output = self.out_block(temp)
        return output

    @staticmethod
    def loss_object(predictions, labels):
        loss = tf.keras.losses.BinaryCrossentropy()
        return loss(predictions, labels)


class Down_Block(keras.layers.Layer):
    def __init__(self, filters, dropout_rate):
        super(Down_Block, self).__init__('name')
        self.conv = keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1),
                                        padding='same', activation='elu')
        self.dropout = keras.layers.Dropout(rate=dropout_rate)
        self.conv_ = keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1,1),
                                            padding='same', activation='elu',)
        self.pooling = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')

    def call(self, inputs, **kwargs):
        conv = self.conv(inputs)
        drop = self.dropout(conv, training=kwargs['training'])
        conv_ = self.conv_(drop)
        pool = self.pooling(conv_)
        return pool, conv_

class Middle_Block(keras.layers.Layer):
    def __init__(self, filters, dropout_rate):
        super(Middle_Block, self).__init__()
        self.conv = keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='elu')
        self.dropout = keras.layers.Dropout(rate=dropout_rate)
        self.conv_ = keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='elu')

    def call(self, inputs, **kwargs):
        conv = self.conv(inputs)
        drop = self.dropout(conv, training=kwargs['training'])
        conv_ = self.conv_(drop)
        return conv_

class Up_Block(keras.layers.Layer):
    def __init__(self, filters, dropout_rate):
        super(Up_Block, self).__init__()
        self.upconv = keras.layers.Conv2DTranspose(filters=filters, kernel_size=(2, 2), strides=(2, 2), padding='valid')
        self.conv = keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='elu')
        self.dropout = keras.layers.Dropout(rate=dropout_rate)
        self.conv_ = keras.layers.Conv2D(filters=filters, kernel_size=(3, 3),  padding='same', activation='elu')

    def call(self, inputs, **kwargs):
        upconv = self.upconv(inputs)
        concat = keras.layers.concatenate([upconv, kwargs['shortcut_inputs']])
        conv = self.conv(concat)
        drop = self.dropout(conv, training=kwargs['training'])
        conv_ = self.conv_(drop)
        return conv_

class Out_Block(keras.layers.Layer):
    def __init__(self):
        super(Out_Block, self).__init__()
        self.conv = keras.layers.Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='sigmoid')

    def call(self, inputs, **kwargs):
        conv = self.conv(inputs)
        return conv