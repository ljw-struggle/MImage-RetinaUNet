# -*- coding: utf-8 -*-
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, \
    Input, Concatenate, Dropout

def get_unet_model(patch_height, patch_width, patch_channel):
    inputs = Input(shape=(patch_height, patch_width, patch_channel))

    # 1\ Down 1
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    # 2\ Down 2
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    # 3\ Middle 3
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

    # 4\ Up 4
    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = Concatenate(axis=-1)([conv2,up1])
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)

    # 5\ Up 5
    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = Concatenate(axis=-1)([conv1,up2])
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)

    # 6\ Final 6
    conv6 = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(conv5)
    outputs = conv6

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='sgd', loss='binary_crossentropy',metrics=['accuracy'])
    return model

class Down_Block(tf.keras.layers.Layer):
    def __init__(self, filters, dropout_rate):
        super(Down_Block, self).__init__('name')
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1),
                                        padding='same', activation='elu')
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.conv_ = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1,1),
                                            padding='same', activation='elu',)
        self.pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')

    def call(self, inputs, **kwargs):
        conv = self.conv(inputs)
        drop = self.dropout(conv, training=kwargs['training'])
        conv_ = self.conv_(drop)
        pool = self.pooling(conv_)
        return pool, conv_

class Middle_Block(tf.keras.layers.Layer):
    def __init__(self, filters, dropout_rate):
        super(Middle_Block, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='elu')
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.conv_ = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='elu')

    def call(self, inputs, **kwargs):
        conv = self.conv(inputs)
        drop = self.dropout(conv, training=kwargs['training'])
        conv_ = self.conv_(drop)
        return conv_

class Up_Block(tf.keras.layers.Layer):
    def __init__(self, filters, dropout_rate):
        super(Up_Block, self).__init__()
        self.upconv = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(2, 2), strides=(2, 2), padding='valid')
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='elu')
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.conv_ = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3),  padding='same', activation='elu')

    def call(self, inputs, **kwargs):
        upconv = self.upconv(inputs)
        concat = tf.keras.layers.concatenate([upconv, kwargs['shortcut_inputs']])
        conv = self.conv(concat)
        drop = self.dropout(conv, training=kwargs['training'])
        conv_ = self.conv_(drop)
        return conv_

class Out_Block(tf.keras.layers.Layer):
    def __init__(self):
        super(Out_Block, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='sigmoid')

    def call(self, inputs, **kwargs):
        conv = self.conv(inputs)
        return conv

class UNet_model(tf.keras.Model):
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