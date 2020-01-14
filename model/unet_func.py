# -*- coding: utf-8 -*-
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, \
    Input, Concatenate, Activation, Dropout
from tensorflow.keras.optimizers import SGD

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
    conv6 = Conv2D(1, (1, 1), activation='relu', padding='same')(conv5)
    conv7 = Activation('softmax')(conv6)

    model = Model(inputs=inputs, outputs=conv7)
    optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['accuracy'])
    return model