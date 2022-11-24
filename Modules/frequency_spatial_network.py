import numpy as np
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, Lambda,MaxPooling2D, concatenate, UpSampling2D,Add, BatchNormalization, Conv3D, MaxPooling3D, UpSampling3D
import tensorflow as tf

L = 3

def ifft_layer(kspace):
    real = Lambda(lambda kspace : kspace[:,:,:,0])(kspace)
    imag = Lambda(lambda kspace : kspace[:,:,:,1])(kspace)
    kspace_complex = K.tf.complex(real,imag)
    rec1 = K.tf.abs(K.tf.signal.ifft2d(kspace_complex))
    rec1 = K.tf.expand_dims(rec1, -1)
    return rec1

def nrmse(y_true, y_pred):
    denom = K.sqrt(K.mean(K.square(y_true), axis=(1,2,3)))
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=(1,2,3)))\
    /denom
    
def model1(H=256,W=256,channels = 2,kshape = (3,3)):
    inputs = Input(shape=(H,W,channels))

    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(conv4)
    
    up1 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv3],axis=-1)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(up1)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(conv5)
    
    up2 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv2],axis=-1)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(up2)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(conv6)
    
    up3 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1],axis=-1)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(up3)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(conv7)
    
    conv8 = Conv2D(2, (1, 1), activation='linear')(conv7)
    res1 = Add()([conv8,inputs])
    model = Model(inputs=inputs, outputs=[res1])
    return model

def model2(H=256,W=256,channels = 1,kshape = (3,3)):
    inputs = Input(shape=(H,W,channels))

    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(conv4)
    
    up1 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv3],axis=-1)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(up1)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(conv5)
    
    up2 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv2],axis=-1)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(up2)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(conv6)
    
    up3 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1],axis=-1)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(up3)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(conv7)
    
    conv8 = Conv2D(1, (1, 1), activation='linear')(conv7)
    res1 = Add()([conv8,inputs])
    model = Model(inputs=inputs, outputs=[res1])
    return model


def combined_model(mu1, sigma1, mu2, sigma2, model1, model2, H=256, W=256, channels=2):

    inputs = Input(shape=(H,W,channels))

    out1 = model1(inputs)

    res1_scaled = Lambda(lambda res1 : (res1 * sigma1 + mu1))(out1)
    rec1 = Lambda(ifft_layer)(res1_scaled)
    rec1_norm = Lambda(lambda rec1 : (rec1 - mu2) / sigma2)(rec1)

    out2 = model2(rec1_norm)

    model = Model(inputs=inputs, outputs=[res1_scaled, out2])

    return model



def wnet(mu1,sigma1,mu2,sigma2,H=256,W=256,channels = 2,kshape = (3,3),kshape2=(3,3)):
    inputs = Input(shape=(H,W,channels))

    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(conv1)
    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(conv2)
    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(conv3)
    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(conv4)
    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(conv4)
    
    up1 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv3],axis=-1)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(up1)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(conv5)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(conv5)
    
    up2 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv2],axis=-1)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(up2)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(conv6)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(conv6)
    
    up3 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1],axis=-1)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(up3)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(conv7)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(conv7)
    
    conv8 = Conv2D(2, (1, 1), activation='linear')(conv7)
    res1 = Add()([conv8,inputs])
    res1_scaled = Lambda(lambda res1 : (res1*sigma1+mu1))(res1)
    
    rec1 = Lambda(ifft_layer)(res1_scaled)
    rec1_norm = Lambda(lambda rec1 : (rec1-mu2)/sigma2)(rec1)
    
    conv9 = Conv2D(48, kshape2, activation='relu', padding='same')(rec1_norm)
    conv9 = Conv2D(48, kshape2, activation='relu', padding='same')(conv9)
    conv9 = Conv2D(48, kshape2, activation='relu', padding='same')(conv9)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv9)
    
    conv10 = Conv2D(64, kshape2, activation='relu', padding='same')(pool4)
    conv10 = Conv2D(64, kshape2, activation='relu', padding='same')(conv10)
    conv10 = Conv2D(64, kshape2, activation='relu', padding='same')(conv10)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv10)
    
    conv11 = Conv2D(128, kshape2, activation='relu', padding='same')(pool5)
    conv11 = Conv2D(128, kshape2, activation='relu', padding='same')(conv11)
    conv11 = Conv2D(128, kshape2, activation='relu', padding='same')(conv11)
    pool6 = MaxPooling2D(pool_size=(2, 2))(conv11)
    
    conv12 = Conv2D(256, kshape2, activation='relu', padding='same')(pool6)
    conv12 = Conv2D(256, kshape2, activation='relu', padding='same')(conv12)
    conv12 = Conv2D(256, kshape2, activation='relu', padding='same')(conv12)
    
    up4 = concatenate([UpSampling2D(size=(2, 2))(conv12), conv11],axis=-1)
    conv13 = Conv2D(128, kshape2, activation='relu', padding='same')(up4)
    conv13 = Conv2D(128, kshape2, activation='relu', padding='same')(conv13)
    conv13 = Conv2D(128, kshape2, activation='relu', padding='same')(conv13)
    
    up5 = concatenate([UpSampling2D(size=(2, 2))(conv13), conv10],axis=-1)
    conv14 = Conv2D(64, kshape2, activation='relu', padding='same')(up5)
    conv14 = Conv2D(64, kshape2, activation='relu', padding='same')(conv14)
    conv14 = Conv2D(64, kshape2, activation='relu', padding='same')(conv14)
    
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv14), conv9],axis=-1)
    conv15 = Conv2D(48, kshape2, activation='relu', padding='same')(up6)
    conv15 = Conv2D(48, kshape2, activation='relu', padding='same')(conv15)
    conv15 = Conv2D(48, kshape2, activation='relu', padding='same')(conv15)
    
    out = Conv2D(1, (1, 1), activation='linear')(conv15)
    model = Model(inputs=inputs, outputs=[res1_scaled,out])
    return model

# This mode is still under construction
def wnet_dc(mu1,sigma1,mu2,sigma2,mask,H=256,W=256,channels = 2,kshape = (3,3),kshape2=(3,3)):
    inputs = Input(shape=(H,W,channels))

    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(conv1)
    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(conv2)
    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(conv3)
    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(conv4)
    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(conv4)
    
    up1 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv3],axis=-1)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(up1)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(conv5)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(conv5)
    
    up2 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv2],axis=-1)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(up2)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(conv6)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(conv6)
    
    up3 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1],axis=-1)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(up3)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(conv7)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(conv7)
    
    conv8 = Conv2D(2, (1, 1), activation='linear')(conv7)
    conv8_dc = Lambda(lambda conv8 : conv8*mask)(conv8)
     
    
    res1 = Add()([conv8_dc,inputs])
    res1_scaled = Lambda(lambda res1 : (res1*sigma1+mu1))(res1)
    
    rec1 = Lambda(ifft_layer)(res1_scaled)
    rec1_norm = Lambda(lambda rec1 : (rec1-mu2)/sigma2)(rec1)
    
    conv9 = Conv2D(48, kshape2, activation='relu', padding='same')(rec1_norm)
    conv9 = Conv2D(48, kshape2, activation='relu', padding='same')(conv9)
    conv9 = Conv2D(48, kshape2, activation='relu', padding='same')(conv9)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv9)
    
    conv10 = Conv2D(64, kshape2, activation='relu', padding='same')(pool4)
    conv10 = Conv2D(64, kshape2, activation='relu', padding='same')(conv10)
    conv10 = Conv2D(64, kshape2, activation='relu', padding='same')(conv10)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv10)
    
    conv11 = Conv2D(128, kshape2, activation='relu', padding='same')(pool5)
    conv11 = Conv2D(128, kshape2, activation='relu', padding='same')(conv11)
    conv11 = Conv2D(128, kshape2, activation='relu', padding='same')(conv11)
    pool6 = MaxPooling2D(pool_size=(2, 2))(conv11)
    
    conv12 = Conv2D(256, kshape2, activation='relu', padding='same')(pool6)
    conv12 = Conv2D(256, kshape2, activation='relu', padding='same')(conv12)
    conv12 = Conv2D(256, kshape2, activation='relu', padding='same')(conv12)
    
    up4 = concatenate([UpSampling2D(size=(2, 2))(conv12), conv11],axis=-1)
    conv13 = Conv2D(128, kshape2, activation='relu', padding='same')(up4)
    conv13 = Conv2D(128, kshape2, activation='relu', padding='same')(conv13)
    conv13 = Conv2D(128, kshape2, activation='relu', padding='same')(conv13)
    
    up5 = concatenate([UpSampling2D(size=(2, 2))(conv13), conv10],axis=-1)
    conv14 = Conv2D(64, kshape2, activation='relu', padding='same')(up5)
    conv14 = Conv2D(64, kshape2, activation='relu', padding='same')(conv14)
    conv14 = Conv2D(64, kshape2, activation='relu', padding='same')(conv14)
    
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv14), conv9],axis=-1)
    conv15 = Conv2D(48, kshape2, activation='relu', padding='same')(up6)
    conv15 = Conv2D(48, kshape2, activation='relu', padding='same')(conv15)
    conv15 = Conv2D(48, kshape2, activation='relu', padding='same')(conv15)
    
    out = Conv2D(1, (1, 1), activation='linear')(conv15)
    model = Model(inputs=inputs, outputs=[res1_scaled,out])
    return model

def ifft_layer_new(kspace):
    
    _ans = []
    
    for i in range(L):
        real = Lambda(lambda kspace : kspace[:,i,:,:,0])(kspace)
        imag = Lambda(lambda kspace : kspace[:,i,:,:,1])(kspace)
        kspace_complex = K.tf.complex(real,imag)
        rec1 = K.tf.abs(K.tf.signal.ifft2d(kspace_complex))
        rec1 = K.tf.expand_dims(rec1, -1)
        _ans.append(rec1)
        
    return tf.stack(_ans, axis = 1)

def get_middle_layer(image_3d):
    
    _ans = image_3d[:,L//2,:,:,:]
    
    return _ans

    # H, W = 256, 256
    # channels = 2
    # kshape = (1, 3, 3)
    # kshape2 = (1, 3, 3)
    # sigma1 = 1
    # mu1 = 1
    # sigma2 = 1
    # mu2 = 1




def wnet_new(mu1,sigma1,mu2,sigma2,H=256,W=256,channels = 2,kshape = (1,3,3),kshape2=(1,3,3)):

    inputs = Input(shape=(L, H, W, channels))

    pool_size = (1, 2, 2)

    conv1 = Conv3D(48, kshape, activation='relu', padding='same')(inputs)
    conv1 = Conv3D(48, kshape, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=pool_size)(conv1)

    conv2 = Conv3D(64, kshape, activation='relu', padding='same')(pool1)
    conv2 = Conv3D(64, kshape, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=pool_size)(conv2)

    conv3 = Conv3D(128, kshape, activation='relu', padding='same')(pool2)
    conv3 = Conv3D(128, kshape, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=pool_size)(conv3)

    conv4 = Conv3D(256, kshape, activation='relu', padding='same')(pool3)
    conv4 = Conv3D(256, kshape, activation='relu', padding='same')(conv4)

    up1 = concatenate([UpSampling3D(size=(1, 2, 2))(conv4), conv3],axis=-1)
    conv5 = Conv3D(128, kshape, activation='relu', padding='same')(up1)
    conv5 = Conv3D(128, kshape, activation='relu', padding='same')(conv5)

    up2 = concatenate([UpSampling3D(size=(1, 2, 2))(conv5), conv2],axis=-1)
    conv6 = Conv3D(64, kshape, activation='relu', padding='same')(up2)
    conv6 = Conv3D(64, kshape, activation='relu', padding='same')(conv6)

    up3 = concatenate([UpSampling3D(size=(1, 2, 2))(conv6), conv1],axis=-1)
    conv7 = Conv3D(48, kshape, activation='relu', padding='same')(up3)
    conv7 = Conv3D(48, kshape, activation='relu', padding='same')(conv7)

    conv8 = Conv3D(2, (1, 1, 1), activation='linear')(conv7)
    res1 = Add()([conv8,inputs])
    res1_scaled = Lambda(lambda res1 : (res1*sigma1+mu1))(res1)

    rec1 = Lambda(ifft_layer_new)(res1_scaled)
    rec1_norm = Lambda(lambda rec1 : (rec1-mu2)/sigma2)(rec1)

    conv9 = Conv3D(48, kshape2, activation='relu', padding='same')(rec1_norm)
    conv9 = Conv3D(48, kshape2, activation='relu', padding='same')(conv9)
    pool4 = MaxPooling3D(pool_size=pool_size)(conv9)

    conv10 = Conv3D(64, kshape2, activation='relu', padding='same')(pool4)
    conv10 = Conv3D(64, kshape2, activation='relu', padding='same')(conv10)
    pool5 = MaxPooling3D(pool_size=pool_size)(conv10)

    conv11 = Conv3D(128, kshape2, activation='relu', padding='same')(pool5)
    conv11 = Conv3D(128, kshape2, activation='relu', padding='same')(conv11)
    pool6 = MaxPooling3D(pool_size=pool_size)(conv11)

    conv12 = Conv3D(256, kshape2, activation='relu', padding='same')(pool6)
    conv12 = Conv3D(256, kshape2, activation='relu', padding='same')(conv12)

    up4 = concatenate([UpSampling3D(size=pool_size)(conv12), conv11],axis=-1)
    conv13 = Conv3D(128, kshape2, activation='relu', padding='same')(up4)
    conv13 = Conv3D(128, kshape2, activation='relu', padding='same')(conv13)

    up5 = concatenate([UpSampling3D(size=pool_size)(conv13), conv10],axis=-1)
    conv14 = Conv3D(64, kshape2, activation='relu', padding='same')(up5)
    conv14 = Conv3D(64, kshape2, activation='relu', padding='same')(conv14)

    up6 = concatenate([UpSampling3D(size=pool_size)(conv14), conv9],axis=-1)
    conv15 = Conv3D(48, kshape2, activation='relu', padding='same')(up6)
    conv15 = Conv3D(48, kshape2, activation='relu', padding='same')(conv15)

    out = Conv2D(1, (1, 1), activation='linear')(conv15)
    out = Lambda(get_middle_layer)(out)
    model = Model(inputs=inputs, outputs=[res1_scaled,out])
    return model

wnet_new(1,2,3,4).summary()