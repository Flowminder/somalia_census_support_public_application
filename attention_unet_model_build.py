"""
Attention U-Net model template for multi-class semantic segmentation.
"""
from keras.layers import Layer, Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Activation, Dropout, Add, Multiply
from keras.models import Model
import keras.backend as K

class AttentionBlock(Layer):
    def __init__(self, filters, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        self.W_g = Conv2D(self.filters, (1,1), padding='same')
        self.W_x = Conv2D(self.filters, (1,1), padding='same')
        self.psi = Conv2D(1, (1,1), padding='same')
        self.relu = Activation('relu')
        self.sigmoid = Activation('sigmoid')
        super(AttentionBlock, self).build(input_shape)

    def call(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(Add()([g1, x1]))
        psi = self.sigmoid(self.psi(psi))
        return Multiply()([x, psi])

def attention_unet_model(n_classes=4, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    # Encoder
    c1 = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2,2))(c1)
    c2 = Conv2D(64, (3,3), activation='relu', padding='same')(p1)
    p2 = MaxPooling2D((2,2))(c2)
    c3 = Conv2D(128, (3,3), activation='relu', padding='same')(p2)
    p3 = MaxPooling2D((2,2))(c3)
    c4 = Conv2D(256, (3,3), activation='relu', padding='same')(p3)
    p4 = MaxPooling2D((2,2))(c4)
    c5 = Conv2D(512, (3,3), activation='relu', padding='same')(p4)

    # Decoder with attention
    u4 = UpSampling2D((2,2))(c5)
    att4 = AttentionBlock(256)(c4, u4)
    u4 = concatenate([u4, att4])
    c6 = Conv2D(256, (3,3), activation='relu', padding='same')(u4)

    u3 = UpSampling2D((2,2))(c6)
    att3 = AttentionBlock(128)(c3, u3)
    u3 = concatenate([u3, att3])
    c7 = Conv2D(128, (3,3), activation='relu', padding='same')(u3)

    u2 = UpSampling2D((2,2))(c7)
    att2 = AttentionBlock(64)(c2, u2)
    u2 = concatenate([u2, att2])
    c8 = Conv2D(64, (3,3), activation='relu', padding='same')(u2)

    u1 = UpSampling2D((2,2))(c8)
    att1 = AttentionBlock(32)(c1, u1)
    u1 = concatenate([u1, att1])
    c9 = Conv2D(32, (3,3), activation='relu', padding='same')(u1)

    outputs = Conv2D(n_classes, (1,1), activation='softmax')(c9)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def get_model(n_classes, img_height, img_width, num_channels):
    return attention_unet_model(
        n_classes=n_classes,
        IMG_HEIGHT=img_height,
        IMG_WIDTH=img_width,
        IMG_CHANNELS=num_channels,
    ) 