"""
U-Net++ model template for multi-class semantic segmentation.
"""
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout
from keras.models import Model

def unet_plus_plus_model(n_classes=4, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    # Encoder
    c1 = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(32, (3,3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2,2))(c1)

    c2 = Conv2D(64, (3,3), activation='relu', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(64, (3,3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2,2))(c2)

    c3 = Conv2D(128, (3,3), activation='relu', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(128, (3,3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2,2))(c3)

    c4 = Conv2D(256, (3,3), activation='relu', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(256, (3,3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2,2))(c4)

    c5 = Conv2D(512, (3,3), activation='relu', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(512, (3,3), activation='relu', padding='same')(c5)

    # Decoder with nested skip connections
    u4_1 = UpSampling2D((2,2))(c5)
    u4_1 = concatenate([u4_1, c4])
    c4_1 = Conv2D(256, (3,3), activation='relu', padding='same')(u4_1)
    c4_1 = Dropout(0.2)(c4_1)
    c4_1 = Conv2D(256, (3,3), activation='relu', padding='same')(c4_1)

    u3_2 = UpSampling2D((2,2))(c4_1)
    u3_2 = concatenate([u3_2, c3])
    c3_2 = Conv2D(128, (3,3), activation='relu', padding='same')(u3_2)
    c3_2 = Dropout(0.2)(c3_2)
    c3_2 = Conv2D(128, (3,3), activation='relu', padding='same')(c3_2)

    u2_3 = UpSampling2D((2,2))(c3_2)
    u2_3 = concatenate([u2_3, c2])
    c2_3 = Conv2D(64, (3,3), activation='relu', padding='same')(u2_3)
    c2_3 = Dropout(0.1)(c2_3)
    c2_3 = Conv2D(64, (3,3), activation='relu', padding='same')(c2_3)

    u1_4 = UpSampling2D((2,2))(c2_3)
    u1_4 = concatenate([u1_4, c1])
    c1_4 = Conv2D(32, (3,3), activation='relu', padding='same')(u1_4)
    c1_4 = Dropout(0.1)(c1_4)
    c1_4 = Conv2D(32, (3,3), activation='relu', padding='same')(c1_4)

    outputs = Conv2D(n_classes, (1,1), activation='softmax')(c1_4)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def get_model(n_classes, img_height, img_width, num_channels):
    return unet_plus_plus_model(
        n_classes=n_classes,
        IMG_HEIGHT=img_height,
        IMG_WIDTH=img_width,
        IMG_CHANNELS=num_channels,
    ) 