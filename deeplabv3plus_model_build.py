"""
DeepLabV3+ model template for multi-class semantic segmentation with ResNet50 backbone.
"""
import tensorflow as tf
from tensorflow.keras import layers, models

def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    if stride == 1:
        depth_padding = 'same'
    else:
        pad_total = kernel_size + (kernel_size - 1) * (rate - 1) - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = layers.ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'
    if not depth_activation:
        x = layers.Activation('relu')(x)
    x = layers.DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                               padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = layers.BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False, name=prefix + '_pointwise')(x)
    x = layers.BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = layers.Activation('relu')(x)
    return x

def ASPP(x, input_shape, out_stride):
    b0 = layers.Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = layers.BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = layers.Activation('relu', name='aspp0_activation')(b0)

    rate = [6, 12, 18] if out_stride == 16 else [12, 24, 36]
    b1 = SepConv_BN(x, 256, 'aspp1', rate=rate[0], depth_activation=True)
    b2 = SepConv_BN(x, 256, 'aspp2', rate=rate[1], depth_activation=True)
    b3 = SepConv_BN(x, 256, 'aspp3', rate=rate[2], depth_activation=True)

    b4 = layers.GlobalAveragePooling2D()(x)
    b4 = layers.Reshape((1, 1, x.shape[-1]))(b4)
    b4 = layers.Conv2D(256, (1, 1), padding='same', use_bias=False, name='image_pooling')(b4)
    b4 = layers.BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = layers.Activation('relu')(b4)
    size_before = tf.keras.backend.int_shape(x)
    b4 = layers.UpSampling2D(size=(size_before[1], size_before[2]), interpolation='bilinear')(b4)

    x = layers.Concatenate()([b4, b0, b1, b2, b3])
    x = layers.Conv2D(256, (1, 1), padding='same', use_bias=False, name='concat_projection')(x)
    x = layers.BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.1)(x)
    return x

def deeplabv3plus_model(n_classes=4, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=3, out_stride=16):
    input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.get_layer('conv4_block6_2_relu').output  # 1/16
    low_level_feat = base_model.get_layer('conv2_block3_2_relu').output  # 1/4

    x = ASPP(x, input_shape, out_stride)
    x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    low_level_feat = layers.Conv2D(48, (1, 1), padding='same', use_bias=False, name='low_level_projection')(low_level_feat)
    low_level_feat = layers.BatchNormalization(name='low_level_projection_BN', epsilon=1e-5)(low_level_feat)
    low_level_feat = layers.Activation('relu')(low_level_feat)
    x = layers.Concatenate()([x, low_level_feat])
    x = SepConv_BN(x, 256, 'decoder_conv0', depth_activation=True)
    x = SepConv_BN(x, 256, 'decoder_conv1', depth_activation=True)
    x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    x = layers.Conv2D(n_classes, (1, 1), padding='same')(x)
    x = layers.Activation('softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=x)
    return model

def get_model(n_classes, img_height, img_width, num_channels):
    return deeplabv3plus_model(
        n_classes=n_classes,
        IMG_HEIGHT=img_height,
        IMG_WIDTH=img_width,
        IMG_CHANNELS=num_channels,
    ) 