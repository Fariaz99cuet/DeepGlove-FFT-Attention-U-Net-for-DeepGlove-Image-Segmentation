import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

# Custom layer for FFT magnitude computation
class FFTMagnitudeLayer(tf.keras.layers.Layer):
    """Custom layer to compute the magnitude of the 2D FFT of the input tensor."""
    def __init__(self, **kwargs):
        super(FFTMagnitudeLayer, self).__init__(**kwargs)

    def call(self, inputs):
        inputs_complex = tf.cast(inputs, tf.complex64)
        fft = tf.signal.fft2d(inputs_complex)
        magnitude = tf.abs(fft)
        return magnitude

# Encoder block with spatial and frequency fusion
def EncoderBlock(input_features, frequency_features, filters, kernel_size=(3, 3), activation='relu'):
    """Encoder block with fusion of spatial and frequency features."""
    # Spatial path
    conv = layers.Conv2D(filters, kernel_size, activation=activation, padding='same')(input_features)
    batch_norm = layers.BatchNormalization()(conv)
    conv = layers.Conv2D(filters, kernel_size, activation=activation, padding='same')(batch_norm)
    batch_norm = layers.BatchNormalization()(conv)
    conv = layers.Conv2D(filters, kernel_size, activation=activation, padding='same')(batch_norm)
    batch_norm = layers.BatchNormalization()(conv)
    
    # Frequency path
    freq_conv = layers.Conv2D(filters, kernel_size, activation=activation, padding='same')(frequency_features)
    freq_batch_norm = layers.BatchNormalization()(freq_conv)
    
    # Fusion
    fused = layers.Concatenate()([batch_norm, freq_batch_norm])
    fused_conv = layers.Conv2D(filters, (1, 1), activation=activation, padding='same')(fused)
    pool = layers.MaxPooling2D(pool_size=(2, 2))(fused_conv)
    return fused_conv, pool

# Decoder block with attention mechanism
def DecoderBlock(input_features, encoder_features, upsample_type, filters, kernel_size=(3, 3), activation='relu'):
    """Decoder block with attention gate before concatenation."""
    if upsample_type == 'conv_transpose':
        up_features = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(input_features)
    elif upsample_type == 'upsample':
        up_features = layers.UpSampling2D((2, 2), interpolation='bilinear')(input_features)
    
    # Attention gate
    intermediate_filters = filters
    g = layers.Conv2D(intermediate_filters, (1, 1), padding='same')(up_features)
    x = layers.Conv2D(intermediate_filters, (1, 1), padding='same')(encoder_features)
    psi = layers.Add()([g, x])
    psi = layers.Activation('relu')(psi)
    psi = layers.Conv2D(1, (1, 1), padding='same')(psi)
    attention = layers.Activation('sigmoid')(psi)
    attended_encoder_features = layers.Multiply()([encoder_features, attention])
    
    # Concatenate
    concat = layers.Concatenate()([attended_encoder_features, up_features])
    
    # Convolutional layers
    conv = layers.Conv2D(filters, kernel_size, activation=activation, padding='same')(concat)
    batch_norm = layers.BatchNormalization()(conv)
    conv = layers.Conv2D(filters // 2, kernel_size, activation=activation, padding='same')(batch_norm)
    batch_norm = layers.BatchNormalization()(conv)
    conv = layers.Conv2D(filters // 2, kernel_size, activation=activation, padding='same')(batch_norm)
    batch_norm = layers.BatchNormalization()(conv)
    return batch_norm

# U-Net model with frequency branch and attention
def U_Net(input_size, filters=[64, 128, 256, 512, 1024], num_classes=1, upsample_type='upsample', activation='relu'):
    """Builds a U-Net with dual-frequency and attention mechanisms."""
    inputs = layers.Input(shape=input_size)
    
    # Frequency branch: Compute FFT magnitude using a custom layer
    fft_layer = FFTMagnitudeLayer()
    fft_magnitude = fft_layer(inputs)
    
    # Frequency encoder
    freq_conv1 = layers.Conv2D(filters[0], (3, 3), activation=activation, padding='same')(fft_magnitude)
    freq_pool1 = layers.MaxPooling2D((2, 2))(freq_conv1)
    freq_conv2 = layers.Conv2D(filters[1], (3, 3), activation=activation, padding='same')(freq_pool1)
    freq_pool2 = layers.MaxPooling2D((2, 2))(freq_conv2)
    freq_conv3 = layers.Conv2D(filters[2], (3, 3), activation=activation, padding='same')(freq_pool2)
    freq_pool3 = layers.MaxPooling2D((2, 2))(freq_conv3)
    freq_conv4 = layers.Conv2D(filters[3], (3, 3), activation=activation, padding='same')(freq_pool3)
    freq_pool4 = layers.MaxPooling2D((2, 2))(freq_conv4)
    freq_conv5 = layers.Conv2D(filters[4], (3, 3), activation=activation, padding='same')(freq_pool4)
    
    # Spatial encoder with frequency fusion
    conv1, pool1 = EncoderBlock(inputs, freq_conv1, filters[0], kernel_size=(3, 3), activation=activation)
    conv2, pool2 = EncoderBlock(pool1, freq_conv2, filters[1], kernel_size=(3, 3), activation=activation)
    conv3, pool3 = EncoderBlock(pool2, freq_conv3, filters[2], kernel_size=(3, 3), activation=activation)
    conv4, pool4 = EncoderBlock(pool3, freq_conv4, filters[3], kernel_size=(3, 3), activation=activation)
    conv5, pool5 = EncoderBlock(pool4, freq_conv5, filters[4], kernel_size=(3, 3), activation=activation)
    
    # Bridge
    bridge = layers.Conv2D(filters[4], (3, 3), activation=activation, padding='same')(pool5)
    batch_norm = layers.BatchNormalization()(bridge)
    bridge = layers.Conv2D(filters[4], (3, 3), activation=activation, padding='same')(batch_norm)
    batch_norm = layers.BatchNormalization()(bridge)
    
    # Decoder with attention gates
    decoder0 = DecoderBlock(batch_norm, conv5, upsample_type, filters[4], kernel_size=(3, 3), activation=activation)
    decoder1 = DecoderBlock(decoder0, conv4, upsample_type, filters[3], kernel_size=(3, 3), activation=activation)
    decoder2 = DecoderBlock(decoder1, conv3, upsample_type, filters[2], kernel_size=(3, 3), activation=activation)
    decoder3 = DecoderBlock(decoder2, conv2, upsample_type, filters[1], kernel_size=(3, 3), activation=activation)
    decoder4 = DecoderBlock(decoder3, conv1, upsample_type, filters[0], kernel_size=(3, 3), activation=activation)
    
    # Output layer
    if num_classes == 1:
        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder4)
    else:
        outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(decoder4)
    
    return Model(inputs=inputs, outputs=outputs)

# Example usage
model = U_Net(input_size=(256, 256, 3), filters=[64, 128, 256, 512, 1024], num_classes=1, upsample_type='upsample', activation='relu')
model.summary()