import tensorflow as tf

def _make_divisible(v, divisor=8, min_value=None):
    """
    Ensures that all layers have a channel number that is divisible by 8.
    This is often used in MobileNet architectures to ensure compatibility
    with hardware accelerators.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def squeeze_excite_block(inputs, ratio=4):
    """
    Squeeze and Excitation Block
    Dynamically recalibrates channel importance to suppress noise.
    """
    filters = inputs.shape[-1]
    se = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(inputs)
    se = tf.keras.layers.Dense(filters // ratio, activation='relu6', kernel_initializer='he_normal', use_bias=False)(se)
    se = tf.keras.layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    return tf.keras.layers.Multiply()([inputs, se])

def inverted_residual_block(inputs, filters, stride, expansion_ratio, use_se=False, use_residual=True):
    """
    Inverted Residual Block (MobileNetV2/V3 style)
    Expands -> Depthwise Conv -> Compress

    Args:
        inputs: Input tensor.
        filters: Number of output filters (channels).
        stride: Stride for the depthwise convolution.
        expansion_ratio: Factor to expand the channels in the expansion phase.
        use_se: Whether to include Squeeze-and-Excitation block.
        use_residual: Whether to include a residual connection.
    """
    in_channels = inputs.shape[-1]
    x = inputs

    # 1. Expansion Phase (1x1 Conv)
    if expansion_ratio > 1:
        expanded_channels = _make_divisible(in_channels * expansion_ratio)
        x = tf.keras.layers.Conv2D(expanded_channels, 1, padding='same',
                                 use_bias=False, kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU(max_value=6)(x)

    # 2. Depthwise Convolution
    x = tf.keras.layers.DepthwiseConv2D((3, 3), padding='same', strides=stride,
                                      use_bias=False, depth_multiplier=1)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(max_value=6)(x)

    # 3. Squeeze and Excitation (Optional)
    if use_se:
        x = squeeze_excite_block(x)

    # 4. Projection Phase (1x1 Conv) - Linear Bottleneck
    # This compresses the channels back down
    x = tf.keras.layers.Conv2D(filters, 1, padding='same',
                             use_bias=False, kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # 5. Residual Connection
    # Only if stride is 1 and input/output channels match
    if use_residual and stride == 1 and in_channels == filters:
        x = tf.keras.layers.Add()([inputs, x])
    
    return x
