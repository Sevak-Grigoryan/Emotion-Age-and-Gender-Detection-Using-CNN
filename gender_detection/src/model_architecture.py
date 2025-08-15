import tensorflow as tf


def CNN_model(input_shape=(48, 48, 1), num_classes=2):

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(input_shape[0], input_shape[1], 3),  # MobileNet expects 3 channels
        include_top=False,
        weights=None  # No pre-trained weights since we have 48x48 grayscale
    )

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(3, (1, 1), input_shape=input_shape),  # Convert grayscale to RGB
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model
