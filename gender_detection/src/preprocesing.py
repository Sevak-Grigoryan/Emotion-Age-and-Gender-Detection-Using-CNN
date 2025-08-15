import tensorflow as tf

def create_data_generators(train_dir,
                           test_dir,
                           target_size=(48, 48),
                           batch_size=32):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale = 1./255,
        rotation_range = 60,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        brightness_range=(0.85, 1.15),
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        fill_mode = 'nearest'
    )
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255,)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size = target_size,
        batch_size = batch_size,
        class_mode = 'categorical',
        color_mode = 'grayscale',
        shuffle = True
    )
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size = target_size,
        batch_size = batch_size,
        class_mode = 'categorical',
        color_mode = 'grayscale',
        shuffle = False
    )
    return train_generator, test_generator

