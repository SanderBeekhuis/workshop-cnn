import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Data Pipeline
def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label


def load_data(seed):
    ds_train, ds_test = image_dataset_from_directory(
        './dataset/train/train',
        labels='inferred',
        label_mode='binary',
        image_size=[128, 128],
        interpolation='nearest',
        batch_size=64,
        shuffle=True,
        validation_split=0.2,
        subset="both",
        seed=seed
    )

    ds_train = (
        ds_train
        .map(convert_to_float)
        .cache()
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )

    ds_test = (
        ds_test
        .map(convert_to_float)
        .cache()
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )

    return ds_train, ds_test


def load_old_data(batched=True):
    if batched: 
        batch_size = 64
    else:
        batch_size = None

    ds_train_ = image_dataset_from_directory(
        './dataset old/train',
        labels='inferred',
        label_mode='binary',
        image_size=[128, 128],
        interpolation='nearest',
        batch_size=batch_size,
        shuffle=True,
    )
    ds_valid_ = image_dataset_from_directory(
        './dataset old/valid',
        labels='inferred',
        label_mode='binary',
        image_size=[128, 128],
        interpolation='nearest',
        batch_size=batch_size,
        shuffle=False,
    )


    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds_train = (
        ds_train_
        .map(convert_to_float)
        .cache()
        .prefetch(buffer_size=AUTOTUNE)
    )
    ds_valid = (
        ds_valid_
        .map(convert_to_float)
        .cache()
        .prefetch(buffer_size=AUTOTUNE)
    )

    return ds_train, ds_valid

def compile_and_fit(model, epochs=50):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(epsilon=0.01),
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )

    ds_train, ds_valid = load_old_data()

    history = model.fit(
        ds_train,
        validation_data=ds_valid,
        epochs=epochs,
    )

    return history