import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

ds_train_ = image_dataset_from_directory(
    './dataset/train',
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=True,
)
ds_valid_ = image_dataset_from_directory(
    './dataset/valid',
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=False,
)

# Data Pipeline
def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label

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

def compile_and_fit(model, epochs=50):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(epsilon=0.01),
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )

    history = model.fit(
        ds_train,
        validation_data=ds_valid,
        epochs=epochs,
    )

    return history