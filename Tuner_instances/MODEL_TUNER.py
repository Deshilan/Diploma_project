import numpy as np
import os
import tensorflow as tf
import tensorflow.python.keras.models
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import keras_tuner as kt

train_ds = tf.keras.utils.image_dataset_from_directory(
    directory="C:\python\PROJEKT_DYPLOMOWY\MY_OWN_DATASET",
    labels='inferred',
    label_mode='binary',
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(150, 150),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset="training",
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False,
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    directory="C:\python\PROJEKT_DYPLOMOWY\MY_OWN_DATASET",
    labels='inferred',
    label_mode='binary',
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(150, 150),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset="validation",
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    directory="C:\python\PROJEKT_DYPLOMOWY\Part_images\FULL_TEST_DATASET",
    labels='inferred',
    label_mode='binary',
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(150, 150),
    shuffle=True,
    seed=123,
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False,
)

normalization_layer = layers.Rescaling(1. / 255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
normalized_ds_val = validation_ds.map(lambda x, y: (normalization_layer(x), y))
normalized_ds_test = test_ds.map(lambda x, y: (normalization_layer(x), y))

def build(hp):
    model = Sequential()
    model.add(layers.Rescaling(1. / 255, input_shape=(150, 150, 3)))
    model.add(layers.Conv2D(hp.Int('input_units',
                                   min_value=8,
                                   max_value=32,
                                   step=8)
                            (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D())

    model.add(layers.Conv2D(hp.Int('input_units',
                                   min_value=8,
                                   max_value=32,
                                   step=8)
                            (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D())

    model.add(layers.Conv2D(hp.Int('input_units',
                                   min_value=8,
                                   max_value=32,
                                   step=8)
                            (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D())

    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense('Dense', min_value=32, max_value=256, step=32, activation='relu'))
    model.add(layers.Activation("softmax"))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-6])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])


tuner = kt.Hyperband(build,
                     objective='val_accuracy',
                     max_epochs=30,
                     factor=3,
                     directory="RESULTS",
                     project_name="FIRST_TRY")


stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)


tuner.search(normalized_ds,
             epochs = 30,
             batch_size=32,
             validation_data = normalized_ds_val)

print(tuner.get_best_models()[0].summary())
print(tuner.get_best_hyperparameters()[0].values)

model = tuner.get_best_models(num_models=1)[0]
print (model.summary())
# Evaluate the best model.
loss, accuracy = model.evaluate(normalized_ds_test)
print('loss:', loss)
print('accuracy:', accuracy)
model.save('prueba1.h5')