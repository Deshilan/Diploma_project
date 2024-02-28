import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt




train_ds = tf.keras.utils.image_dataset_from_directory(
        directory = "/mnt/c/python/PROJEKT_DYPLOMOWY/MY_OWN_DATASET",
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
        directory = "/mnt/c/python/PROJEKT_DYPLOMOWY/MY_OWN_DATASET",
        labels='inferred',
        label_mode='binary',
        class_names=None,
        color_mode='rgb',
        batch_size=128,
        image_size=(150, 150),
        shuffle=True,
        seed=123,
        validation_split=0.2,
        subset="validation",
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False,
    )


gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

normalization_layer = layers.Rescaling(1. / 255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

num_classes = 2


model = tf.keras.applications.resnet50.ResNet50(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(150, 150, 3),
    pooling=None,
    classes=2,
    classifier_activation='relu'
)

opt = tf.keras.optimizers.Adam(
    learning_rate=0.00001)

model.compile(optimizer=opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'],
                  )

model.summary()

epochs = 100
with tf.device('/device:GPU:0'):
    history = model.fit(
                train_ds,
                validation_data=validation_ds,
                epochs=epochs
            )

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


model.save_weights('/mnt/c/python/PROJEKT_DYPLOMOWY/Weights')
model.save('my_model.keras')