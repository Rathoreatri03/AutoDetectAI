import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, Flatten
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model

# Allow memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory limit to 4GB (4096MB)
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
        )
    except RuntimeError as e:
        print(e)


# -------------------- CNN Network for Training -------------------- #
def create_model(input_shape=(224, 224, 3), model_path=None):
    """
    Create and return a CNN model for training or load an existing model.

    Args:
    - input_shape (tuple): Shape of input images (width, height, channels).
    - model_path (str): Path to an existing model file to load.

    Returns:
    - model (tensorflow.keras.Model): CNN model instance.
    """
    if model_path and os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"Loaded existing model from {model_path}")
    else:
        base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
        print("Creating a new model")
        x = base_model.output
        x = Flatten()(x)
        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        class_output = Dense(1, activation='sigmoid', name='class_output')(x)
        bbox_output = Dense(4, activation='sigmoid', name='bbox_output')(x)

        model = Model(inputs=base_model.input, outputs=[class_output, bbox_output])
    return model


# -------------------- Model Training -------------------- #
def train_model(model, X_train, y_class_train, y_bbox_train, batch_size, epochs=10, validation_data=None):
    """
    Train the provided model using the given training data.

    Args:
    - model (tensorflow.keras.Model): Model to be trained.
    - X_train (numpy.ndarray): Input data (images).
    - y_class_train (numpy.ndarray): Target labels for classification.
    - y_bbox_train (numpy.ndarray): Target bounding box coordinates.
    - batch_size (int): Batch size for training.
    - epochs (int): Number of epochs for training.
    - validation_data (tuple): Validation data (X_val, y_class_val, y_bbox_val).

    Returns:
    - history (History): Training history object.
    """

    model.compile(optimizer='adam',
                  loss={'class_output': 'binary_crossentropy', 'bbox_output': 'mean_squared_error'},
                  metrics={'class_output': 'accuracy'})

    history = model.fit(X_train, {'class_output': y_class_train, 'bbox_output': y_bbox_train},
                        batch_size=batch_size, epochs=epochs, verbose=1, validation_data=validation_data)
    return history


# -------------------- Model Saving -------------------- #
def save_model(model, path):
    """
    Save the provided model to the specified path.

    Args:
    - model (tensorflow.keras.Model): Model to be saved.
    - path (str): Path where the model should be saved.

    Returns:
    - None
    """
    model.save(path)
    print(f"Model saved at {path}")
