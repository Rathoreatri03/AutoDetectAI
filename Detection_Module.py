import cv2
import numpy as np
import tensorflow as tf
frame_width = 224
frame_height = 224
def pred_frame(model, frame, user_input):
    """
    Perform prediction on a single frame using the provided model.


    Args:
    - model (tensorflow.keras.Model): Trained model for prediction.
    - frame (numpy.ndarray): Input frame for prediction (should be resized to 224 x 224).

    Returns:
    - bbox (list): List containing the predicted bounding box coordinates (rect_start, rect_end).
    - class_name (str): Predicted class name ('yes' or 'no').
    - frame (numpy.ndarray): Input frame with predicted bounding box drawn.
    """

    rect_start = (0, 0)
    rect_end = (0, 0)
    bbox = []
    img_resized = cv2.resize(frame, (frame_width, frame_height))
    img_resized = img_resized.transpose((1, 0, 2))
    img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img.astype('float32') / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)  # Add batch dimension

    # Predict the class and bounding box
    class_prediction, bbox_prediction = model.predict(img_batch)
    print('Prediction: ', class_prediction)
    # Get the class prediction (0 or 1)
    class_label = np.round(class_prediction[0]).astype(int)
    class_name = f'{user_input}' if class_label == 1 else f'no_{user_input}'

    # Denormalize the bounding box coordinates
    x_center_normalized, y_center_normalized, width_normalized, height_normalized = bbox_prediction[0]

    x_center = x_center_normalized * frame_width
    y_center = y_center_normalized * frame_height
    width = width_normalized * frame_width
    height = height_normalized * frame_height

    x_min = int(x_center - (width / 2))
    y_min = int(y_center - (height / 2))
    x_max = int(x_center + (width / 2))
    y_max = int(y_center + (height / 2))

    if x_min > frame_width: x_min = frame_width
    if y_min > frame_height: y_min = frame_height
    if x_max > frame_width: x_max = frame_width
    if y_max > frame_height: y_max = frame_height

    rect_start = (x_min, y_min)
    rect_end = (x_max, y_max)
    bbox.append(rect_start)
    bbox.append(rect_end)

    return bbox, class_name, frame
