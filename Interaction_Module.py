import cv2
import numpy as np
import keyboard
import time
import os
import tensorflow as tf
from Training_Module import train_model, save_model, create_model
from Detection_Module import pred_frame
from Data_Augmentation import apply_augmentation


class InteractionModule:
    """
    Class for handling interaction with video input, frame annotation, and model interaction.
    """

    def __init__(self):
        """
        Initialize the InteractionModule with necessary variables and model.
        """
        self.rect_start = (0, 0)
        self.rect_end = (0, 0)
        self.drawing = False
        self.rectangles = []
        self.boundary_box = []
        self.frame_cache = []
        self.current_frame = None
        self.frame_width = 224
        self.frame_height = 224
        self.batch_size = 5
        self.class_no = []
        self.model_path = self.model_creation()
        self.model = create_model(input_shape=(self.frame_width, self.frame_height, 3), model_path=self.model_path)
        self.user_input = None
        self.class_names = self.class_name_selection()
        self.class_name_map = {'0': self.class_names[0], '1': self.class_names[1]}

    def class_name_selection(self):
        names = []
        self.user_input = input('Enter the Class name for the detection: ')
        print("-----------------------------------------------------------------------------------")
        names.append(self.user_input)
        names.append(f'no_{self.user_input}')

        return names

    def model_creation(self):
        """
           Creates or loads a model based on user input.

           If user chooses to train a new model:
           - Prompts for the directory to save the model.
           - Checks for existing models in the directory and increments version numbers to avoid overwriting.
           - Returns the path to the new model.

           If user chooses to use an existing model:
           - Prompts for the absolute path of the existing `.h5` model file.
           - Validates the file path and ensures it ends with `.h5`.
           - Returns the path to the existing model.

           Returns:
               str: Absolute path to the model file (.h5) that either is newly created or exists.
        """

        # Ask user whether to train a new model or use an existing one
        while True:
            print("-----------------------------------------------------------------------------------")
            train_new_model = input("Do you want to train a new model? (yes/no): ")
            print("-----------------------------------------------------------------------------------")
            if train_new_model.lower() == 'yes':

                base_path = input('Enter the absolute path of model saving directory: ')
                print("-----------------------------------------------------------------------------------")
                model_name = 'Model'
                model_extension = '.h5'
                version = 1
                model_path = os.path.join(base_path, f"{model_name}{model_extension}")

                # Check if the initial path already exists
                while os.path.exists(model_path):
                    # Increment version number and update path
                    version += 1
                    model_path = os.path.join(base_path, f"{model_name}-{version}{model_extension}")

                return model_path

            elif train_new_model.lower() == 'no':
                while True:
                    base_path = input('Enter the absolute path of the model: ')
                    print("-----------------------------------------------------------------------------------")
                    if base_path.endswith('.h5') and os.path.isfile(base_path):
                        return base_path
                    else:
                        print('Incorrect file format or file does not exist. Please enter a valid .h5 model file path.')
                        print("-----------------------------------------------------------------------------------")
            else:
                print('Incorrect option choose from yes/no')
                print("-----------------------------------------------------------------------------------")
                continue

    def preprocess_rectangles(self, rectangles):
        """
        Preprocess rectangles for training by extracting features and normalizing coordinates.

        Args:
            rectangles (list): List of tuples (bbox, class_name, frame).

        Returns:
            tuple: X (numpy array), y_class (numpy array), y_bbox (numpy array)
                X contains resized and normalized frames,
                y_class contains binary labels,
                y_bbox contains normalized bounding box coordinates.
        """
        X = []
        y_class = []
        y_bbox = []

        for bbox, class_name, frame in rectangles:
            x_min, y_min, x_max, y_max = bbox

            # Normalize coordinates
            x_center = (x_min + x_max) / 2.0
            y_center = (y_min + y_max) / 2.0
            width = x_max - x_min
            height = y_max - y_min

            x_center_normalised = x_center / self.frame_width
            y_center_normalised = y_center / self.frame_height
            width_normalised = width / self.frame_width
            height_normalised = height / self.frame_height

            # Resize frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame, (self.frame_width, self.frame_height))
            frame_resized = frame_resized.astype('float32') / 255.0
            resized_img_transposed = frame_resized.transpose((1, 0, 2))

            X.append(resized_img_transposed)
            y_class.append(0 if class_name == 'no_accident' else 1)
            y_bbox.append([x_center_normalised, y_center_normalised, width_normalised, height_normalised])

        return np.array(X), np.array(y_class), np.array(y_bbox)

    def draw_rectangle(self, event, x, y, flags, param):
        """
        Handle drawing of rectangles on the frame using mouse events.

        Args:
            event (int): The event type (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE, etc.).
            x (int): The x-coordinate of the mouse event.
            y (int): The y-coordinate of the mouse event.
            flags: Additional flags passed by OpenCV.
            param: Additional parameters passed by OpenCV.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.rect_start = (x, y)
            self.rect_end = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.rect_end = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.rect_end = (x, y)
            if self.rect_start != self.rect_end:
                x1 = max(0, min(self.rect_start[0], self.rect_end[0]))
                y1 = max(0, min(self.rect_start[1], self.rect_end[1]))
                x2 = min(self.frame_width - 1, max(self.rect_start[0], self.rect_end[0]))
                y2 = min(self.frame_height - 1, max(self.rect_start[1], self.rect_end[1]))

                scaled_bbox = ((x1, y1), (x2, y2))
                bbox = (x1, y1, x2, y2)

                while True:
                    class_name = input(f"Enter class option 0 (no_{self.user_input}) or 1 ({self.user_input})): ")
                    if class_name in self.class_name_map:
                        class_name = self.class_name_map[class_name]
                        break
                    else:
                        print('Invalid option')

                self.rectangles.append((scaled_bbox, class_name, self.current_frame.copy()))
                frame = self.current_frame.copy()
                augmented_data = apply_augmentation(bbox, class_name, frame)
                for item in augmented_data:
                    bbox, class_get, frame = item
                    self.boundary_box.append((bbox, class_get, frame))
                    if class_get == f'no_{self.user_input}':
                        self.class_no.append(0)
                    else:
                        self.class_no.append(1)

                frame_to_show = self.current_frame.copy()
                cv2.rectangle(frame_to_show, scaled_bbox[0], scaled_bbox[1], (255, 0, 0), 2)
                cv2.imshow('Draw Rectangle', frame_to_show)

                self.rect_start = (0, 0)
                self.rect_end = (0, 0)

        if self.drawing:
            frame_to_show = self.current_frame.copy()
            cv2.rectangle(frame_to_show, self.rect_start, self.rect_end, (0, 255, 0), 2)
            cv2.imshow('Draw Rectangle', frame_to_show)

    def choose_input(self):
        """
        Prompt user to choose between camera or video input.

        Returns:
            cv2.VideoCapture: OpenCV VideoCapture object for chosen input.
        """
        while True:
            print("-----------------------------------------------------------------------------------")
            decision = input("Enter '0' for the camera and '1' for video input: ")
            print("-----------------------------------------------------------------------------------")
            if decision == '0':
                cap = cv2.VideoCapture(0)
                cap.set(3, self.frame_width)
                cap.set(4, self.frame_height)
                return cap
            elif decision == '1':
                video_path = input("Enter the absolute path to the video:")
                print("-----------------------------------------------------------------------------------")
                cap = cv2.VideoCapture(video_path)
                return cap
            else:
                print("Error Occurs!!!\nWrong input!! Please Choose correct option either '0' or '1'")
                print("-----------------------------------------------------------------------------------")

    def run_frame(self):
        """
        Main loop to run interaction with frames from chosen input source.
        """
        frame_index = 0
        cap = self.choose_input()
        print("Press 'space' to capture the next frame.")
        print("Press and hold the 'right arrow' key to fast forward.")
        print("Press and hold the 'left arrow' key to rewind.")
        print("Press 'enter' to enter annotation mode.")
        print("Press 'ESC' to exit or train the model at instance.")
        print("-----------------------------------------------------------------------------------")

        while True:
            if frame_index < len(self.frame_cache):
                self.current_frame = self.frame_cache[frame_index]
            else:
                success, self.current_frame = cap.read()
                if not success:
                    print("Failed to capture image or end of video reached")
                    break
                self.current_frame = cv2.resize(self.current_frame, (self.frame_width, self.frame_height))
                self.frame_cache.append(self.current_frame)

            cv2.imshow('frame', self.current_frame)

            while True:
                if keyboard.is_pressed('space'):
                    frame_index += 1
                    break

                if keyboard.is_pressed('right'):
                    frame_index += 1
                    if frame_index >= len(self.frame_cache):
                        success, img = cap.read()
                        if success:
                            img = cv2.resize(img, (self.frame_width, self.frame_height))
                            self.frame_cache.append(img)
                            cv2.imshow('frame', img)
                        else:
                            print("Failed to capture image or end of video reached")
                            frame_index -= 1
                    else:
                        cv2.imshow('frame', self.frame_cache[frame_index])
                    time.sleep(0.01)

                if keyboard.is_pressed('left'):
                    frame_index = max(0, frame_index - 1)
                    cv2.imshow('frame', self.frame_cache[frame_index])
                    time.sleep(0.01)

                if keyboard.is_pressed('enter'):
                    if os.path.exists(self.model_path):
                        recent_frame = self.frame_cache[frame_index].copy()
                        bbox_prediction, class_label, frame = pred_frame(self.model, recent_frame, self.user_input)

                        bbox_prediction = [(abs(x), abs(y)) for (x, y) in bbox_prediction]
                        x_min, y_min = bbox_prediction[0]
                        x_max, y_max = bbox_prediction[1]

                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                        cv2.putText(frame, f'Confidence: {class_label}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                    (0, 255, 0))

                        cv2.namedWindow('Prediction')
                        cv2.imshow('Prediction', frame)
                        while True:
                            pred_key = cv2.waitKey(1) & 0xFF
                            if pred_key == 27:  # ESC key
                                self.rectangles.append((bbox_prediction, class_label, recent_frame))
                                bbox = (x_min, y_min, x_max, y_max)
                                augmented_data = apply_augmentation(bbox, class_label, recent_frame)
                                for item in augmented_data:
                                    bbox, class_get, frame = item
                                    self.boundary_box.append((bbox, class_get, frame))
                                    if class_label == 'no_accident':
                                        self.class_no.append(0)
                                    else:
                                        self.class_no.append(1)
                                cv2.destroyWindow('Prediction')
                                break
                            if pred_key == 13:  # Enter key
                                cv2.namedWindow('Draw Rectangle')
                                cv2.setMouseCallback('Draw Rectangle', self.draw_rectangle)

                                while True:
                                    self.current_frame = self.frame_cache[frame_index].copy()
                                    for rect, class_name, frame_to_show in self.rectangles:
                                        scaled_rect = (
                                            (int(rect[0][0]), int(rect[0][1])), (int(rect[1][0]), int(rect[1][1])))
                                        cv2.rectangle(frame_to_show, scaled_rect[0], scaled_rect[1], (255, 0, 0), 2)

                                    if self.drawing:
                                        cv2.rectangle(self.current_frame, self.rect_start, self.rect_end, (0, 255, 0),
                                                      2)

                                    cv2.imshow('Draw Rectangle', self.current_frame)
                                    if cv2.waitKey(1) & 0xFF == 27:
                                        cv2.destroyWindow('Draw Rectangle')
                                        cv2.destroyWindow('Prediction')
                                        break
                                break
                    else:
                        print("-----------------------------------------------------------------------------------")
                        print("Use left mouse button to draw rectangles.")
                        print("Right-click to redraw the last drawn rectangle.")
                        print("Press 'ESC' to exit.")
                        print("-----------------------------------------------------------------------------------")

                        cv2.namedWindow('Draw Rectangle')
                        cv2.setMouseCallback('Draw Rectangle', self.draw_rectangle)

                        while True:
                            self.current_frame = self.frame_cache[frame_index].copy()
                            for rect, class_name, frame_to_show in self.rectangles:
                                scaled_rect = ((int(rect[0][0]), int(rect[0][1])), (int(rect[1][0]), int(rect[1][1])))
                                cv2.rectangle(frame_to_show, scaled_rect[0], scaled_rect[1], (255, 0, 0), 2)

                            if self.drawing:
                                cv2.rectangle(self.current_frame, self.rect_start, self.rect_end, (0, 255, 0), 2)

                            cv2.imshow('Draw Rectangle', self.current_frame)
                            if cv2.waitKey(1) & 0xFF == 27:
                                break
                        cv2.destroyWindow('Draw Rectangle')
                    print("-----------------------------------------------------------------------------------")
                    print(self.class_no)
                    num_zeros = self.class_no.count(0)
                    num_ones = self.class_no.count(1)
                    print(f'Data Collected: no_{self.user_input}-->{num_zeros}, {self.user_input}-->{num_ones}, Total-->{num_ones+num_zeros}')
                    print("-----------------------------------------------------------------------------------")
                    if (num_zeros >= int(self.batch_size)) and (num_ones >= self.batch_size):
                        indices = np.random.permutation(len(self.boundary_box))
                        shuffled_boundary_box = [self.boundary_box[i] for i in indices]
                        X_train, y_class_train, y_bbox_train = self.preprocess_rectangles(shuffled_boundary_box)
                        train_model(self.model, X_train, y_class_train, y_bbox_train, len(self.boundary_box))
                        save_model(self.model, self.model_path)
                        print("Model retrained and saved.")
                        self.model = tf.keras.models.load_model(self.model_path)
                        print("Model Loaded Successfully.")
                        self.boundary_box = []
                        self.class_no = []
                        self.rectangles = []

                if cv2.waitKey(1) & 0xFF == 27:
                    cap.release()
                    cv2.destroyAllWindows()
                    while True:
                        print("-----------------------------------------------------------------------------------")
                        decision = input("Do you want to open another video? (yes/no): ")
                        print("-----------------------------------------------------------------------------------")
                        if decision.lower() == 'no':
                            if len(self.boundary_box) > 0:
                                indices = np.random.permutation(len(self.boundary_box))
                                shuffled_boundary_box = [self.boundary_box[i] for i in indices]
                                X_train, y_class_train, y_bbox_train = self.preprocess_rectangles(shuffled_boundary_box)
                                train_model(self.model, X_train, y_class_train, y_bbox_train, len(self.boundary_box))
                                save_model(self.model, self.model_path)
                                print("Final model retrained and saved with last collected data.")
                                self.boundary_box = []
                                self.class_no = []
                            exit()

                        elif decision.lower() == 'yes':
                            self.run_frame()
                            break
                        else:
                            print('Incorrect option choose from yes/no')
                            print("-----------------------------------------------------------------------------------")
                            continue

if __name__ == "__main__":
    interaction_module = InteractionModule()
    interaction_module.run_frame()
