# AutoDetectAI

AutoDetectAI is a Python-based application for real-time binary classification using computer vision techniques. It enables users to annotate video frames, train a model, and predict accident scenarios with high accuracy.

## Features

- **Real-time Video Input:** Choose between camera or video file input for real-time frame processing.
- **Frame Annotation:** Interactive frame annotation using mouse events for marking accident scenarios.
- **Model Training:** Train and retrain the model based on annotated data to improve accuracy.
- **Prediction:** Predict accident scenarios in real-time with confidence scores.
- **Retrain:** After predicting if it predicts wrong you can again train the model with new data frames in single go.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Rathoreatri03/AutoDetectAI.git
   cd AutoDetectAI
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download pre-trained models (if applicable):**
   If using pre-trained models, download them and place them in the appropriate directories.

## Usage

1. **Run the application:**
   ```bash
   python Interaction_Module.py
   ```
   Follow the on-screen instructions to interact with the application.


2. **Model Selection**
   - Choose whether to train a new model or use an existing one.
   - When training a new model, specify the directory to save the model.
   - When using an existing model, provide the absolute path to the .h5 model file.
   - Give the proper class name to detect.
    

3. **Interaction Modes:**
   - Press 'space' to capture the next frame.
   - Press 'right arrow' key to fast-forward.
   - Press 'left arrow' key to rewind.
   - Press 'enter' to enter annotation mode.
   - Press 'ESC' to exit or train the model instantly.


4. **Annotation:**
   - Use left mouse button to draw rectangles around accident scenarios.
   - Give the label to the detected frame and bbox.


5. **Model Training:**
   - If the data reaches to the batch size which is typically 50 it will automatically train the model.
   - Else if you have enough data and press 'ESC' to trigger model training with collected data.
   - Model will be retrained and saved automatically for improved accuracy.


6. **Model Reload**
   - After each train the model reload itself and start for training the next batch.
   - If the data frame reaches to end there is option to open new data frame to continue training. 

## Contributing

Contributions are welcome! If you have suggestions or improvements, please fork the repository and create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE]((https://github.com/Rathoreatri03/AutoDetectAI/blob/main/LICENSE)) file for details.

## Support

For support, please open an issue on our [GitHub repository](https://github.com/Rathoreatri03/AutoDetectAI/issues).
```
