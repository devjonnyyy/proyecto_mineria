🤟 Sign Language Alphabet Classifier

This project uses a multilayer perceptron neural network to recognize static hand signs representing letters of the American Sign Language (ASL) alphabet from images and real-time video using a webcam.

✅ Instructions to Test the Project

1. Clone the repository

```
git clone https://github.com/devjonnyyy/proyecto_mineria.git
cd proyecto_mineria
```

2. Install dependencies

Run the following command inside the project folder:

```
pip install opencv-python mediapipe numpy pandas matplotlib scikit-learn seaborn onnx onnxruntime skl2onnx
```

3. Generate the CSV file from images *(Optional – only if you want to use your own data)*

- Place your images in subfolders named by letter (e.g. dataset/a, dataset/b, etc.)
- In the GUI, select the option **"Generate CSV"**

This will process the images and create `hand_landmarks.csv` with landmark data.

4. Train the model *(Optional – only if you don't want to use the pre-trained model)*

- In the GUI, select the option **"Train Model"**
- If you want to use the pre-trained model, simply select **trained_model.onnx**

5. Test with a static image

- In the GUI, select the option **"Test with Image"**
- The system will:
  - Ask for an image file
  - Process it
  - Display the predicted letter and an image with landmarks overlaid

6. Test in real-time using a webcam

- In the GUI, select the option **"Test in Real Time"**
- This will:
  - Open your webcam
  - Detect your hand in each frame
  - Predict and display the corresponding letter live on the video feed

📁 Included Files

- trained_model.onnx: Pre-trained model ready for prediction
- hand_landmarks.csv: Dataset already processed with landmark coordinates

🧠 Technologies Used

- Python
- MediaPipe
- OpenCV
- scikit-learn
- ONNX / ONNX Runtime

📌 Notes

- Make sure your camera is connected and accessible.
- Use clear, well-lit images with the right hand in front of a neutral background for best results.


# Sign_Language_ANN_Python
Leveraging OpenCV, MediaPipe, and Neural Networks for Sign Language Interpretation for Python

Part 1. Extraction of the characteristics form a dataset.

dataset extracted from the next reference:
mrgeislinger (2015, August 12). ASL Fingerspelling Images (RGB & Depth). [Dataset]. Kaggle. https://www.kaggle.com/datasets/mrgeislinger/asl-rgb-depth-fingerspelling-spelling-it-out

hand_landmarks.csv is a file made of 64 characteristics and one tag related to the alphabet letter. each row represents a recognized hand from the dataset and each collum is a coordinate of a hand landmark.
hand_landmarks.csv was created by running multiple instances of sign_language_images_to_csv.py, where each instance corresponed to a diferent folder containing all alphabet letters.