# Bubble Sheet Grader - OMR using Deep Learning

This project implements an **Optical Mark Recognition (OMR)** system to automatically grade bubble sheet answer sheets. The system uses a convolutional neural network (CNN) to detect filled bubbles and compare student answers against an answer key. A Gradio web interface makes it easy to upload student sheets and answer keys and get instant results.

## Features

- **Automated bubble detection** using Hough Circle Transform.
- **CNN-based classifier** to distinguish filled vs. empty bubbles (binary classification).
- **Data augmentation** to improve generalization.
- **Gradio interface** for easy testing and deployment.
- **Results table** showing per-question comparison and final score.

## Project Structure

- `PatternFinal.ipynb` – main notebook containing all code.
- `OMR_Image_Resized/` – folder of pre-processed bubble sheet images (145 images).
- `OMR_Ans.csv` – CSV file with ground truth answers for each image (30 questions, choices A-D).

## Dataset

The dataset consists of 145 scanned bubble sheet images (resized) and a corresponding CSV file with the correct answers. Each image contains 30 questions with 4 bubbles each (A, B, C, D).

## Model Architecture

The CNN model is defined as follows:

- Input: 32x32 grayscale bubble images.
- Conv2D(32, 3x3) + ReLU + MaxPooling(2x2)
- Conv2D(64, 3x3) + ReLU + MaxPooling(2x2)
- Flatten
- Dense(64) + ReLU + Dropout(0.5)
- Output: Dense(1) with sigmoid activation (binary classification: empty vs. filled).

Loss function: Binary cross-entropy  
Optimizer: Adam  
Metrics: Accuracy

## Training

- Data split: 70% training, 15% validation, 15% test (stratified by image).
- Data augmentation: rotation, width/height shift, zoom.
- Training epochs: 20
- Final test accuracy: ~93.9%

## Results

- Confusion matrix and accuracy/loss plots are displayed.
- A sample confusion matrix from the notebook shows strong performance with low misclassification.

## How to Use

1. **Prepare images**  
   Place your bubble sheet images (resized to consistent dimensions) in a folder. The provided dataset already contains OMR_Image_Resized.

2. **Train the model**  
   Run the notebook cells sequentially. The model will be saved as `bubble_detector_model.keras` in your Google Drive.

3. **Grade new sheets**  
   Use the Gradio interface:
   - Upload a student sheet image.
   - Upload the answer key sheet image.
   - Click "Submit" to receive a results dataframe and summary score.

## Code Explanation

- **Extraction**: `extract_bubble_images()` uses Hough Circles to detect bubbles and crops each bubble region.
- **Training data preparation**: `prepare_training_data_from_names()` extracts bubbles from training images and labels them based on average grayscale intensity.
- **Grading**: `grade_with_trained_model()` predicts filled bubbles, extracts correct answers from the answer key, and compares them.
- **Gradio UI**: Built with `gr.Interface` for easy file uploads and result display.

## Requirements

Install the required packages (automatically in Colab):
tensorflow
gradio
seaborn
scikit-learn
matplotlib
opencv-python
pandas
numpy




