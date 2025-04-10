# The Princess Bride classification
This project is an end-to-end ML pipeline for classifying images of five characters from one of my favourite films, the Princess Bride. The goal is to build a classifier which can correctly identify which of the following five characters is in the image.

Is it princess Buttercup or Westley?

<img src="https://github.com/user-attachments/assets/9c18b706-c9e9-4cc9-b13e-0bdace6c1ce1" height="190"> <img src="https://github.com/user-attachments/assets/952f0a2b-b354-4155-9caf-25d3898b9c8b" height="190">


Or perhaps it is one of the bandits (top to bottom) Fezzik, Inigo or Vizzini?

<img src="https://github.com/user-attachments/assets/da3fcfe7-2a8f-4941-b1fc-8e097b0742d7" height="190"> 

These are the five classes we consider. 

The outline of the project is roughly as follows: 
1. web-scraping for photos of the five actors using the Fatkun Batch Download chrome extension
2. face detection with OpenCV and Haar cascade classifiers
3. feature extraction: stacking original images alongside their wavelet transforms, reshaping and normalizing the feature vectors
4. performing three different classification algorithms, namely SVM, logistic regression and random forest
5. choosing the best algorithm and parameters using cross-validation


In the future I hope to build a UI such that a user can drop an image and receive the predicted label.


# Image Classification of Five Iconic Actors

This project focuses on the task of classifying images into one of five distinct classes, each corresponding to a well-known actor. The goal is to build a machine learning pipeline that can accurately predict the identity of an actor based on a facial image.

## Project Overview

The project is structured into several phases:

1. **Data Collection**  
   Images were scraped from the web using the Fatkun Batch Download Chrome extension. This allowed efficient downloading of multiple photos per actor from Google Images.

2. **Face Detection**  
   We utilized OpenCV's pre-trained Haar cascade classifiers to detect faces in the raw images. This step ensures that only the relevant regions (faces) are used for training, improving model focus and performance.

3. **Discarding irrelevant images**
  Some images contain multiple faces, all of which are recognized by the Haar cascade classifiers. Thus, we must remove the irrelevant images from the datasets. This includes pictures with faces of other people and pictures where faces are not clearly visible. This step was done manually.

5. **Feature Engineering**  
   For each image, we applied a wavelet transform to extract texture features. These wavelet-transformed images were then stacked alongside the original images. The resulting feature vectors were flattened and normalized, creating a consistent numerical format for the models.

6. **Model Training**
   Three different classification algorithms were evaluated:  
   - Support Vector Machine (SVM)  
   - Logistic Regression  
   - Random Forest  

   Each model was wrapped in a pipeline with data scaling, and their hyperparameters were fine-tuned using `GridSearchCV`.

7. **Model Selection**  
   We compared the models using cross-validation, selecting the best performer based on accuracy and generalization. Evaluation included the use of confusion matrices for visual inspection of results.

## Future Work

As a next step, I plan to deploy the model using a user-friendly interface, where users can upload an image and receive the predicted actor label. This would involve developing a web app with Flask and potentially integrating OpenCV for on-the-fly face detection.


