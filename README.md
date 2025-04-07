# The Princess Bride classification
This project is an end-to-end ML pipeline for classifying images of five characters from one of my favourite films, the Princess Bride. The goal is to build a classifier which can correctly identify which of the following five characters is in the image.

Is it princess Buttercup or Westley?

<img src="https://github.com/user-attachments/assets/9c18b706-c9e9-4cc9-b13e-0bdace6c1ce1" width="200"> <img src="https://github.com/user-attachments/assets/952f0a2b-b354-4155-9caf-25d3898b9c8b" width="200">


Or perhaps it is one of the bandits (top to bottom) Fezzik, Inigo or Vizzini?

<img src="https://github.com/user-attachments/assets/da3fcfe7-2a8f-4941-b1fc-8e097b0742d7" width="200"> 

These are the five classes we consider. 

The outline of the project is roughly as follows: 
1. web-scraping for photos of the five actors using the Fatkun Batch Download chrome extension
2. face detection with OpenCV and Haar cascade classifiers
3. feature extraction: stacking original images alongside their wavelet transforms, reshaping and normalizing the feature vectors
4. performing three different classification algorithms, namely SVM, logistic regression and random forest
5. choosing the best algorithm and parameters using cross-validation

