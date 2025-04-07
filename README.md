# princess-bride-classification
This project is an end-to-end ML pipeline for classifying images of five characters from one of my favourite films, the Princess Bride.

The outline is roughly as follows: 
1. web-scraping for photos of the five actors using the Fatkun Batch Download chrome extension
2. face detection with OpenCV and Haar cascade classifiers
3. feature extraction: stacking original images alongside their wavelet transforms, reshaping and normalizing the feature vectors
4. performing three different classification algorithms, namely SVM, logistic regression and random forest
5. choosing the best algorithm and parameters using cross-validation

