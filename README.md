# Self-Driving-Car-Engineer-Udacity
Projects of Self-Driving Car Engineer Specialization from udacity

# P1: Line Detection:-
> In this model I have used traditional computer vision techniques (Color Thresholding, Canny edge detection, Hough Transform) to detect lane lines in roads.

# P2: MiniFlow:-
> In this project, I have designed simple computational graph module from scratch which can be used in deep learning projects

each graph can have one of 6 types of Nodes:
1. Input  (holds inputs) 
2. Add    (add several inputs)
3. Mul    (multiply several inputs)
4. sigmoid(apply sigmoid function to input)
5. Linear (multiply weight and input matrices then add bias)
6. MSE    (Calculate mean square error between tow vectors).
I have added some features for deeplearning training (ex. callbacks: EarlyStopping, ReduceLronPlateau, LearningRateScheduler)

# P3: Traffic Signs Classification: 
> In this project, I have train a deeplearning model to classify images of traffic signs,<br> I have used LeNet Architecture to build this model,<br> I trained model on 43-classes of traffic signs.

# P4: Behavioral Cloning: 
> In this project, I have collected some labled images of roads and steering angles attached to each posistion using udacity car simulator,<br> I also have trained simple deeplearning model based on AlexNet to predict the suitable angle in each position.

# P5: Image Preprocessing:
In this task:
1. I have calculated ditortion and calibration matrix from chess board images.<br>
2. I have used these matrices to unditrot distorted images captured using the same camera.<br>
3. I also have transformed image from a presprective to another to have better view of the image.<br>
4. I have implemented Sobel operators, magnitude and angle of gradients, thresholding them to get edges.<br>
5. I have explained some color spaces and difference between them.

# P6: Advanced Lane Detection: 
In this project, I have build a pipeline that takes an image and: 
1. undistort image.
2. warp image to focus only on lane lines in road.
3. combine multiple color channels together and threshold them tot isolate lane lines only in foreground.
4. build an algorithm to detect lines from this binary thresholded image.
5. test algorithm on test images. 


<img src="https://i.ibb.co/vmckJqj/color-spaces.png" alt="Color Spaces" style="width:100%">
<img src="https://i.ibb.co/JzLjZ1W/line-detection.png" alt="Line detection" style="width:100%">


7. test algorithm on recorded **video**.

# P7: Machine Learning Algorithms: 
In this task: 
1. I have build some machine learning algorithms **Decision Tree, Gaussian Naive Bayes** from scratch.<br> 
2. I have trained these algorithms on randomly generated data, comparing them with scikit-learn **SVC** algorithm.

<img src="https://i.ibb.co/kmxt2Cs/machine-learning-algos.png" alt="Naive Bayes - SVM - Decision Tree" style="width:100%">
