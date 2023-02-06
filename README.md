# Self-Driving-Car-Engineer-Udacity
Projects of Self-Driving Car Engineer Specialization from udacity

[P1: Line Detection](#P1)<br>
[P2: MiniFlow](#P2)<br>
[P3: Traffic Signs Classification](#P3)<br>
[P4: Behavioral Cloning](#P4)<br>
[P5: Image Preprocessing](#P5)<br>
[P6: Advanced Lane Detection](#P6)<br>
[P7: Machine Learning Algorithms](#P7)<br>
[P8: Vehicle Detection using traditional computer vision](#P8)<br>
[P9: Sensor Fusion Using extended kalman filter](#P9)<br>
[P10: Localization](#P10)<br>
[P11: PID Controller](#P11)<br>


<a id='P1'></a>
# P1: Line Detection:-
> In this model I have used traditional computer vision techniques (Color Thresholding, Canny edge detection, Hough Transform) to detect lane lines in roads.

<a id='P2'></a>
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

<a id='P3'></a>
# P3: Traffic Signs Classification: 
> In this project, I have train a deeplearning model to classify images of traffic signs,<br> I have used LeNet Architecture to build this model,<br> I trained model on 43-classes of traffic signs.

<a id='P4'></a>
# P4: Behavioral Cloning: 
> In this project, I have collected some labled images of roads and steering angles attached to each posistion using udacity car simulator,<br> I also have trained simple deeplearning model based on AlexNet to predict the suitable angle in each position.

<a id='P5'></a>
# P5: Image Preprocessing:
In this task:
1. I have calculated ditortion and calibration matrix from chess board images.<br>
2. I have used these matrices to unditrot distorted images captured using the same camera.<br>
3. I also have transformed image from a presprective to another to have better view of the image.<br>
4. I have implemented Sobel operators, magnitude and angle of gradients, thresholding them to get edges.<br>
5. I have explained some color spaces and difference between them.

<a id='P6'></a>
# P6: Advanced Lane Detection: 
In this project, I have build a pipeline that takes an image and: 
1. undistort image.
2. warp image to focus only on lane lines in road.
3. combine multiple color channels together and threshold them tot isolate lane lines only in foreground.
4. build an algorithm to detect lines from this binary thresholded image.
5. test algorithm on test images. 


<img src="https://i.ibb.co/vmckJqj/color-spaces.png" alt="Color Spaces" style="width:100%">
<img src="https://i.ibb.co/JzLjZ1W/line-detection.png" alt="Line detection" style="width:100%">

7. test algorithm on recorded **video** .

<a id='P7'></a>
# P7: Machine Learning Algorithms: 
In this task: 
1. I have build some machine learning algorithms **Decision Tree, Gaussian Naive Bayes** from scratch.<br> 
2. I have trained these algorithms on randomly generated data, comparing them with scikit-learn **SVC** algorithm.

<img src="https://i.postimg.cc/NFQ7H91b/machine-learning-algos.png" alt="Naive Bayes - SVM - Decision Tree" style="width:100%">


<a id='P8'></a>
# P8: Vehicle Detection using traditional computer vision:-  
In this Project: 
1. I have used traditional computer vision techniques to extract features from images: <br>
```Features: ```<br>
  a. color-based features: (Histogram of pixels' color in image, color of each pixel).<br>
  b. edges-based features: (HOG: Histogram of Gradients).
2. I used these features to train a support vector machine model to classisfy whether image contains car.
3. I passed a window over image to classify each part of image to find cars in different locations and scales.
<a><img src="https://i.ibb.co/wzJTGcz/car-detection.png" alt="car-detection" border="0"></a>

<a id='P9'></a>
# P9: Sensor Fusion Using extended kalman filter:-  
In this Task: 
1. I have designed kalman filter in both  ```C++``` and ```python``` to predict the correct posistion and velocity of objects and pedestrians in road, kalman filter takes his input from two different sources: 
    1. **Lidar** sensor that calculates posisition of object in both x, y directions.
    2. **Radar** sensor that calculates: <br>
            a) Rho: $\rho$,<br>
            b) theta: $\theta$,<br>
            c) Rho_dot: $\rho^*$.<br>
2. I used extended Kalman filter to trasform radar readings to cartesian coordinates whera locations are: $x$, $y$ and velocity: $v_x, v_y$.


<a id='P10'></a>
# P10: Localization:-  
In this Task: 
1. I have implemented a 1D map realization of **Markov Localization Filter** in C++.

<a id='P11'></a>
# P11: PID Controller:-  
In this Task: 
1. I created a ```.ipynb``` notebook, where I have explained PID controller and the importance of each component (Proportional-Intergral-Derivative) and compared between each of this component with graphs.

