# **Behavioral Cloning** 

## Writeup Template


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./loss.jpg "loss images"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* clone.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* run1 file contating all the images when drive autonomous
* run1.mp4 is a video based on images found in the run1 directory
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The clone.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My final model based on the [NVIDIA CNN architecture](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf), which consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------------:| 
| Input         		| 160x320x3 image   								| 
|Cropping				|90x320x3 image 									|
|Normalized input planes| (x / 255) - 0.5				|
| Convolution 5x5x24     	| 2x2 stride, same padding, outputs 45x160x24, activation= 'RELU' 	|
| Convolution 5x5x36     	| 2x2 stride, valid padding, outputs 21x78x36, activation= 'RELU' 	|
| Convolution 5x5x48     	| 2x2 stride, valid padding, outputs 9x37x48, activation= 'RELU' 	|
| Convolution 5x5x64     	| 2x2 stride, valid padding, outputs 3x17x64, activation= 'RELU' 	|
| Convolution 3x3x64	    | 1x1 stride, valid padding, outputs 1x15x64, activation= 'RELU'    |
| Flatten					| outputs 960														|
| Dropout 	| keep_prob=0.5	|
| Fully connected		| outputs 100, activation= 'RELU'        									|
| Dropout 	| keep_prob=0.5	|
| Fully connected		| outputs 50, activation= 'RELU'        									|
| Fully connected		| outputs 10, activation= 'RELU'        									|
| Fully connected		| outputs 1       									|


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting . 

The model was trained and validated on different data sets to ensure that the model was not overfitting . The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.
Other hyperparameters I chose are following:

|	Hyperparameters 	| Value 	|
|:------------:|:-----:|
| 	Batch size 	|	128 |
| 	Epochs 	| 10	|

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. In this project, I use sample driving data which is already included in `/opt/carnd_p3/data/`.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was based on the NIVIDA paper (End to End Learning for Self-Driving Cars). Then I modify every layers to meet the input shape `160x320x3`.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by adding two dropout layers after first two fully connect layers. The keep_prob is set to 0.5 which can gain a well-generalization model. 

![alt text][image1]

As you can see the picture above, loss on the training and validation set both became lower.


The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, especially occur on turns. To improve the driving behavior in these cases, I augment data using multiple cameras, then crop data into `90x320x3`.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture is shown above.

#### 3. Creation of the Training Set & Training Process

* To capture good driving behavior, I augment the data sat using multiple cameras.
* Then I crop images from `160x320x3` to `90x320x3`, which will be more efficent when train the model.
* After cropping step, use lambda layer `(x/255)-0.5`, which is simple way to have normalization and zero-mean data.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10. I used an adam optimizer so that manually training the learning rate wasn't necessary.
