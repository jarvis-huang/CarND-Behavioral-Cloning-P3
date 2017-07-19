# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* **model.py** containing the script to create and train the model
* **finetune.py** containing the script to finetune a pretrained model
* **drive.py** for driving the car in autonomous mode
* **model_lenet.h5** containing a trained convolution neural network based on LeNet architecture.
* **model_nvidia.h5** containing a trained convolution neural network based on NVIDIA architecture.
* **run1_lenet.mp4** video showing my LeNet completing the course.
* **run2_nvidia.mp4** video showing my NVIDIA-net completing the course.
* **writeup_report.md** summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_lenet.h5
```
or
```sh
python drive.py model_nvidia.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. 
The file shows the pipeline I used for training and validating the model, and it contains comments to explain 
how the code works.

### Model Architecture and Training Strategy

#### 1. The LeNet architecture

This model is based entirely on the LeNet architecture (model.py lines 85-103).

|   LeNet       |
|:-------------:|
|     Normalization            |  
|     Cropping2D               |  
| conv5-6  -> Relu -> maxpool2 |      
| conv5-16 -> Relu -> maxpool2 |
| fc-120 -> dropout -> Relu    | 
| fc-84  -> dropout -> Relu    |
| fc-1                         | 

Here, conv5-6 means 5x5 convolution layer with 6 output channels.
The model includes RELU layers to introduce nonlinearity, 
and the data is normalized in the model using a Keras lambda layer (code line 86).
To focus the network's attention on road portions, I cropped the part above the horizon line
and below the ego hood using a Cropping2D layer.

#### 2. The NVIDIA architecture

This model is based entirely on the NVIDIA paper (model.py lines 105-125).

|   NVIDIA      |
|:-------------:|
|     Normalization            |  
|     Cropping2D               |  
| conv5-24-s2  -> Relu |      
| conv5-36-s2 -> Relu  |
| conv5-48-s2 -> Relu  |
| conv3-64-s1 -> Relu  |
| conv3-64-s1 -> Relu  |
| fc-100 -> dropout -> Relu    | 
| fc-50 -> dropout -> Relu     |
| fc-10 -> dropout -> Relu     |
| fc-1                         | 

Here conv5-24-s2 means 5x5 convolution layer with 6 output channels and stride of 2.
The model includes RELU layers to introduce nonlinearity, 
and the data is normalized in the model using a Keras lambda layer (code line 106).
To focus the network's attention on road portions, I cropped the part above the horizon line
and below the ego hood using a Cropping2D layer.

#### 3. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 98, 101). 
The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 38). 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 4. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually).


#### 5. Training strategy

I adopted a two-stage training strategy. In the first stage, I only used the original dataset provided by Udacity.
I trained for 5 epochs and reached a validation loss of 0.0099. When testing on the test track, it was able to complete 75%
of the course. Which means that this is a good baseline model to further improve upon.

Next in the second stage, I used
data collected by myself driving the simulator, including many curve and recovery scenarios. I also reduced the learning
rate from 0.001 to 0.0002. I trained another 8 epochs till validation loss reached 0.0271 on the new set. This loss was higher
because the new set was more challenging. When testing the final model, it was able to complete the full course.

#### 6. Preparing training and testing data

For all the training images, I also mirror it left-right to augment the dataset and avoid dataset biasing. Overall, 
12857 and 14514 images are used in the first and second training stage.

There were a few curves most challenging and causing my car to veer off the track. In order to solve this problem, I
manually drive through these curves several times and add to the training data.

I randomly shuffled the data set and put 20% of the data into a validation set. 

For LeNet, I did not use left/right camera images but added manually collected data in challenging scenarios.
For NVIDIA, I only used Udacity dataset but used all left, center and right images. NVIDIA network is a very
power network and is able to perform very well without adding additional training data. The training data
includes 38568 images and I trained for 8 epochs. The final val_loss is 0.0162, but **its performance is
visibly higher than LeNet**.

I used generator to reduce the memory usage of the training process.

#### 7. Analysis and thoughts
NVIDIA network is very power and outperforms LeNet with fewer training data.

NVIDIA network is also smaller and contains fewer parameters.
