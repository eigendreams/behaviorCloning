# Udacity SDCND - Behavior Cloning

## 1. Generalities

This project ended up being far more frustrating than I expected. It seems most of my frustration 
came from the need to generate good data, which is difficult if using a keyboard. That being said, 
I found out that you really do not need to use a GPU or AWS in order to train the model, which was 
surprising. Though I must disclaim first that:

1.- I mostly used IPYTHON for everything, so formatting of the model.py file might be somewhat awkward.
2.- I generated most of the data myself, but I did use the dataset provided by Udacity for training as well.
3.- I did not have a joystick available, so all my data was generated using the keyboard. This seems to 
need significantly more data, or data augmentation (which I did not use) in order to work well.
4.- I generated a validatio set post training, since I was using generators, and frankly, the only way to 
really validate is to run the simulation and let the network try to provide the angle.
5.- I reduced the throttle speed in the drive.py file, as at higher speeds at first I was weary of just 
letting it go, in retrsopective this is nonsense, as it is a simulation, and anyway it still works at the 
original throttle.

## 2. Data

I could only use the keyboard. I generated around 3GB of data, driving around the loop many times, and also 
taking some recordings for recovery scenarios. I think that, in the end, those recovery scenarios were the 
crucial secret sauce needed to train the model.

I do not remember how many times I drove around the lap, but they were many. I also iterated data recording 
and training for a while, but I saw later on that the network (at least mine) works better if it is trained 
with the entire dataset, after adding 'special sections'. So training took a long long while. I

For data augmentation, the only thing I really did was horizontal flips of the image. And I also added an 
'offset angle' to the side (left and right) camera images, since we want to add 'pressure' for the car to go 
back to the center. In the code:

"""
left_ang    = center_ang + abs(center_ang * 0.75) + 5 * pi / 180.0
right_ang   = center_ang - abs(center_ang * 0.75) - 5 * pi / 180.0
"""

The images were all reduce to 1/4 of their original size, and cropped to reflect only the area with lane 
markers (ie, ot the sky and not the car itself). I used all three RGB channels as well. No further 
preprocessing (beyond normalization to a -0.5,0.5 range) was done.

Also, and most importantly, no filtering was used whatsoever for the angles. At first I tried using a low 
pass filter, but performance in curves was horrible. After obtaining a 'curated' dataset and just avoiding 
any filtering, performance improved a lot. Of course, I had to take some extra recordings on some sections, 
like the curve right after the bridge, and retrain with this new data added to all previous datasets. It is 
not perfect, but I was feed up at the moment, and right after that it was holidays.

## Model

I tried to use many models. I wrote a description once on the forum, but my question dissapeared on the 
transition! Basically, at first I tried with mode sophisticated models, but the training time as horrible, 
and performance was subpar. In the end, I settled on a much simpler network:

Input shape = (20, 80, 3)

Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
convolution2d_4 (Convolution2D)  (None, 20, 80.0, 48)  1344        convolution2d_input_3[0][0]      
____________________________________________________________________________________________________
maxpooling2d_4 (MaxPooling2D)    (None, 10, 40.0, 48)  0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
activation_9 (Activation)        (None, 10, 40.0, 48)  0           maxpooling2d_4[0][0]             
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 10, 40.0, 64)  27712       activation_9[0][0]               
____________________________________________________________________________________________________
maxpooling2d_5 (MaxPooling2D)    (None, 5, 20.0, 64)   0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
activation_10 (Activation)       (None, 5, 20.0, 64)   0           maxpooling2d_5[0][0]             
____________________________________________________________________________________________________
convolution2d_6 (Convolution2D)  (None, 5, 20.0, 80)   46160       activation_10[0][0]              
____________________________________________________________________________________________________
maxpooling2d_6 (MaxPooling2D)    (None, 2, 10.0, 80)   0           convolution2d_6[0][0]            
____________________________________________________________________________________________________
activation_11 (Activation)       (None, 2, 10.0, 80)   0           maxpooling2d_6[0][0]             
____________________________________________________________________________________________________
flatten_2 (Flatten)              (None, 1600.0)        0           activation_11[0][0]              
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 1600.0)        0           flatten_2[0][0]                  
____________________________________________________________________________________________________
dense_7 (Dense)                  (None, 512)           819712      dropout_3[0][0]                  
____________________________________________________________________________________________________
activation_12 (Activation)       (None, 512)           0           dense_7[0][0]                    
____________________________________________________________________________________________________
dense_8 (Dense)                  (None, 256)           131328      activation_12[0][0]              
____________________________________________________________________________________________________
activation_13 (Activation)       (None, 256)           0           dense_8[0][0]                    
____________________________________________________________________________________________________
dense_9 (Dense)                  (None, 128)           32896       activation_13[0][0]              
____________________________________________________________________________________________________
activation_14 (Activation)       (None, 128)           0           dense_9[0][0]                    
____________________________________________________________________________________________________
dense_10 (Dense)                 (None, 64)            8256        activation_14[0][0]              
____________________________________________________________________________________________________
activation_15 (Activation)       (None, 64)            0           dense_10[0][0]                   
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 64)            0           activation_15[0][0]              
____________________________________________________________________________________________________
dense_11 (Dense)                 (None, 16)            1040        dropout_4[0][0]                  
____________________________________________________________________________________________________
activation_16 (Activation)       (None, 16)            0           dense_11[0][0]                   
____________________________________________________________________________________________________
dense_12 (Dense)                 (None, 1)             17          activation_16[0][0]              
====================================================================================================
Total params: 1068465

Please note, that all this was done interactively on IPYTHON, so, if model.py is executed from scratch,
most likely it will not work as well.

The network architecture is simply:

conv2d(3x3, 48) -> maxpool(2x2) -> ELU() 
conv2d(3x3, 64) -> maxpool(2x2) -> ELU() 
conv2d(3x3, 80) -> maxpool(2x2) -> ELU() 
Flatten()
Dropout(0.2)
Dense(512) -> ELU()
Dense(256) -> ELU()
Dense(128) -> ELU()
Dense(64)  -> ELU()
Dropout(0.2)
Dense(16)  -> ELU()
Dense(1)

The first convolutional layers are there to try to catch general features of the road, while the last fully
connected layers try to use those found features to produce the steering angle.

The dropout layers were added to try to combat overfitting. At first I tried to use batch normalization, but
it went horribly wrong.

The last layer has no activation (linear) since this is a regression problem.

For optimization the loss function used was MSE, and the optimizar was ADAM. I had to reduce the default ADAM 
learning rate to 0.00001, as I had some weird problems before where the network would settle on a loss but really 
not learn anything.

## Validation

Ok, so I really believe that in this particular project, the only meaningfull validation is trying the
simulator itself. But anyhow, I geerated a dataset consisting of two normal laps, and used that as the
validation set. Since I had to use generators, and the data itself is augmented, if only a bit, I did not
want to use a subset of the generated datasets for training.

## Performance

The model works. Though it still has troubles at very steep curves. I do not see an easy way to solve this
if using the keyboard to generate the data. But it works! Even after feeding it such bad data. I have even
seen that much simpler networks could also work, though it seems that such students used a lot of
data augmentation, and also a proper joystick.