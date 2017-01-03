# Udacity SDCND - Behavior Cloning

## 1. Generalities

Please also see the bcBase.ipynb file, as I worked there for the most part. I also committed the HTML
version of it to GitHub.

This project ended up being far more frustrating than I expected. It seems most of my frustration 
came from the need to generate good data, which is difficult if using a keyboard. That being said, 
I found out that you really do not need to use a GPU or AWS in order to train the model, which was 
surprising. Though I must disclaim first that:

		1.- I mostly used IPYTHON for everything, so formatting of the model.py file might be somewhat awkward.
			Because of this, the model.py or even the IPYNB will of course NOT reflect the real number of epochs
			used, as I generated more data and trained with that on an as needed basis.
		2.- I generated most of the data myself, but I did use the dataset provided by Udacity for training as well.
		3.- I did not have a joystick available, so all my data was generated using the keyboard. This seems to 
			need significantly more data, or data augmentation (which I did not use) in order to work well.
			Udacity should improve the interface of their simulator. As I use a corporate laptop, I am not even
			sure that I could use a joystick.
		4.- I generated a validation set post training, since I was using generators, and frankly, the only way to 
			really validate is to run the simulation and let the network try to provide the angle. Though in later
			experiments a validator set does seems good as a suggestions for when to stop, but a lower validation
			error is NOT a reflect of a better NN in this case. NNs with much higher validation losses worked
			better in some cases. I blame bad data.
		5.- I manually tested the generated NNs first epoch by epoch, and alter on, after including a validation
			set, tested the ones with least error. This was done as the first epochs were very very long, but 
			later ones were smaller, and more data was added to the dataset for difficult sections. In a way, this
			first part can be though as giving the NN a rough start, and the later part as refinaments.

## 2. Data

I could only use the keyboard. I generated around 3GB of data, driving around the loop many times, and also 
taking some recordings for recovery scenarios. I think that, in the end, those recovery scenarios were the 
crucial secret sauce needed to train the model.

I do not remember how many times I drove around the lap, but they were many. I also iterated data recording 
and training for a while, but I saw later on that the network (at least mine) works better if it is trained 
with the entire dataset, after adding 'special sections'. So training took a long long while.

For data augmentation, the only thing I really did was horizontal flips of the image. And I also added an 
'offset angle' to the side (left and right) camera images, since we want to add 'pressure' for the car to go 
back to the center. In the code:

		left_ang    = center_ang + abs(center_ang * 0.75) + 6 * pi / 180.0
		right_ang   = center_ang - abs(center_ang * 0.75) - 6 * pi / 180.0

The images were all reduce to 1/4 of their original size, and cropped to reflect only the area with lane 
markers (ie, not the sky nor the car). I used all three RGB channels as well. No further 
preprocessing (beyond normalization to a -0.5,0.5 range) was done. However, the firs layers of the
network perform 1x1 convolutions to let the network learn its color space, as suggested by the paper:
https://arxiv.org/pdf/1606.02228v2.pdf

Also, and most importantly, no filtering was used whatsoever for the angles. At first I tried using a low 
pass filter, but performance in curves was horrible. After obtaining a 'curated' dataset and just avoiding 
any filtering, performance improved a lot. Of course, I had to take some extra recordings on some sections, 
like the curve right after the bridge, and retrain with this new data added to all previous datasets. It is 
not perfect, but I was feed up at the moment, and right after that it was holidays. Also, mixing filtering
and abusive use of the pause button will 'pollute' the data.

It is also convenient, it seems, to set INTER_AREA as interpolation method for resizing on opencv. IT seems
this works as a smoothing algo (over the image) of sorts by itself.

## Model

I tried to use many models. I wrote a description once on the forum, but my question dissapeared on the 
transition! Basically, at first I tried with mode sophisticated models, but the training time as horrible, 
and performance was subpar. In the end, I settled on a much simpler network:

		Input shape = (20, 80, 3)

		____________________________________________________________________________________________________
		Layer (type)                     Output Shape          Param #     Connected to                     
		====================================================================================================
		batchnormalization_1 (BatchNorma (None, 20, 80.0, 3)   6           batchnormalization_input_1[0][0] 
		____________________________________________________________________________________________________
		convolution2d_1 (Convolution2D)  (None, 20, 80.0, 10)  40          batchnormalization_1[0][0]       
		____________________________________________________________________________________________________
		activation_1 (Activation)        (None, 20, 80.0, 10)  0           convolution2d_1[0][0]            
		____________________________________________________________________________________________________
		convolution2d_2 (Convolution2D)  (None, 20, 80.0, 3)   33          activation_1[0][0]               
		____________________________________________________________________________________________________
		activation_2 (Activation)        (None, 20, 80.0, 3)   0           convolution2d_2[0][0]            
		____________________________________________________________________________________________________
		convolution2d_3 (Convolution2D)  (None, 20, 80.0, 60)  1680        activation_2[0][0]               
		____________________________________________________________________________________________________
		maxpooling2d_1 (MaxPooling2D)    (None, 10, 40.0, 60)  0           convolution2d_3[0][0]            
		____________________________________________________________________________________________________
		activation_3 (Activation)        (None, 10, 40.0, 60)  0           maxpooling2d_1[0][0]             
		____________________________________________________________________________________________________
		convolution2d_4 (Convolution2D)  (None, 10, 40.0, 90)  48690       activation_3[0][0]               
		____________________________________________________________________________________________________
		maxpooling2d_2 (MaxPooling2D)    (None, 5, 20.0, 90)   0           convolution2d_4[0][0]            
		____________________________________________________________________________________________________
		activation_4 (Activation)        (None, 5, 20.0, 90)   0           maxpooling2d_2[0][0]             
		____________________________________________________________________________________________________
		convolution2d_5 (Convolution2D)  (None, 3, 18.0, 120)  97320       activation_4[0][0]               
		____________________________________________________________________________________________________
		maxpooling2d_3 (MaxPooling2D)    (None, 3, 6.0, 120)   0           convolution2d_5[0][0]            
		____________________________________________________________________________________________________
		activation_5 (Activation)        (None, 3, 6.0, 120)   0           maxpooling2d_3[0][0]             
		____________________________________________________________________________________________________
		flatten_1 (Flatten)              (None, 2160.0)        0           activation_5[0][0]               
		____________________________________________________________________________________________________
		dropout_1 (Dropout)              (None, 2160.0)        0           flatten_1[0][0]                  
		____________________________________________________________________________________________________
		dense_1 (Dense)                  (None, 640)           1383040     dropout_1[0][0]                  
		____________________________________________________________________________________________________
		activation_6 (Activation)        (None, 640)           0           dense_1[0][0]                    
		____________________________________________________________________________________________________
		dense_2 (Dense)                  (None, 320)           205120      activation_6[0][0]               
		____________________________________________________________________________________________________
		activation_7 (Activation)        (None, 320)           0           dense_2[0][0]                    
		____________________________________________________________________________________________________
		dense_3 (Dense)                  (None, 160)           51360       activation_7[0][0]               
		____________________________________________________________________________________________________
		activation_8 (Activation)        (None, 160)           0           dense_3[0][0]                    
		____________________________________________________________________________________________________
		dense_4 (Dense)                  (None, 80)            12880       activation_8[0][0]               
		____________________________________________________________________________________________________
		activation_9 (Activation)        (None, 80)            0           dense_4[0][0]                    
		____________________________________________________________________________________________________
		dense_5 (Dense)                  (None, 20)            1620        activation_9[0][0]               
		____________________________________________________________________________________________________
		activation_10 (Activation)       (None, 20)            0           dense_5[0][0]                    
		____________________________________________________________________________________________________
		dense_6 (Dense)                  (None, 1)             21          activation_10[0][0]              
		====================================================================================================
		Total params: 1801810

Please note, that all this was done interactively on IPYTHON, so, if model.py is executed from scratch,
most likely it will not work as well.

The network architecture is simply:

		BatchNormalization()
		Convolution2D(1x1, 10)  -> ELU() 
		Convolution2D(1x1, 3)   -> ELU() 
		Convolution2D(3x3, 60, 'same')   -> MaxPooling2D(2x2) -> ELU() 
		Convolution2D(3x3, 90, 'same')   -> MaxPooling2D(2x2) -> ELU() 
		Convolution2D(3x3, 120, 'valid') -> MaxPooling2D(1x3) -> ELU() 
		Flatten()
		Dropout(0.2)
		Dense(640) -> ELU()
		Dense(320) -> ELU()
		Dense(160) -> ELU()
		Dense(80)  -> ELU()
		Dense(20)  -> ELU()
		Dense(1)

The first convoltional layers let the network learn its own color space. It seems useful to do a 
feature wise batch normalization, but since the data preprocessing already normalizes the images, there 
is no notorious difference in performance.
		
The next convolutional layers are there to try to catch general features of the road, while the last fully
connected layers try to use those found features to produce the steering angle.

The dropout layer was added to try to combat overfitting. At first I tried to use batch normalization, but
it went horribly wrong. I suspect that, if I had good data, BN could work much much better, but this is 
not the case. This might actually happen because when using BN, the network learns 'too' well and easily
fits the data, trouble is, the data is not very good.

The last layer has no activation (linear) since this is a regression problem.

For optimization the loss function used was MSE, and the optimizar was ADAM. I had to reduce the default ADAM 
learning rate to 0.00001, as I had some weird problems before where the network would settle on a loss but really 
not learn anything.

## Validation

Ok, so I really believe that in this particular project, the only meaningfull validation is trying the
simulator itself. But anyhow, I geerated a dataset consisting of two normal laps, and used that as the
validation set. Since I had to use generators, and the data itself is augmented, if only a bit, I did not
want to use a subset of the generated datasets for training (which I think is reasonable). Since the training
strategy was first try a few very big epochs and testing each manually until the NN started to do something
more or less passable, no validation set was used in here.

Later on, when adding even more data for difficult sections, for refining the network (also using smaller
epochs), I used the validation set to monitor the NN. While lower validation losses did NOT mean better networks, 
it was handy to know when to stop training and produce candidate networks to test.

## Performance

The model works. Though it still has troubles at very steep curves. I do not see an easy way to solve this
if using the keyboard to generate the data. But it works! Even after feeding it such bad data. I have even
seen that much simpler networks could also work, though it seems that such students used a lot of
data augmentation, and also a proper joystick.