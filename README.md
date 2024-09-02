# Description

The program is an image recognition AI that can sort 32x32 images into ten different preset classes using a convoluted neural network.
This program uses convoluted neural networks, keras, and numpy conecpts.

## Step By Step Walkthrough

The program starts by loading the images and labels we need from keras.
Two tuples are created, one storing the training images and labels, and the other storing the testing images and labels.
Both the training images and testing images have an activation value between 0-255 based on how bright they are, so the program divides it by 255 so the activation value is between 0-1.
Next, the program makes a list of all ten class names we will be sorting the images into.
It is important to note that the list has to be in the exact order that it is defined in the code.
We then assign a variable each for the training images and labels, and the testing images and labels.
The training images and labels are assigned to the first 20000 in the dataset, and the testing images and labels are assigned to the first 4000 in the set.
The number of images and labels used can be increased, but it will greatly slow down the program for only a slight accuracy boost.
Next, models.Sequential() is used to define a sequential model that can add layers later in the code.
The first layer added is the convolutional input layer, defined with a 32x32 input, which is the amount of pixels in each image, and it is also defined to read three color waves (red green and blue).
The first layer is also defined to have 32 neurons that have a 3x3 filter.
The "relu" activation function, or rectified linear unit, is used for the input layer and all other convolutional layers due to its speed, gradient based optimization, and how it passes negative numbers as 0
A "MaxPooling2D" layer is defined with a 2x2 filter, which essentially finds the max value in each 2x2 window so mostly only the essential information is stored.
MaxPooling can cause an increase in loss, but is overall worth it due to its increase in accuracy.
Two more convolutional layers with another pooling layer in between are all defined.
These hidden layers use 64 neurons instead of 32 to add more paramters.
The matrix is then simplified into a one dimensional array, using .Flatten(), and this is done so the output is in vector form.
The next layer is defined as a dense layer, or a fully connected layer, which makes every single neuron compute the sum of the weight from its own inputs, essentially adding more paramters.
This layer also uses 64 neurons for increased accuracy.
The final layer is another dense layer, but this time, there are only ten neurons to match the ten classifications.
This layer also uses the "softmax" activation function instead of "relu" so that the output gets scaled to numbers that add up to one.
The data is then compiled using the optimizer "adam" which is known for its memory efficiency, adaptive learning rate, and its ability to work well with a lot of paramters.
The compiler also checks for accuracy and loss, and the loss function used is "sparse_categorical_crossentropy", which is used for its ability to use integer labels.
After compiling, the model is trained over ten epochs and then validated using the testing data.
An epoch is the amount of times the model sees the same data, so here, the model sees the same data ten times.
The validation data is then evaluated, and we find the model has a loss of 1.0211286544799805 and an accuracy of 0.6517500281333923, which is a solid number considering how low quality the images are.
Finally, the loss and accuracy are printed, and the model is saved so we can test our own images without having to retrain the model every time.

## Ways to improve the model
More in depth parameter tuning could work, but I found that the amound of neurons in the hidden layers and the amount of epochs already set works best.
If you have a strong computer, training on more data would make it slightly more accurate. 
Using images of higher resolution would make the program way more accurate, but it would be way slower due to the increase in amount of neurons.
The use of batch gradient descent would make the program more accurate, specifically vs other forms of gradient descent due to how different the images are, but this would be a very slow process.
