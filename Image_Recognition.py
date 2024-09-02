import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from keras import datasets, layers, models

#Stores data in a training tuple and a testing tuple
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
#Scales pixel activation number down to be between 0 and 1
#Divides by 255 since that is the highest pixel activation number
training_images, testing_images = training_images/255, testing_images/255

#Labels for all the images
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

#Uses the first 20000 images and labels to train with and uses the first 4000 images and labels to test with
#Can increase but will slow down the program
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

#Creates a sequential model that can add layers later
model = models.Sequential()
#Defines a convolutional layer with an input of 32x32 pixels with 3 different color waves (red green blue)
#Creates the layer with 32 neurons, since there are 32 features on the input data, and they filter on a 3x3 scale
#Uses rectified linear unit activation to pass negative values as zero and positive values as themselves
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape = (32, 32, 3)))
#Finds the maximum value in 2x2 windows and only keeps those max values, reducing the image to more essential information
#Its stride is the size of the fiter, so in this case, 2x2
model.add(layers.MaxPooling2D((2, 2)))
#Defines a hidden layer with 64 neurons instead of 32
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
#Creates one dimensional matrix, or an array, of all the data
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
#Output layer, ten units for ten classifications
#softmax scales so that the output is percentages that adds to one
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Trains the data and tests it using the testing images and labels
#Uses ten epochs, so that algorithm sees the same data ten different times
model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

#Finds and prints the loss and accuracy found while using the testing images and labels
loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f'loss: {loss}')
print(f'accuracy: {accuracy}')

#Saves the model so we only have to train it once, we can use the string to load the model
model.save('image_classifier.keras')

'''
model = models.load_model('image_classifier.keras')

img = cv.imread('pic.jpg')
#Swaps blue and red values (BGR to RGB)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)

#Divides every image by 255 so the color activation is a value between 0 and 1 inclusive
prediction = model.predict(np.array([img]) / 255)
index = np.argmax(prediction)
print(f'Prediction is {class_names[index]}')'''