import cv2 as csv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

#This is a dataset of 50,000 32x32 color training images and 10,000 test images, labeled over 10 categories. See more info at the CIFAR homepage.
(training_images,training_labels),(testing_images,testing_labels)=datasets.cifar10.load_data() #returns data in the form the given tuple

#Normalising the data. Each pixel has an activation of 0-255 and so for normalising we are diving it by f(x)=(x-0)/(255-0)
training_images,testing_images=training_images/255,testing_images/255

#These are the objects that the neural network should be able to classify
#Note that the order of the class_names are important. Here the labels are in order with the class labels of cifar10
class_names=['Plane',"Car","Bird","Cat","Deer","Frog","Horse","Ship","Truck"]


#Reducing the size the of the dataset to fasten up the training process of the neural network
training_images= training_images[:20000]
training_labels= training_labels[:20000]
testing_images=testing_images[:4000]
testing_labels=testing_labels[:4000]

model=models.Sequential()
#This convolution network filters for features in an image
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))

#This reduces the image to just essential information
# model.add(layers.MaxPooling2D(2,2))
# model.add(layers.Conv2D(64,(3,3),activation='relu'))
# model.add(layers.MaxPooling2D(2,2))
# model.add(layers.Conv2D(64,(3,3),activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64,activation='relu'))
# model.add(layers.Dense(10,activation='softmax'))

# model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# model.fit(training_images,training_labels,epochs=10,validation_data=(testing_images,testing_labels))

# loss,accuracy=model.evaluate(testing_images,testing_labels)
# print(f"Loss: {loss}")
# print(f"Accuracy: {accuracy}")

# model.save('image_classifier.model')
model=models.load_model('image_classifier.model')
img=csv.imread('plane.jpeg')
img=csv.cvtColor(img,csv.COLOR_BGR2RGB)

plt.imshow(img,cmap=plt.cm.binary)

prediction=model.predict(np.array([img])/255)
index=np.argmax(prediction)
print(f"Prediction : {class_names[index]}")





