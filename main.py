#%% Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout

#%% Data 

train_data_path = 'C:/Users/Asus/Desktop/test_project_mnist/data/mnist_train.csv'
test_data_path = 'C:/Users/Asus/Desktop/test_project_mnist/data/mnist_test.csv'
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

#%% Seperating data 

labels_train_data = train_data['label']
labels_test_data = test_data['label']
print("\nLabels of the train data:\n")
print(labels_train_data)
print("\nLabels of the test data:\n")
print(labels_test_data)
features_train_data = train_data.drop('label', axis=1)
features_test_data = test_data.drop('label', axis=1)
print("\nFeatures (pixel values) of the train data, without labels:\n")
print(features_train_data)
print("\nFeatures (pixel values) of the test data, without labels:\n")
print(features_test_data)

#%% Setting classes

number_classes = 10
labels_train_data = keras.utils.to_categorical(labels_train_data, number_classes)
labels_test_data = keras.utils.to_categorical(labels_test_data, number_classes)

#%% Normalizing data

features_train_data = features_train_data / 255.0
features_test_data = features_test_data / 255.0

#%% Shaping data

features_train_data = np.array(features_train_data)
features_test_data = np.array(features_test_data)

features_train_data = features_train_data.reshape(features_train_data.shape[0], -1)
features_test_data = features_test_data.reshape(features_test_data.shape[0], -1)

#%% Creating Neural Network

model = Sequential()

model.add(Dense(units=128, input_shape=(784,), activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

#%% Training

model.fit(x=features_train_data, y=labels_train_data, batch_size=512, epochs=10)

#%% Loss Function and Accuracy Performance

test_loss, test_acc = model.evaluate(features_test_data, labels_test_data)
print("Test Loss: {}, Test Accuracy: {}".format(test_loss, test_acc))

#%% Prediction

prediction = model.predict(features_test_data)
prediction_classes = np.argmax(prediction, axis=1)
print("\nPredictions:\n", prediction)
print("\nPrediction Classes:\n", prediction_classes)

#%% Checking a Random Image Prediction in Test Dataset

random_number = np.random.choice(len(prediction))

print(f"Random number is: {random_number}\n")

print("Result of the prediction (possibilities/output of softmax activation function) as an array: \n", prediction[random_number])

print("\nHighest value in the array:\n", np.max(prediction[random_number]))

print(f"\nTrue class label of the '{random_number}' numbered data:\n", labels_test_data[random_number])

print("\nIndex of highest value:\n", np.argmax(prediction[random_number]))

print(f"\nThe prediction as a number:\n{np.argmax(prediction[random_number])}") 

print(f"\nTrue label:\n{np.argmax(labels_test_data, axis=1)[random_number]}") 

if (np.argmax(labels_test_data, axis=1)[random_number]) == (np.argmax(prediction[random_number])):
    print("\n*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
    print("\nTHE PREDICTION IS CORRECT!\n")
    print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
    
#%% Checking a Random Image Prediction in Test Dataset with Plotting

random_number = np.random.choice(len(features_test_data))

feature_sample = features_test_data[random_number]
label_true = np.argmax(labels_test_data, axis=1)
label_sample_true = label_true[random_number]
label_sample_pred_class = prediction_classes[random_number]

plt.title("Predicted: {}, Real Label: {}".format(label_sample_pred_class, label_sample_true))
plt.imshow(feature_sample.reshape(28, 28), cmap='gray')

#%% Prediction of External Image

from tensorflow.keras.preprocessing.image import img_to_array, load_img

image_path = r'C:\Users\Asus\Desktop\test_image\canan2.png'
image = load_img(image_path, color_mode='grayscale', target_size=(28, 28))
image = img_to_array(image)

image = image.reshape(1, 784)
image = image.astype('float32') / 255.0

real_label = 2
myprediction = model.predict(image)
mypredicted_class = np.argmax(myprediction, axis=1)
print("Predicted class:", mypredicted_class)
plt.title("Predicted: {}, Real Label: {}".format(mypredicted_class, real_label))
plt.imshow(image.reshape(28, 28), cmap='gray')
 
