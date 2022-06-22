#Switching CPU operation instructions to AVX AVX2
import os

from tensorflow.keras import metrics
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#Adding progression logging
import logging
logging.getLogger().setLevel(logging.INFO)
#Standard imports ^

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

"""
Training set

"""
train_labels = []
train_samples = []

for i in range(50):
  #The ~5% of younger individuals who did experience side effects
  random_younger = randint(13,64)
  train_samples.append(random_younger)
  train_labels.append(1)

  #The ~5% of older individuals who did not experience side effects
  random_older = randint(65,100)
  train_samples.append(random_older)
  train_labels.append(0)

for i in range(1000):
  #The ~95% of younger individuals who did not experience side effects
  random_younger = randint(13,64)
  train_samples.append(random_younger)
  train_labels.append(0)

  #The ~95% of older individuels who did experience side effects
  random_older = randint(65,100)
  train_samples.append(random_older)
  train_labels.append(1)

train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
train_labels, train_samples = shuffle(train_labels,train_samples)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_training_samples = scaler.fit_transform(train_samples.reshape(-1,1))

"""
Testing set

"""
test_labels = []
test_samples = []

for i in range(10):
  #The ~5% of younger individuals who did experience side effects
  random_younger = randint(13,64)
  test_samples.append(random_younger)
  test_labels.append(1)

  #The ~5% of older individuals who did not experience side effects
  random_older = randint(65,100)
  test_samples.append(random_older)
  test_labels.append(0)

for i in range(200):
  #The ~95% of younger individuals who did not experience side effects
  random_younger = randint(13,64)
  test_samples.append(random_younger)
  test_labels.append(0)

  #The ~95% of older individuels who did experience side effects
  random_older = randint(65,100)
  test_samples.append(random_older)
  test_labels.append(1)

test_labels = np.array(test_labels)
test_samples = np.array(test_samples)
test_labels, test_samples = shuffle(test_labels,test_samples)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_test_samples = scaler.fit_transform(test_samples.reshape(-1,1))


model = Sequential([
  Dense(units=16, input_shape=(1,), activation='relu'),
  Dense(units=32, activation='relu'),
  Dense(units=2, activation='softmax'),
])

model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x=scaled_training_samples, y=train_labels, validation_split=0.1, batch_size=10, epochs=30, shuffle=True, verbose=2)

predictions = model.predict(x=scaled_test_samples, batch_size=10, verbose=0)
for i in predictions:
  print(i)

rounded_predictions = np.argmax(predictions, axis=-1)
for i in rounded_predictions:
  print(i)


"""
Confusion Matrix part

"""
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true=test_labels, y_pred=rounded_predictions)

def plot_confusion_matrix(cm,classes,
                          normalize=False,
                          title='Confusion Matrix', 
                          cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by the setting 'normalize'
  """
  plt.imshow(cm,interpolation='nearest',cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1) [:,np.newaxis]
    print("Normalized confusion matrix")
  else:
    print("Confusion matrix, without normalization")

  print(cm)

  thresh = cm.max() / 2
  for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
    plt.text(j,i,cm[i,j],
             horizontalalignment="center",
             color="white" if cm[i,j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel("True Label")
  plt.xlabel("Predicted Label")
  plt.show()

cm_plot_labels = ["no_side_effects","had_side_effects"]
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title="Confusion Matrix")

"""
Save and load model

"""
#Checks first to see if file exists already.
#If not, the model is saved to disk.
import os.path
if os.path.isfile('KerasLearning/models/medical_trial_model.h5') is False:
  model.save('KerasLearning/models/medical_trial_model.h5')


"""
Model to json

"""
#Save as JSON
json_string = model.to_json()

"""
Model.save_weights()

"""
if os.path.isfile('KerasLearning/models/my_model_weights.h5') is False:
  model.save_weights('KerasLearning/models/my_model_weights.h5')