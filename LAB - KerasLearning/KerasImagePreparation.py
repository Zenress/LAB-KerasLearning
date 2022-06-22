#Switching CPU operation instructions to AVX AVX2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#Adding progression logging
import logging
logging.getLogger().setLevel(logging.INFO)
#Standard imports ^

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense,Flatten,BatchNormalization,Conv2D,MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

os.chdir('Machine Learning Datasets/dogs-vs-cats')
if os.path.isdir('train/dog') is False:
  os.makedirs('train/dog')
  os.makedirs('train/cat')
  os.makedirs('valid/dog')
  os.makedirs('valid/cat')
  os.makedirs('test/dog')
  os.makedirs('test/cat')

for c in random.sample(glob.glob('cat*'), 500):
  shutil.move(c, 'train/cat' )
for c in random.sample(glob.glob('dog*'), 500):
  shutil.move(c, 'train/dog' )
for c in random.sample(glob.glob('cat*'), 100):
  shutil.move(c, 'valid/cat' )
for c in random.sample(glob.glob('dog*'), 100):
  shutil.move(c, 'valid/dog' )
for c in random.sample(glob.glob('cat*'), 50):
  shutil.move(c, 'test/cat' )
for c in random.sample(glob.glob('dog*'), 50):
  shutil.move(c, 'test/dog' )

os.chdir('../../')