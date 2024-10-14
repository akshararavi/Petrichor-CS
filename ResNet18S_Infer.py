# Importing the libraries
import os
import sys
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import tensorflow as tf
import pandas as pd
import seaborn as sns
from tensorflow import keras
from keras.models import load_model
from keras.models import Model
import matplotlib.pyplot as plt
from sklearn import metrics
from keras.utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.resnet_v2 import preprocess_input

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from numpy import expand_dims
from numpy import unravel_index
from PIL import Image
import csv
import subprocess
from subprocess import call

# Setting the path for the dataset
path="./RGB_Data224/test"
model_path="./Small_Models"
output_path="./Small_Outputs"

# Defining hyperparameters
batch_testsize = 32
learning_rate = 0.001
weight_decay = 0.0001

model = load_model(os.path.join(model_path, "ResNet18.keras"), compile=False)

# Show the model architecture
model.summary()
plot_model(model, show_shapes=True, show_layer_names=True, to_file=os.path.join(model_path, 'ResNet18.png'))
plt.close()

'''************************************Inference****************************************'''

#Loading data
X = []
y = []
imgnames = []
Files = ['Benignware', 'Malware']
label_val = 0

for files in Files:
    cpath = os.path.join(path, files)
    for img in os.listdir(cpath):
        img_path = os.path.join(cpath, img)
        image_array = cv2.imread(img_path, cv2.IMREAD_COLOR)
        X.append(image_array)
        y.append(label_val)
        imgnames.append(img)
    label_val = 1

X_test = np.asarray(X)
y_test = np.asarray(y)
print(f'Shape of X_test: {X_test.shape}')
print(f'Shape of y_test: {y_test.shape}')  

# Calculating the steps per epoch
STEP_SIZE_TEST = int( np.ceil(X_test.shape[0] / batch_testsize) )

# Normalize the data.
X_test = X_test.astype('float32')
X_test /= 255.0

# Compiling the model
optimizer = tf.keras.optimizers.AdamW(
	learning_rate=learning_rate, weight_decay=weight_decay
)
model.compile(
	optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
)

# Checking the accuracy
checkpoint_filepath = os.path.join(model_path, "resnet18best.h5")
model.load_weights(checkpoint_filepath)
_, accuracy, top_5_accuracy = model.evaluate(X_test, y_test, batch_size=batch_testsize)
print(f"Test accuracy: {round(accuracy * 100, 2)}%")
print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")
    
# Making predictions
#print(X_test[0])
y_pred=model.predict(X_test, steps=STEP_SIZE_TEST)
y_pred=np.argmax(y_pred,axis=1)
    
labels = {'Benignware': 0, 'Malware': 1}
print(f"Labels and their corresponding encodings: {labels}")
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in y_pred]
    
# Get the classification report
print(classification_report(y_pred,y_test))
# Get the confusion matrix
conf_mat = confusion_matrix(y_pred,y_test)

# create a data-frame from the confusion matrix and plot it as a heat-map using the seaborn library.
confusion_matrix_df = pd.DataFrame(conf_mat).rename(columns=labels, index=labels)
fig, ax = plt.subplots(figsize=(20,10))         
sns.heatmap(confusion_matrix_df, annot=True, fmt='d', ax=ax)
plt.savefig(os.path.join(output_path, "confusion_matrix_ResNet18.png"))
plt.close()
