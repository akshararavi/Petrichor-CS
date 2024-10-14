# Importing the libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import csv
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D, BatchNormalization, Add
from keras.models import Sequential
from keras.models import Model

# Setting the path for the dataset
path="./RGBL_Data224/train"
model_path="./Large_Models"
output_path="./Large_Outputs"

# Defining hyperparameters
num_classes = 2
IMG_SIZE = 224
input_shape = (IMG_SIZE, IMG_SIZE, 3)
learning_rate = 0.001
weight_decay = 0.0001
num_epochs = 50
batch_size = 32

# Creating the datasets
X = []
y = []
Files = ['Benignware', 'Malware']
label_val = 0

for files in Files:
    cpath = os.path.join(path, files)
    for img in os.listdir(cpath):
        image_array = cv2.imread(os.path.join(cpath, img), cv2.IMREAD_COLOR)
        X.append(image_array)
        y.append(label_val)
    label_val = 1

X = np.asarray(X)
y = np.asarray(y)

# Set aside 20% for validation set and remaining for training set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state= 8) 

# Checking the shape of the datasets
print(f"X_train: {X_train.shape} - y_train: {y_train.shape}")
print(f"X_val: {X_val.shape} - y_val: {y_val.shape}")
#print(f"X_test: {X_test.shape} - y_test: {y_test.shape}")

# Calculating the steps per epoch
#STEP_SIZE_TRAIN = training_set.n // training_set.batch_size
STEP_SIZE_TRAIN = int( np.ceil(X_train.shape[0] / batch_size) )
#STEP_SIZE_VALID = val_set.n // val_set.batch_size
STEP_SIZE_VALID = int( np.ceil(X_val.shape[0] / batch_size) )

# Normalize the data.
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_train /= 255
X_val /= 255

"""
ResNet-18
"""

class ResNet18(Model):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
    
        #self.relu = ReLU()
        
        # BLOCK-1 (starting block) input=(224x224) output=(56x56)
        self.conv1 = Conv2D(64, (7, 7), strides=2, padding="same", kernel_initializer="he_normal")
        self.batchnorm1 = BatchNormalization()
        self.maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding="same")
        
        # BLOCK-2 (1) input=(56x56) output = (56x56)
        self.conv2_1_1 = Conv2D(64, (3, 3), strides=1, padding="same", kernel_initializer="he_normal")
        self.batchnorm2_1_1 = BatchNormalization()
        self.conv2_1_2 = Conv2D(64, (3, 3), strides=1, padding="same", kernel_initializer="he_normal")
        self.batchnorm2_1_2 = BatchNormalization()
        self.add2_1_1 = Add()
        # BLOCK-2 (2)
        self.conv2_2_1 = Conv2D(64, (3, 3), strides=1, padding="same", kernel_initializer="he_normal")
        self.batchnorm2_2_1 = BatchNormalization()
        self.conv2_2_2 = Conv2D(64, (3, 3), strides=1, padding="same", kernel_initializer="he_normal")
        self.batchnorm2_2_2 = BatchNormalization()
        self.add2_2_1 = Add()
        
        # BLOCK-3 (1) input=(56x56) output = (28x28)
        self.conv3_1_1 = Conv2D(128, (3, 3), strides=2, padding="same", kernel_initializer="he_normal")
        self.batchnorm3_1_1 = BatchNormalization()
        self.conv3_1_2 = Conv2D(128, (3, 3), strides=1, padding="same", kernel_initializer="he_normal")
        self.batchnorm3_1_2 = BatchNormalization()
        self.add3_1_1 = Add()
        self.conv3_1_3 = Conv2D(128, (1, 1), strides=2, padding="same", kernel_initializer="he_normal")
        self.batchnorm3_1_3 = BatchNormalization()
        # BLOCK-3 (2)
        self.conv3_2_1 = Conv2D(128, (3, 3), strides=1, padding="same", kernel_initializer="he_normal")
        self.batchnorm3_2_1 = BatchNormalization()
        self.conv3_2_2 = Conv2D(128, (3, 3), strides=1, padding="same", kernel_initializer="he_normal")
        self.batchnorm3_2_2 = BatchNormalization()
        self.add3_2_1 = Add()
        
        # BLOCK-4 (1) input=(28x28) output = (14x14)
        self.conv4_1_1 = Conv2D(256, (3, 3), strides=2, padding="same", kernel_initializer="he_normal")
        self.batchnorm4_1_1 = BatchNormalization()
        self.conv4_1_2 = Conv2D(256, (3, 3), strides=1, padding="same", kernel_initializer="he_normal")
        self.batchnorm4_1_2 = BatchNormalization()
        self.add4_1_1 = Add()
        self.conv4_1_3 = Conv2D(256, (1, 1), strides=2, padding="same", kernel_initializer="he_normal")
        self.batchnorm4_1_3 = BatchNormalization()
        # BLOCK-4 (2)
        self.conv4_2_1 = Conv2D(256, (3, 3), strides=1, padding="same", kernel_initializer="he_normal")
        self.batchnorm4_2_1 = BatchNormalization()
        self.conv4_2_2 = Conv2D(256, (3, 3), strides=1, padding="same", kernel_initializer="he_normal")
        self.batchnorm4_2_2 = BatchNormalization()
        self.add4_2_1 = Add()
        
        # BLOCK-5 (1) input=(14x14) output = (7x7)
        self.conv5_1_1 = Conv2D(512, (3, 3), strides=2, padding="same", kernel_initializer="he_normal")
        self.batchnorm5_1_1 = BatchNormalization()
        self.conv5_1_2 = Conv2D(512, (3, 3), strides=1, padding="same", kernel_initializer="he_normal")
        self.batchnorm5_1_2 = BatchNormalization()
        self.add5_1_1 = Add()
        self.conv5_1_3 = Conv2D(512, (1, 1), strides=2, padding="same", kernel_initializer="he_normal")
        self.batchnorm5_1_3 = BatchNormalization()
        # BLOCK-5 (2)
        self.conv5_2_1 = Conv2D(512, (3, 3), strides=1, padding="same", kernel_initializer="he_normal")
        self.batchnorm5_2_1 = BatchNormalization()
        self.conv5_2_2 = Conv2D(512, (3, 3), strides=1, padding="same", kernel_initializer="he_normal")
        self.batchnorm5_2_2 = BatchNormalization()
        self.add5_2_1 = Add()
        
        # Final Block input=(7x7) 
        self.avgpool = GlobalAveragePooling2D()
        #self.flat = Flatten()
        self.fc = Dense(num_classes, activation="softmax")
        # END
        
    def get_config(self):
        config = {
            "num_classes" : num_classes,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def call(self, x):
        
        # block 1 --> Starting block
        x = tf.nn.relu(self.batchnorm1(self.conv1(x)))
        op1 = self.maxpool1(x)
        
        
        # block2 - 1
        x = tf.nn.relu(self.batchnorm2_1_1(self.conv2_1_1(op1)))    # conv2_1 
        x = self.batchnorm2_1_2(self.conv2_1_2(x))                 # conv2_1
        x = self.add2_1_1([x, op1])
        # block2 - Adjust - No adjust in this layer as dimensions are already same
        # block2 - Concatenate 1
        op2_1 = tf.nn.relu(x)
     
        # block2 - 2
        x = tf.nn.relu(self.batchnorm2_2_1(self.conv2_2_1(op2_1)))  # conv2_2 
        x = self.batchnorm2_2_2(self.conv2_2_2(x))                 # conv2_2
        x = self.add2_2_1([x, op2_1])
        # op - block2
        op2 = tf.nn.relu(x)
    
        
        # block3 - 1[Convolution block]
        x = tf.nn.relu(self.batchnorm3_1_1(self.conv3_1_1(op2)))    # conv3_1
        x = self.batchnorm3_1_2(self.conv3_1_2(x))                 # conv3_1
        # block3 - Adjust
        op2 = self.conv3_1_3(op2) # SKIP CONNECTION
        op2 = self.batchnorm3_1_3(op2)
        # block3 - Concatenate 1
        x = self.add3_1_1([x, op2])
        op3_1 = tf.nn.relu(x)
        # block3 - 2[Identity Block]
        x = tf.nn.relu(self.batchnorm3_2_1(self.conv3_2_1(op3_1)))  # conv3_2
        x = self.batchnorm3_2_2(self.conv3_2_2(x))                 # conv3_2 
        x = self.add3_2_1([x, op3_1])
        # op - block3
        op3 = tf.nn.relu(x)
        
        
        # block4 - 1[Convolition block]
        x = tf.nn.relu(self.batchnorm4_1_1(self.conv4_1_1(op3)))    # conv4_1
        x = self.batchnorm4_1_2(self.conv4_1_2(x))                 # conv4_1
        # block4 - Adjust
        op3 = self.conv4_1_3(op3) # SKIP CONNECTION
        op3 = self.batchnorm4_1_3(op3)
        # block4 - Concatenate 1
        x = self.add4_1_1([x, op3])
        op4_1 = tf.nn.relu(x)
        # block4 - 2[Identity Block]
        x = tf.nn.relu(self.batchnorm4_2_1(self.conv4_2_1(op4_1)))  # conv4_2
        x = self.batchnorm4_2_2(self.conv4_2_2(x))                 # conv4_2
        x = self.add4_2_1([x, op4_1])
        # op - block4
        op4 = tf.nn.relu(x)

        
        # block5 - 1[Convolution Block]
        x = tf.nn.relu(self.batchnorm5_1_1(self.conv5_1_1(op4)))    # conv5_1
        x = self.batchnorm5_1_2(self.conv5_1_2(x))                 # conv5_1
        # block5 - Adjust
        op4 = self.conv5_1_3(op4) # SKIP CONNECTION
        op4 = self.batchnorm5_1_3(op4)
        # block5 - Concatenate 1
        x = self.add5_1_1([x, op4])
        op5_1 = tf.nn.relu(x)
        # block5 - 2[Identity Block]
        x = tf.nn.relu(self.batchnorm5_2_1(self.conv5_2_1(op5_1)))  # conv5_2
        x = self.batchnorm5_2_1(self.conv5_2_2(x))                 # conv5_2
        x = self.add5_2_1([x, op5_1])
        # op - block5
        op5 = tf.nn.relu(x)

        # FINAL BLOCK - classifier 
        x = self.avgpool(op5)
        #x = self.flat(x)
        x = self.fc(x)

        return x
        
    def model(self):
    	x = tf.keras.layers.Input(input_shape)
    	return tf.keras.Model(inputs=[x], outputs=self.call(x))
    	
model = ResNet18(num_classes)
model.build(input_shape = (None,IMG_SIZE,IMG_SIZE,3))
res_model = model.model()
res_model.summary()

# Compiling the model
optimizer = tf.keras.optimizers.AdamW(
	learning_rate=learning_rate, weight_decay=weight_decay
)

res_model.compile(
	optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
)

# Save the Keras model weights at some frequency (here, when the best validation accuracy is achieved)
checkpoint_filepath = os.path.join(model_path, "resnet18best.h5")
checkpoint_callback = ModelCheckpoint(
	checkpoint_filepath,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        save_weights_only=True,
)
# Early stopping to avoid overfitting of model when there is a contiguous increase of validation loss for more than 5 epochs
early_stop=EarlyStopping(patience= 8, restore_best_weights=True, monitor="val_accuracy")
callbacks_list = [checkpoint_callback, early_stop]

# Training the model
history = res_model.fit(
	x=X_train,
        y=y_train,
        steps_per_epoch=STEP_SIZE_TRAIN,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data=(X_val,y_val),
        validation_steps=STEP_SIZE_VALID,
        callbacks=callbacks_list,
        shuffle=True
)
   
res_model.load_weights(os.path.join(model_path, "resnet18best.h5"))

# Save the model.
res_model.save(os.path.join(model_path, "ResNet18.keras"))
    
# Plotting the accuracies
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.savefig(os.path.join(output_path, 'Accuracy_ResNet18.png'))
plt.close()
    
# Plotting the loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.savefig(os.path.join(output_path, 'Loss_ResNet18.png'))
plt.close()
