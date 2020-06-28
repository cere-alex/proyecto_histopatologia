from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD, Adam, Adagrad
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import cv2
import seaborn as sns
from random import shuffle

# Total Samples Available
print('Train Images = ', len(os.listdir('./train')))
print('Test Images = ', len(os.listdir('./test')))

df = pd.read_csv('./train_labels.csv')
print('Shape of DataFrame', df.shape)
df.head()


TRAIN_DIR = './train/'

fig = plt.figure(figsize=(20, 8))
index = 1
for i in np.random.randint(low=0, high=df.shape[0], size=10):
    file = TRAIN_DIR + df.iloc[i]['id'] + '.tif'
    img = cv2.imread(file)
    ax = fig.add_subplot(2, 5, index)
    ax.imshow(img, cmap='gray')
    index = index + 1
    color = ['green' if df.iloc[i].label == 1 else 'red'][0]
    ax.set_title(df.iloc[i].label, fontsize=18, color=color)
plt.tight_layout()
plt.show()

# removing this image because it caused a training error previously
df[df['id'] != 'dd6dfed324f9fcb6f93f46f32fc800f2ec196be2']
# img = cv2.imread('./train/dd6dfed324f9fcb6f93f46f32fc800f2ec196be2.tif')
# plt.imshow(img)
# removing this image because it's black
df[df['id'] != '9369c7278ec8bcc6c880d99194de09fc2bd4efbe']
df.head()

fig = plt.figure(figsize=(6, 6))
ax = sns.countplot(df.label).set_title('Label Counts', fontsize=18)
plt.annotate(df.label.value_counts()[0],
             xy=(0, df.label.value_counts()[0] + 2000),
             va='bottom',
             ha='center',
             fontsize=12)
plt.annotate(df.label.value_counts()[1],
             xy=(1, df.label.value_counts()[1] + 2000),
             va='bottom',
             ha='center',
             fontsize=12)
plt.ylim(0, 150000)
plt.ylabel('Count', fontsize=16)
plt.xlabel('Labels', fontsize=16)
plt.show()


SAMPLE_SIZE = 80000
# take a random sample of class 0 with size equal to num samples in class 1
df_0 = df[df['label'] == 0].sample(SAMPLE_SIZE, random_state=0)
# filter out class 1
df_1 = df[df['label'] == 1].sample(SAMPLE_SIZE, random_state=0)

# concat the dataframes
df_train = pd.concat([df_0, df_1], axis=0).reset_index(drop=True)
# shuffle
df_train = df_train.sample(frac=1).reset_index(drop=True)
df_train['label'].value_counts()

y = df_train['label']
df_train, df_val = train_test_split(df_train, test_size=0.1, random_state=0, stratify=y)


#################################

base_dir = 'base_dir'
os.mkdir(base_dir)


# Folder Structure

'''
    * base_dir
        |-- train_dir
            |-- 0   #No Tumor
            |-- 1   #Has Tumor
        |-- val_dir
            |-- 0
            |-- 1
'''
# create a path to 'base_dir' to which we will join the names of the new folders
# train_dir
train_dir = os.path.join(base_dir, 'train_dir')
os.mkdir(train_dir)

# val_dir
val_dir = os.path.join(base_dir, 'val_dir')
os.mkdir(val_dir)

# create new folders inside train_dir
no_tumor = os.path.join(train_dir, '0')
os.mkdir(no_tumor)
has_tumor = os.path.join(train_dir, '1')
os.mkdir(has_tumor)


# create new folders inside val_dir
no_tumor = os.path.join(val_dir, '0')
os.mkdir(no_tumor)
has_tumor = os.path.join(val_dir, '1')
os.mkdir(has_tumor)


print(os.listdir('base_dir/train_dir'))
print(os.listdir('base_dir/val_dir'))


#########################################

# Set the id as the index in df_data
df.set_index('id', inplace=True)

# Get a list of train and val images
train_list = list(df_train['id'])
val_list = list(df_val['id'])


# Transfer the train images
for image in train_list:

    # the id in the csv file does not have the .tif extension therefore we add it here
    file_name = image + '.tif'
    # get the label for a certain image
    target = df.loc[image, 'label']

    # these must match the folder names
    if target == 0:
        label = '0'
    elif target == 1:
        label = '1'

    # source path to image
    src = os.path.join('./train', file_name)
    # destination path to image
    dest = os.path.join(train_dir, label, file_name)
    # copy the image from the source to the destination
    shutil.copyfile(src, dest)


# Transfer the val images

for image in val_list:

    # the id in the csv file does not have the .tif extension therefore we add it here
    file_name = image + '.tif'
    # get the label for a certain image
    target = df.loc[image, 'label']

    # these must match the folder names
    if target == 0:
        label = '0'
    elif target == 1:
        label = '1'

    # source path to image
    src = os.path.join('./train', file_name)
    # destination path to image
    dest = os.path.join(val_dir, label, file_name)
    # copy the image from the source to the destination
    shutil.copyfile(src, dest)


print(len(os.listdir('base_dir/train_dir/0')))
print(len(os.listdir('base_dir/train_dir/1')))

#########################################

IMAGE_SIZE = 96
train_path = 'base_dir/train_dir'
valid_path = 'base_dir/val_dir'
test_path = './test'

num_train_samples = len(df_train)
num_val_samples = len(df_val)
train_batch_size = 32  # 10
val_batch_size = 32  # 10

train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)


datagen = ImageDataGenerator(rescale=1.0/255)

train_gen = datagen.flow_from_directory(train_path,
                                        target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                        batch_size=train_batch_size,
                                        class_mode='categorical')

val_gen = datagen.flow_from_directory(valid_path,
                                      target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                      batch_size=val_batch_size,
                                      class_mode='categorical')

# Note: shuffle=False causes the test dataset to not be shuffled
test_gen = datagen.flow_from_directory(valid_path,
                                       target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                       batch_size=1,
                                       class_mode='categorical',
                                       shuffle=False)


class CancerNet:
    @staticmethod
    def build(width, height, depth, classes):

        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # CONV => RELU => POOL
        model.add(SeparableConv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # (CONV => RELU => POOL) * 2
        model.add(SeparableConv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(SeparableConv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # # (CONV => RELU => POOL) * 3
        # model.add(SeparableConv2D(128, (3, 3), padding="same"))
        # model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chanDim))
        # model.add(SeparableConv2D(128, (3, 3), padding="same"))
        # model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chanDim))
        # model.add(SeparableConv2D(128, (3, 3), padding="same"))
        # model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chanDim))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        model.summary()

        # return the constructed network architecture
        return model


model = CancerNet.build(width=96, height=96, depth=3, classes=2)
# model = CancerNet.build(width = 96, height = 96, depth = 3, classes = 2)
# Edit:: Adagrad(lr=1e-2, decay= 1e-2/10) was used previous;y
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
plot_model(model, to_file='model.png')
#########################

filepath = "checkpoint.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
                             save_best_only=True, mode='max')  # Save Best Epoch

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2,
                              verbose=1, mode='max', min_lr=0.00001)
callbacks_list = [checkpoint, reduce_lr]  # LR Scheduler Used here

history = model.fit_generator(train_gen, steps_per_epoch=train_steps,
                              validation_data=val_gen,
                              validation_steps=val_steps,
                              epochs=11,
                              verbose=1,
                              callbacks=callbacks_list)

# Plot training & validation accuracy values

history.history
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='best')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='best')
plt.show()
