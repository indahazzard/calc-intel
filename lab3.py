import os
import zipfile
import random
import shutil
import tensorflow as tf
import matplotlib.pyplot as plt
from shutil import copyfile 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

local_zip = '/tmp/cats-and-dogs.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()

source_path = '/tmp/PetImages'
source_path_dogs = os.path.join(source_path, 'Dog')
source_path_cats = os.path.join(source_path, 'Cat')


print(f"There are {len(os.listdir(source_path_dogs))} images of dogs")

print(f"There are {len(os.listdir(source_path_cats))} images of cats")

root_dir = '/tmp/cats-v-dogs'

if os.path.exists(root_dir):
    shutil.rmtree(root_dir)

def create_train_val_dirs(root_path):
    train_dir = os.path.join(root_path, 'training')
    validation_dir = os.path.join(root_path, 'validation')

    training_cats_dir = os.path.join(train_dir, 'cats')
    training_dogs_dir = os.path.join(train_dir, 'dogs')

    validation_cats_dir = os.path.join(validation_dir, 'cats')
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')

    os.makedirs(training_cats_dir)
    os.makedirs(training_dogs_dir)
    os.makedirs(validation_cats_dir)
    os.makedirs(validation_dogs_dir)

    train_cat_dir = os.listdir(training_cats_dir)
    train_dog_dir = os.listdir(training_dogs_dir)

def split_data(SOURCE_DIR, TRAINING_DIR, VALIDATION_DIR, SPLIT_SIZE):
    stripped_names = os.listdir(SOURCE_DIR)

    for name in os.listdir(SOURCE_DIR):
        if os.path.getsize(SOURCE_DIR +'/'+name) == 0:
            print(name + ' is zero length, so ignoring.')
            stripped_names.remove(name)
    
    shuff_names = random.sample(stripped_names, len(stripped_names))

    for name in shuff_names[:int(len(stripped_names) * SPLIT_SIZE)]:
        copyfile(SOURCE_DIR+ '/' + name, TRAINING_DIR+ '/' + name)

    for name in shuff_names[int(len(stripped_names)*SPLIT_SIZE):]:
        copyfile(SOURCE_DIR + '/' + name, VALIDATION_DIR + '/' + name)

def train_val_generators(TRAINING_DIR, VALIDATION_DIR):
    train_datagen = ImageDataGenerator(rescale=(1./255))
    
    train_generator = train_datagen.flow_from_directory(directory=TRAINING_DIR,
                                                        batch_size=20,
                                                        class_mode='binary',
                                                        target_size=(150,150))

    validation_datagen = ImageDataGenerator(rescale=(1./255))

    validation_generator = validation_datagen.flow_from_directory(directory=VALIDATION_DIR,
                                                                  batch_size=20,
                                                                  class_mode='binary',
                                                                  target_size=(150,150))

    return train_generator, validation_generator

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150,150,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


create_train_val_dirs(root_dir)

training_dir_dog = '/tmp/cats-v-dogs/training/dogs'
training_dir_cat = '/tmp/cats-v-dogs/training/cats'

validation_dir_dog = '/tmp/cats-v-dogs/validation/dogs'
validation_dit_cat = '/tmp/cats-v-dogs/validation/cats'

split_data(source_path_dogs, training_dir_dog, validation_dir_dog, SPLIT_SIZE=0.9)
split_data(source_path_cats, training_dir_cat, validation_dit_cat, SPLIT_SIZE=0.9)

training_dir = '/tmp/cats-v-dogs/training'
validation_dir = '/tmp/cats-v-dogs/validation'
train_generator, validation_generator = train_val_generators(training_dir, validation_dir)

model = create_model()

history = model.fit(train_generator,
                    epochs=15,
                    verbose=1,
                    validation_data=validation_generator)

acc=history.history['accuracy'] 
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(len(acc))

plt.plot(epochs, acc, 'r', "Training Accuracy") 
plt.plot(epochs, val_acc, 'b', "Validation Accuracy") 
plt.title('Training and validation accuracy') 
plt.show()

print("")

plt.plot(epochs, loss, 'r', "Training Loss") 
plt.plot(epochs, val_loss, 'b', "Validation Loss") 
plt.show()
