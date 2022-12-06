import json
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import sys

training = np.genfromtxt('./assets/IMDB_dataset_2.csv', delimiter=';', skip_header=1, usecols=(1, 2), dtype=None, encoding="utf8")
train_x = [x[1] for x in training]
train_y = np.asarray([x[0] for x in training])

max_words = 3000

tokenizer = Tokenizer(num_words=max_words)

tokenizer.fit_on_texts(train_x)

dictionary = tokenizer.word_index

with open('dictionary.json', 'w') as dictionary_file:
    json.dump(dictionary, dictionary_file)


def convert_text_to_index_array(text):
    return [dictionary[word] for word in kpt.text_to_word_sequence(text)]


allWordIndices = []
for text in train_x:
    wordIndices = convert_text_to_index_array(text)
    allWordIndices.append(wordIndices)

allWordIndices = np.asarray(allWordIndices)

train_x = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')
train_y = keras.utils.to_categorical(train_y, 2)

model = Sequential()
model.add(Dense(512, input_shape=(max_words,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adamax', metrics=['accuracy'])

model.fit(train_x, train_y, batch_size=32, epochs=15, verbose=1,
          validation_split=0.1, shuffle=True)

model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

model.save('model.h5')

print('saved model!')
