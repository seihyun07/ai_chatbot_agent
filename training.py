# training.py
import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True) 
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()

try:
    with open("intents.json", "r", encoding="utf-8-sig") as file:
        intents = json.load(file)
except FileNotFoundError:
    raise Exception("오류: 'intents.json' 파일이 존재하지 않습니다.")

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',', '을', '를', '이', '가', '에서']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

with open('words.pkl', 'wb') as f:
    pickle.dump(words, f)
with open('classes.pkl', 'wb') as f:
    pickle.dump(classes, f)

training_data_x = []
training_data_y = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
        
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    
    training_data_x.append(bag)
    training_data_y.append(output_row)

combined = list(zip(training_data_x, training_data_y))
random.shuffle(combined)
training_data_x, training_data_y = zip(*combined)

train_x = np.array(training_data_x)
train_y = np.array(training_data_y)

print(f"모델 훈련 준비 - Input 차원: {train_x.shape}, Output 차원: {train_y.shape}")

model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(len(train_x[0]),)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

print("인공신경망 학습 시작...")
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

model.save('chatbot_model.keras')
print("훈련 완료! 'chatbot_model.keras' 모델이 업데이트되었습니다.")