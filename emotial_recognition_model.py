import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

df = pd.read_csv('fer2013.csv')

# print(df.head())

# sns.countplot(x=df['emotion'])
# plt.show()

#Обработка пикселей
def preproces_pixels(pixels):
    pixels = np.array(pixels.split(), dtype=np.float32)
    pixels = pixels.reshape(48,48, 1)
    pixels /= 255.0
    return pixels

df['pixels'] = df['pixels'].apply(preproces_pixels)
X = np.stack(df['pixels'].values)
Y = to_categorical(df['emotion'], num_classes = 7)
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)
# print(x_train.shape)
# print(x_test.shape)

#
# plt.figure(figsize = (10,5))
# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.imshow(x_train[i].reshape(48,48), cmap='gray')
#     plt.title(np.argmax(y_train[i]))
# plt.show()



#Создание Модели
model = Sequential([
    Conv2D(32, (3,3), activation = 'relu', input_shape = (48,48,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation = 'relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation = 'relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation = 'relu'),
    Dropout(0.5),
    Dense(7, activation = 'softmax')
])

model.compile(optimizer = Adam(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(x_train, y_train, validation_data = (x_test,y_test), epochs=20, batch_size = 64, verbose = 1)
loss, accuracy = model.evaluate(x_test, y_test)
# print(accuracy*100)


model.save('live_face_recognition_model.h5')
#Обработка емоций в реальном времени

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48,48))
        face = np.expand_dims(face, axis=[0,-1]) / 255.0
        emotion_pred = model.predict(face)
        emotion_label = np.argmax(emotion_pred)
        labels = ["Злость","Отвращение","Страх","Счастье","Грусть","Удивление","Нейтральное"]
        text = labels[emotion_label]
        color = (0,255,0) if emotion_label == 3 else (0,0,255) 
        cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)       
    cv2.imshow('Face and Emotion Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

