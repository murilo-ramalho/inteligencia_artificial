import json
import pickle
import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Baixa os recursos do NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

lemmatizer = WordNetLemmatizer()

# Carrega as intenções
words = []
documents = []
intents = json.loads(open('intents.json', encoding='utf-8').read())  # UTF-8 para evitar problemas de codificação
classes = [i['tag'] for i in intents['intents']]
ignore_words = ["!", "@", "#", "$", "%", "*", "?"]

# Processa as intenções
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        documents.append((word, intent['tag']))

# Lematiza as palavras e ignora as palavras irrelevantes
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Salva as palavras e classes para uso futuro
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

# Prepara os dados para o treinamento
for document in documents:
    bag = []
    pattern_words = document[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    bag = [1 if word in pattern_words else 0 for word in words]

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
x = np.array([item[0] for item in training], dtype=np.float32)
y = np.array([item[1] for item in training], dtype=np.float32)

# Modelo de Rede Neural
model = Sequential()
model.add(Dense(128, input_shape=(len(x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y[0]), activation='softmax'))

# Compilação do Modelo
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Treinamento
model.fit(x, y, epochs=200, batch_size=5, verbose=1)
model.save('model.keras')

print("Treinamento concluído.")
