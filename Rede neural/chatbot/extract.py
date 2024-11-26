import random
import nltk
import json
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import pickle

# Carrega o modelo de rede neural treinado
model = load_model('model.keras')

# Carrega as intenções (com codificação UTF-8 para evitar erros de caracteres)
intents = json.loads(open('intents.json', encoding='utf-8').read())  # UTF-8 para evitar problemas de codificação

# Carrega as palavras e classes usadas no treinamento
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

lemmatizer = WordNetLemmatizer()

# Mapeamento para corrigir variações de gênero (masculino/feminino)
gender_map = {
    "obrigado": "obrigada",  # Exemplo para garantir que "obrigado" seja tratado como "obrigada"
    "bom": "boa",  # Garantir que "bom" seja tratado como "boa"
    # Adicione outras palavras conforme necessário
}

def clear_writing(writing):
    """
    Limpa todas as sentenças inseridas, corrigindo palavras com variações de gênero.
    """
    sentence_words = nltk.word_tokenize(writing)
    cleaned_words = []
    for word in sentence_words:
        word = lemmatizer.lemmatize(word.lower())
        # Verifica se a palavra tem uma variação de gênero que precisa ser padronizada
        if word in gender_map:
            word = gender_map[word]
        cleaned_words.append(word)
    return cleaned_words

def class_prediction(sentence, model):
    """
    Faz a previsão da classe para uma sentença dada.
    """
    sentence_words = clear_writing(sentence)
    bag_of_words = [0] * len(words)

    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag_of_words[i] = 1

    return model.predict(np.array([bag_of_words]))[0]

def get_response(ints, intents_json):
    """
    Retorna a resposta do bot com base na classe prevista.
    """
    ERROR_THRESHOLD = 0.25
    predicted_class = np.argmax(ints)

    if ints[predicted_class] > ERROR_THRESHOLD:
        response = random.choice(intents_json['intents'][predicted_class]['responses'])
    else:
        response = "Desculpe, não entendi. Pode reformular a pergunta?"

    return response
