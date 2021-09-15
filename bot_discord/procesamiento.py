from nltk import word_tokenize
from pickle import dump
from random import shuffle
from numpy import array


def preprocess_data(data, lemmatizer, ignored_words):
    """
    @param data: obtenido del archivo json
    @param lemmatizer: objeto necesario para 'lematizar' palabras
    @param iignored_words: caracteres a evitar para el análisis

    Se retorna una tupla con lo siguiente:
    @return words: arreglo de palabras lemmatizadas.
    @return classes: arreglo de tags únicos.
    @return documents: arreglo de tuplas con las palabras y
                       el tag correspondiente.
    """
    words = []
    classes = []
    documents = []

    for intent in data['contenido']:
        # tkoenizar cada palabra
        for pattern in intent['patrones']:
            w = word_tokenize(pattern)
            words += w  # Agregar el contenido del arreglo
            documents.append((w, intent['tag']))

            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    with open('words.pkl', 'wb') as archivo_pickle:
        dump(words, archivo_pickle)

    with open('classes.pkl', 'wb') as archivo_classes:
        dump(classes, archivo_classes)

    words = [lemmatizer.lemmatize(w.lower())
             for w in words if w not in ignored_words]

    classes = sorted(list(set(classes)))  # Lista ordenada de clases únicas.
    words = sorted(list(set(words)))      # Lista ordenada de palabras únicas.

    return words, classes, documents


def create_training_testing_data(classes, documents, lemmatizer, words):
    """
    @returns: Tupla con los arreglos de entrenamiento.
    """
    training = []
    output_row_template = [0] * len(classes)   # arreglo de 0's

    for doc in documents:
        bag = []
        # Lemmatización de cada palabra
        pattern_words = [lemmatizer.lemmatize(
            word.lower()) for word in doc[0]]

        for w in words:
            bag.append(1 if w in pattern_words else 0)
            output_row = output_row_template[:]
            output_row[classes.index(doc[1])] = 1
            training.append([bag, output_row])

    shuffle(training)
    training = array(training)

    train_patterns = list(training[:, 0])
    train_intents = list(training[:, 1])

    return train_patterns, train_intents
