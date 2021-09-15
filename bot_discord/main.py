from json import load
from numpy import array
from random import choice
from discord import Client
from nltk import word_tokenize
from modelo import create_model
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
from procesamiento import preprocess_data, create_training_testing_data


def get_sentence_words(sentence, lemmatizer):
    # Tokenizamos y lemmatizamos las palabras de
    # la oración.
    return [lemmatizer.lemmatize(
        word.lower()) for word in word_tokenize(sentence)]


def get_bag_of_words(sentence_words, words):
    bag = [0] * len(words)

    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1

    return array(bag)


def predict_class(model, classes, bag_of_words):
    ERROR_THRESHOLD = 0.25

    res = model.predict(array([bag_of_words]))[0]
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []

    for r in results:
        print("Intent: ", classes[r[0]])
        print("Probabilidad: ", r[1])
        if r[1] > .9:
            return_list.append(
                {"intent": classes[r[0]], "probability": str(r[1])})
        else:
            return_list.append(
                {"intent": "desconocido", "probability": str(r[1])})

    return return_list


def get_response(prediction, data):
    tag = prediction[0]['intent']
    list_of_intents = data['contenido']

    for i in list_of_intents:
        if(i['tag'] == tag):
            result = choice(i['respuestas'])
            break

    return result


def main():
    DISCORD_TOKEN = "ODg1NjI5MjY1Mjg1MDk5NTgw.YTp0hw.JpJt1SypUNNPAGCNzoZi59jNLec"

    lemmatizer = WordNetLemmatizer()

    ignore_words = ['¿', '?', '¡', '!']

    with open("contenido copy.json", encoding="utf-8") as file:
        data = load(file)

    # Paso 1 - Preprocesar los datos
    words, classes, documents = preprocess_data(
        data, lemmatizer, ignore_words)

    # Paso 2 (A) - Crear datos de entrenamiento y prueba
    train_x, train_y = create_training_testing_data(
        classes, documents, lemmatizer, words)

    # Paso 3 (A) - Crear el modelo
    model = create_model(train_x, train_y)
    # Paso 2 (B) - En caso de tener listo el modelo
    #model = load_model('chatbotprueba_model.h5')

    user_input = ''

    # INICIALIZACION DEL BOT
    bot_client = Client()

    @bot_client.event
    async def on_ready():
        print("Bot escuchando mensajes...")

    # @bot_client.event
    # async def on_typing(channel, user, when):
    #     greeting = f"Hola {user.mention} en qué te puedo ayudar?"

    #     await channel.send(greeting)

    @bot_client.event
    async def on_message(user_input):
        if user_input.author == bot_client.user:
            return

        sentence_words = get_sentence_words(user_input.content, lemmatizer)
        bag_of_words = get_bag_of_words(sentence_words, words)
        prediction = predict_class(model, classes, bag_of_words)
        response = get_response(prediction, data)

        await user_input.channel.send(response)

    bot_client.run(DISCORD_TOKEN)


if __name__ == '__main__':
    main()
