from numpy import array
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD


def create_model(train_patterns, train_intents):
    """
    Función encargada de la creación del modelo de 3 capas
    Capa 1: 128 neuronas
    Capa 2:  64 neuronas.
    Capa 3:   n neuronas. Donde n es el tamaño del arreglo de intents.
    """
    model = Sequential()
    model.add(Dense(128, input_shape=(
        len(train_patterns[0]),), activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(len(train_intents[0]), activation='softmax'))

    # Compilación del modelo con Stochastic gradient descent (SGD)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])

    hist = model.fit(array(train_patterns), array(train_intents),
                     epochs=200, batch_size=5, verbose=1)
    model.save('chatbotprueba_model.h5', hist)

    return model
