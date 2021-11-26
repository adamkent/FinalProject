import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

path_data = "C:/Users/Adam/Desktop/Project/data_10.json"

def getdeta(path_data):
  
    with open(path_data, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"]) # list to array
    y = np.array(data["code"]) # list to array
    print("Loaded dataset")
    return  X, y

def overfittingModel(X_input, X_test, y_input, y_test):

    mlp = keras.Sequential([

        keras.layers.Flatten(input_shape=(X.shape[1], X.shape[2])), # input

        keras.layers.Dense(512, activation='relu'),        # hidden layer 1

        keras.layers.Dense(128, activation='relu'),        # hidden layer 2

        keras.layers.Dense(64, activation='relu'),        # hidden layer 3

        keras.layers.Dense(10, activation='softmax')        # output
    ])
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    mlp.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
                  
    mlp.summary()
    training = mlp.fit(X_input, y_input, validation_data=(X_test, y_test), batch_size=32, epochs=50)

    ig, axs = plt.subplots(2)

    axs[0].plot(training.history["accuracy"], label="Training") # plot accuracy
    axs[0].plot(training.history["val_accuracy"], label="Testing")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xlabel("Epoch Number")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy accross 50 epochs")

    axs[1].plot(training.history["loss"], label="Training") # plot loss
    axs[1].plot(training.history["val_loss"], label="Testing")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epoch Number")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss accross 50 epochs")
    plt.show()

def fixedModel(X_input, X_test, y_input, y_test):

    mlp = keras.Sequential([

        keras.layers.Flatten(input_shape=(X.shape[1], X.shape[2])), # input

        keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),    # hidden layer 1
        keras.layers.Dropout(0.5),

        keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),    # hidden layer 2
        keras.layers.Dropout(0.2),

        keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)), # hidden layer 3
        keras.layers.Dropout(0.2),

        keras.layers.Dense(10, activation='softmax')    # output

    ])
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    mlp.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    mlp.summary()
    training = mlp.fit(X_input, y_input, validation_data=(X_test, y_test), batch_size=32, epochs=300)
    
    fig, axs = plt.subplots(2)

    axs[0].plot(training.history["accuracy"], label="train accuracy") # plot accuracy
    axs[0].plot(training.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    axs[1].plot(training.history["loss"], label="train loss") # plot loss
    axs[1].plot(training.history["val_loss"], label="test loss")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss eval")

    plt.show()
    
    predictX = X_test[50]
    predicty = y_test[50]
    predictX = predictX[np.newaxis, ...] # extra dimension for input data
    guess = mlp.predict(predictX) # attempt prediction
    # get index with max value
    maxvalueindex = np.argmax(guess, axis=1) # select highest value
    print("Target: " + str(predicty) + "Predicted genre code: " + str(maxvalueindex))
        

if __name__ == "__main__":

    X, y = getdeta(path_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    
    #overfittingModel(X_train, X_test, y_train, y_test)
    fixedModel(X_train, X_test, y_train, y_test)

