def mainModel(X_train, X_test, Y_train, Y_test):  
    model = Sequential()
    model.add(Conv2D(32, (4,4), input_shape = (90,120,3) ))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(64, (3,3), activation="relu", padding='same'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.4))
    model.add(Conv2D(128, (5,5), activation="relu", padding='same'))
    model.add(MaxPooling2D(pool_size = (2,2))) 
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model
    
    
    
    history = model.fit_generator(genImageArrays(X_train, Y_train, 30), epochs = 50, steps_per_epoch = 150,
                       verbose = 1, validation_data = genImageArrays(X_test, Y_test, 50), validation_steps=50, validation_freq=1)


    # Create count of the number of epochs
    print(history.history.keys())
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    testLoss, testAcc = model.evaluate_generator(genImageArrays(X_test, Y_test, 50), steps=100)
    predictions = model.predict_generator(genImageArrays(X_test, Y_test, 50), steps=100)
    print("test Accuracy:", testAcc)
    print("test Loss", testLoss)
    file = open("predictions.txt", "wb")
    pickle.dump(predictions, file)
    file.close()
    print("test Accuracy:", testAcc)
    print("test Loss", testLoss)
    file = open("trainModel.pkl", "wb")
    pickle.dump(model, file)
    
'''   Doesn't work
def plotConfusionMatrix(cMatrix):
    classes = ["EDM", "HipHop", "Metal", "Pop", "Rock" ]
    for i, j in itertools.product(range(cMatrix.shape[0]), range(cMatrix.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')'''