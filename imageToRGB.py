import numpy as np
from PIL import Image
import os
from multiprocessing import Queue, Process
import pickle
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score, roc_curve, auc
import random
from tqdm import tqdm
from matplotlib import pyplot as plt
import itertools
import tensorflow as tf
import pandas as pd


'''This function takes in an image as a parameter and then returns a RGB 2D array'''
def imageToRGB(inputFile, normalise = False):
    os.chdir("/home/spchee/CodeProjects/School Project/images")
    img = Image.open(inputFile) #Opens File
    pixels = (np.asarray(img)).astype("float32") #Converts it to a numpy array
    if normalise: #This normalises it between the values of 0 and 1
        pixels= pixels/255.0
    img.close()
    return pixels


def genDataSet():
    dataSet = []
    os.chdir("./images")
    for dirPath, dirNames, fileNames in os.walk("."):
            for dirName in dirNames: #Loops through every directory
                for file in os.listdir(dirName): #Loops through every single file within the current directory
                    #If the file ends with .wav then it will append the path of that file to the dataset
                    if file.endswith(".png"): 
                        dataSet.append(["./" + dirName + "/" + file,dirName])
    os.chdir("..")
    return dataSet
    
    

    
#Splits the dataset into training and testing data. The parameter n is the percentage split of testing and training.
def createXYDataSets(dataSet, n = 0.8): 
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    totalLength = len(dataSet) #Total length of the dataset 
    while len(X_train) < 0.8*totalLength: #Loops whilst the training data is lower than 80% of the total length, 
        i = random.randint(0,len(dataSet)-1) #Picks random track number
        X_train.append(dataSet[i][0]) #Appends the track to the end of the training data
        Y_train.append(getLabel(dataSet[i][1])) #Appends the result (the genre) to the end of training data
        del dataSet[i] #Deletes added track from original list

    while len(dataSet) != 0:
        image = random.randint(0,len(dataSet)-1)
        X_test.append(dataSet[image][0]) #Adds remaining data to testing data
        Y_test.append(getLabel(dataSet[image][1]))
        del dataSet[image]
    return X_train, X_test, Y_train, Y_test
    
def getLabel(genre): #Assigns integer label for each genre
    label = [0,0,0,0,0]
    dataLabels={
        "EDM":0,
        "HipHop":1,
        "Metal":2,
        "Pop":3,
        "Rock":4
        }
    label[dataLabels[genre]] = 1
    return label

'''
def genImageArrays(X, Y, batchSize):
    while True:
        #Will loop through every track in the testing and training dataset and add it to the 
        #data to be trained until it meets the batchsize requirements
        for i in range(len(X)): 
            
            #Adds the 2d image array to the training data list.
            try:
                X_dataset = np.vstack((X_dataset, [imageToRGB(X[i], True)]))
            except:
                #If trainingData is empty, it will create a new numpy array as using vstack will cause an error.
                X_dataset = np.array([imageToRGB(X[i], True)])
                
            #Adds the testing data to the testing data list.
            try:
                Y_dataset = np.vstack([Y_dataset [Y[i]]])
            except:
                #If testingData is empty, it will create a new numpy array as using vstack will cause an error.
                Y_dataset = np.array([Y[i]])
            
            #If it meets the batchSize requirements, yield and rest both testing and trainin data lists.
            if (i+1)%batchSize == 0:
                yield (X_dataset.astype(np.float32), Y_dataset.astype(np.float32))
                X_dataset = np.array([])
                Y_dataset = np.array([])
'''
            
class generator:
    def __init__(self, X, Y, batchSize):
        self.X = X
        self.Y = Y
        self.batchSize = batchSize
        self.index = 0
        self.actualLabels = np.array([])
        
    def __genActualLabels__(self, Y_batch):
        try:
            #Creates an actualLabels attribute for when we want to create our confusion matrix.
            self.actualLabels = np.vstack([self.actualLabels, Y_batch])
        except:
            #If it's empty, then it'll create a new np array as else it'll cause an error.
            self.actualLabels = np.array(Y_batch) 
            
    def genBatch(self): #Will return the next batch each time. 
        for image in self.X[self.index * self.batchSize : (self.index + 1) * self.batchSize]:
            try:
                X_batch = np.vstack((X_batch, [imageToRGB(image, True)]))
            except:
                X_batch = np.array([imageToRGB(image, True)])
        
        Y_batch = np.array(self.Y[self.index * self.batchSize : (self.index + 1) * self.batchSize])
        self.__genActualLabels__(Y_batch) #Appends Y_batch to the actualLabels.
        self.index+=1 #Appends index so that next time, then next batch will be returned.
        return (X_batch.astype(np.float32), Y_batch.astype(np.float32))
        
    def mainGen(self): #The main loop which will continuously yield each batch.
        while True:
            yield self.genBatch()
            if self.index > (len(self.X) - 1)/self.batchSize:
                self.index = 0
            
        
        
    


cores = []
RGBImages = [[],[]]

pixelImages = []



dataSet = genDataSet()
X_train, X_test, Y_train, Y_test = createXYDataSets(dataSet, 0.9)
print(X_train[0:5], X_test[0:5], Y_train[0:5], Y_test[:5])

model = Sequential() #Instantiates initial model
model.add(Conv2D(32, (4,4), input_shape = (90,120,3) )) #Convolutional layer
model.add(MaxPooling2D(pool_size = (2,2))) #Max pooling layer
model.add(Dropout(0.3)) #Dropout layer to remove 30% of all neurons
model.add(Conv2D(64, (3,3), activation="relu", padding='same')) #Convolutional layer 2
model.add(MaxPooling2D(pool_size = (2,2))) #Max pooling layer 2
model.add(Dropout(0.4)) #Drop out layer 2
model.add(Conv2D(128, (5,5), activation="relu", padding='same')) #Convolutional layer 3
model.add(MaxPooling2D(pool_size = (2,2)))  #Max pooling layer 3
model.add(Dropout(0.4)) #Drop out layer 3
model.add(Flatten()) #Flattens it into a single dimension
model.add(Dense(256, activation="relu")) #Fully connected dense layer
model.add(Dense(5, activation='softmax')) #Final connected dense output layer. Outputs the final prediction
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]) #Compiles the model

trainGenerator = generator(X_train, Y_train, 30)
validationGenerator = generator(X_train, Y_train, 50)
#Will train the model and save its progress to 'history'
history = model.fit_generator(trainGenerator.mainGen(), epochs = 50, steps_per_epoch = 30,
                       verbose = 1, validation_data = validationGenerator.mainGen(), validation_steps=50, validation_freq=5, use_multiprocessing=True)
model.save("/home/spchee/CodeProjects/School Project/genreModel.h5") #Will save the model to a file.



evalGenerator = generator(X_test, Y_test, batchSize=50)
classes = ["EDM", "HipHop","Metal", "Pop", "Rock"] 
testLoss, testAcc = model.evaluate_generator(evalGenerator.mainGen(), steps=(len(X_test)/50 - 1)) #Evaluates the model
print(testAcc)

predGenerator = generator(X_test, Y_test, batchSize=50)
predictions = model.predict_generator(predGenerator.mainGen(), steps=50)

try:
    print(tf.math.confusion_matrix(labels=tf.argmax(predGenerator.actualLabels, 1), predictions=tf.argmax(predictions, 1))) #Will predoce a confusion matrix

except: 
    print("unable to print confusion matrix")



#Plots testing and training accuracy
try:
    #testing Accuracy
    #Different versions of keras/TF use either "acc" or "accuracy"
    plt.plot(history.history['acc']) 
except:
    plt.plot(history.history['accuracy'])
     
try:
    #Training accuracy
    plt.plot(history.history['val_acc'])
except:
    plt.plot(history.history['val_accuracy'])

plt.title('model accuracy') #Graph title
plt.ylabel('accuracy') #Y-axis label
plt.xlabel('epoch') #X-Axis label
plt.legend(['train', 'test'], loc='upper left') #Actual graph lines
plt.show()


#Plots testing and training loss
plt.plot(history.history['loss']) #training loss
plt.plot(history.history['val_loss']) #Testing loss
plt.title('model loss') #Graph Title
plt.ylabel('loss') #Y-axis label
plt.xlabel('epoch') #X-Axis Label
plt.legend(['train', 'test'], loc='upper left') #The actual graph lines
plt.show()


'''
os.chdir("/home/spchee/CodeProjects/School Project")

class BatchGenerator(keras.utils.Sequence):
    def __init__(self, otherArgs):
      code
    def __len__(self):
        return length

    def __getitem__(self, index):
        return batch
    def on_epoch_end(self):
        do_something'''

'''def splitData(dataSet, n): 
    #Finds the length of each individual dataset will be once it's split up
    length = int(len(dataSet)/n) 
    
    dataSets = []
    for i in range(n-1):
        dataSets += [dataSet[i*length : (i+1)*length]] 
    #We add on the last section of the dataset on seperately as else 
    #Certain tracks may be removed due to variable length being rounded 
    #down when converted to an integer
    dataSets.append(dataSet[(i+1)*length:])  
    return dataSets'''

'''def processImages(dataSet):
    for image in dataSet:
        
        RGBImages[0].append(imageToRGB(image[0]))
        RGBImages[1].append(image[1])
    return RGBImages
'''
'''def worker():
    i = 0
    while True:
        i+=1
        item = q.get() #Gets the first item from the queue
        if item == "FINISH": #It'll stop working once it encouters this item
            break
        else:
            #When it'll process each image and then add it on to the queue
            q.put([imageToRGB(item[0], True),item[1]])
            print(i)'''

'''
q = Queue()

for image in genDataSet()[0:1000]:
    q.put(image)
    q.put("FINISH")
worker()

for i in range(os.cpu_count()):
    
    cores.append(Process(target=worker)) #Makes a new process for each core containing different datasets
    q.put("FINISH")
for core in cores: #Starts each individual process.
    core.start()
for core in cores: #Starts each individual process.
    core.join() 
pixelImages = []
while q.qsize() != 0:
    pixelImages.append(shared_queue.get())
'''





