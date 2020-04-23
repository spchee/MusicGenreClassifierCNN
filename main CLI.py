import numpy as np
import os, sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from keras.models import Sequential,Input,Model, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from pydub import AudioSegment
from PIL import Image
import os
import matplotlib
matplotlib.use('Agg') # No pictures displayed 
import pylab
import librosa
import librosa.display
import matplotlib.pyplot as plt
from math import floor

import argparse
 
def splitIntoPixelSegments(trackPath):
    try:
        track = AudioSegment.from_file(trackPath, os.path.splitext(trackPath)[1][1:]) #Loads the track
        
        trackSegments = []
        segmentLength = 10000 #10000ms = 10 seconds
        pixelSegments = []
        
        for i in range(0,floor(len(track)/10000)): #For every 10 second segment
            #Splits into 10 second segment and saves
            segment = track[i*10000:(i+1)*10000] 
            #Saves 10 second segment temporarily.
            segment.export("temp.wav", format = "wav")
            #Generates mel-spectrogram temporarily.
            genMelSpectrogram("temp.wav", "temp.png")
            
            try: #Converts mel-melspectrogram into RGB numpy array. 
                pixelSegments = np.vstack((pixelSegments, [imageToRGB("temp.png", True)])) #Adds the RGB numpy arr to pixelSegments.
            except:
                #If pixelSegments is empty, it'll produce an error when using vstack so we need to create a new one.
                pixelSegments = np.array([imageToRGB("temp.png", True)])
                
            try: #Removes the temp files.
                os.remove("temp.wav")
                os.remove("temp.png")
            except:
                pass
        return pixelSegments
    
    except Exception as e:
        print(e)
        
        
def genMelSpectrogram(inputFile, saveFile):
    #Taken and modified from https://stackoverflow.com/questions/46031397/using-librosa-to-plot-a-mel-spectrogram
    try:
        sig, fs = librosa.load(inputFile)   
        plt.axis("off")
        plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edges
        S = librosa.feature.melspectrogram(y=sig, sr=fs, power = 2, n_mels = 120 ) 
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
        plt.savefig(fname = saveFile, bbox_inches=None, pad_inches=0) #Save the melSpectrogram to the location 'saveFile'
        plt.close()
    except Exception as e:
        print(e)
def imageToRGB(inputFile, normalise = False):
    img = Image.open(inputFile) #Opens File
    img = img.resize((120,90), Image.ANTIALIAS) #Resizes the image
    pixels = (np.asarray(img)).astype("float32")[:,:,:3] #Converts it to a numpy array and discards alpha channel value
    if normalise: #This normalises it between the values of 0 and 1
        pixels= pixels/255.0
    img.close()
    return pixels

def predictGenre(trackPath, modelPath):
    
    trackSegments = splitIntoPixelSegments(trackPath) #Generates multiple numpy arrays to predict.
    
    model = load_model(modelPath) #Loads the model
    predictions = model.predict(trackSegments) #Predicts for each numpy array.
    meanPrediction = np.mean(predictions, axis = 0) #finds the mean prediction
    return meanPrediction
    
    
myParser = argparse.ArgumentParser()
myParser.add_argument('Path',
                       metavar='path',
                       type=str,
                       help='the path to the song file')
args = myParser.parse_args()

if not os.path.isfile(args.Path):
    print('The file specified does not exist')
    sys.exit()

predictions = predictGenre(args.Path, "model.h5")

genres = ["EDM", "HipHop", "Metal", "Pop", "Rock"]
for i in range(len(predictions)):
    print(genres[i] + ":", round(predictions[i], 3))
    




