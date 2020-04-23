import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from multiprocessing import Process


'''This takes the track and generates a melSpectrogram out of it'''
def genMelSpectrogram(inputFile, saveFile):
    #Taken and modified from https://stackoverflow.com/questions/46031397/using-librosa-to-plot-a-mel-spectrogram
    try:
        sig, fs = librosa.load(inputFile)   
        plt.axis('off') # Removes axis from the melspectrogram
        plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edges
        S = librosa.feature.melspectrogram(y=sig, sr=fs, power = 2, n_mels = 120 ) 
        plt.display.specshow(librosa.power_to_db(S, ref=np.max))
        plt.savefig(saveFile, bbox_inches=None, pad_inches=0) #Save the melSpectrogram to the location 'saveFile'
        plt.close()
    except:
        pass
    
'''Loops through every track within the dataSet and creates a melspectrogram'''
def conversion(dataSet):
    
    for track in dataSet:
        genMelSpectrogram(track[0], "./images" + track[0][1:-3] + "jpg")
    
    
'''This function will generate a 2D array containing the paths to each track and their genre'''
def genDataSet():
    dataSet = []
    for dirPath, dirNames, fileNames in os.walk("."):
            for dirName in dirNames: #Loops through every directory
                for file in os.listdir(dirName): #Loops through every single file within the current directory
                    #If the file ends with .wav then it will append the path of that file to the dataset
                    if file.endswith(".wav"): 
                        dataSet.append(["./" + dirName + "/" + file, dirName])
    return dataSet


'''This splits the dataset up into n equal parts'''
def splitData(dataSet, n): 
    #Finds the length of each individual dataset will be once it's split up
    length = int(len(dataSet)/n) 
    
    dataSets = []
    for i in range(n-1):
        dataSets += [dataSet[i*length : (i+1)*length]]
        
    '''We add on the last section of the dataset on seperately as else 
    Certain tracks may be removed due to variable length being rounded 
    down when converted to an integer'''
    dataSets.append(dataSet[(i+1)*length:])  
    return dataSets


   
#Splits data into X number of datasets based on number of CPU cores.
dataSets = splitData(genDataSet(), os.cpu_count()) 

cores = []
for i in range(os.cpu_count()): #Makes a new process for each core containing different datasets
    cores.append(Process(target=conversion, args=(dataSets[i],))) 
for core in cores: #Starts each individual process.
    core.start()
for core in cores:
    core.join()
