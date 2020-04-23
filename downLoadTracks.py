####################################
import pandas as pd                 
import numpy as np                  
import os                           
import random                       
import youtube_dl                   
import pydub                          
import multiprocessing    
from multiprocessing import Process, Queue 
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')
import pylab
import librosa
import librosa.display
from PIL import Image
import os
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
import itertools
###################################


#This class is a class used for importing a dataSet from youtube and downloading it.
class youtubeDataSet:  
    
    def viewDataSet(self):
        try:
            for key, value in self.dict_filteredDataSet.items(): #Will print the dataset and how many elements are in each label
                print(key + ":", len(value)) #Prints length of each label
                print(key + ":", value[0:5]) #Prints the first 5 elements within each label

        except:
            try:
                print("length:", len(self.list_unfilteredDataSet)) #Prints the length of the dataset.
                print(self.list_unfilteredDataSet[0:5]) #Will print first 5 elements of the dataset.
            except:
                print("No dataset exists use 'importData' to create an initial dataset.")
    
    def importData(self, str_dataSet):

        #Will import the dataset as a list of each lines from the csv file
        tracks = open(str_dataSet, 'r').readlines()
        
        #split each string of lines into individual arrays for individual elements of the current dataset.
        for i in range(len(tracks)):
            tracks[i] = tracks[i].split(',')
        self.list_unfilteredDataSet = tracks[3:]
     
     
        
    def filterDataSet(self, dict_filters):
        if not hasattr(self, 'list_unfilteredDataSet'): #Will check whether the current dataset object contains a filtered data set.
            print("Please first use 'importData'")
            return
        self.dict_filters = dict_filters

        #Creates dictionary for containing all the tracks of each genre
        self.dict_filteredDataSet = {} 
        for filter in self.dict_filters:
            self.dict_filteredDataSet[filter] = []

        #Loop through every element in our dataset
        for track in self.list_unfilteredDataSet:
            flag = False
            #Loop through every genre in our list of genres.
            for label in self.dict_filters.keys():
                #Now loop though every subLabel of the current label being looped through
                for subLabel in dict_filters[label]:
                    if subLabel in track: #If the subLabel is found in the current track
                        #We only need to append the first 3 indexes as we only now need the timestamps and youtube url.
                        self.dict_filteredDataSet[label].append(track[0:3]) 
                        flag = True #Set flag to true to indicate we've already appended the data element to the list.
                        break
        
                if flag:
                    break       
                
    #This will remove tracks from selected labels until they are under a certain amount.
    def balanceDataSet(self, list_labels, int_maxElements): 
        for label in list_labels: 
            length = len(self.dict_filteredDataSet[label]) #Initial length of label's list
            while (length)>int_maxElements: #Whilst current length > wanted length, delete random items in list.
                del self.dict_filteredDataSet[label][random.randint(0,length-1)] #Deletes a random item
                length -=1
                
    def KeyValueGen(self, data): #Will create pairs of the key and value.
        # will yield ('A', 1), ('A', 2), .... ('C', 8)
        for key, values in data.items():
            for value in values:
                yield key, value #Returns key and value
                
    def SplitDictionary(self, n):
        dataSets = []
        totalLength = 0;
        for lists in self.dict_filteredDataSet.values(): #Finds the total length of the lists within the dictionary
            totalLength += len(lists)
            
        for index, (key, value) in enumerate(KeyValueGen(self.dict_filteredDataSet.values)): 
            if index%int(totalLength/n) == 0 and len(dataSets) <n: #Will create a new dictionary when the current one exceeds the intended length
                dataSets.append({})

            dataSets[-1].setdefault(key, []).append(value) #Appends the value to the current dictionary
        return dataSets
                 
            
    def downloadTracks(self, directory, multiCore = True, dataSet= None): #"/home/spchee/CodeProjects/School Project/"
        if dataSet == None:
            if not hasattr(self, 'dict_filteredDataSet'): #Will check whether the current dataset object contains a filtered data set.
                print("Please first use 'importData' and create a filteredDataSet using 'filterDataSet'")
                return
            else:
                dataSet = self.dict_filteredDataSet
                
        if multiCore:
            coreCount = multiprocessing.cpu_count()
            dataSets = SplitDictionary(coreCount)
            
            cores = []
            for i in range(coreCount):
                cores.append(Process(target=genreDataSet.downloadTracks, args=(directory, False, dataSets[i],)))#Makes a new process for each core containing different sub-dataset
            
            for core in cores:
                core.start()
            for core in cores:
                core.join()
            
        else:    
            for label in dataSet: 
                os.chdir(directory)#Resets the directory.
                ydl_opts = {
                    'format': 'worstaudio/worst',
                    'quiet': True,
                    'outtmpl': 'temp.%(ext)s',
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'wav',
                        'preferredquality': '192',
                        }],
                    }
                try:
                    os.mkdir(label) #if the folder of the label doesn't exist then it will create a new folder for that label.
                except:
                    pass
            
                os.chdir(label)#Changes the directory to the current label's directory.
                i = 1
                
                for track in dataSet[label]: #Loops through every track from the label in the dataset
                    try:
                        youtube_dl.YoutubeDL(ydl_opts).download(['https://www.youtube.com/embed/' + track[0]]);
                        song = pydub.AudioSegment.from_wav("temp.wav") #Splits the track into into sections
                        song = song[int(float(track[1]))*1000:int(float(track[2])*1000)] #Splits it into the inteded 10 second segment
                        
                        #If two different processes are downloading tracks from the same label, this will ensure they don't overwrite any other tracks.
                        while os.path.isfile(label + str(i) + ".wav"): 
                            i+=1 #i is the song number in that genre.
                            
                        try:
                            print(i)
                            song.export(label+ str(i) + ".wav", format = "wav") #Exports it to <genrename><number>.wav 
                            os.remove("temp.wav")
                            i+=1 
                        except Exception as e:
                            i+=1
                            print(e)
                    
                    except Exception as e:
                        print(e)
                        #Will print error if an error occured and the song couldn't be downloaded e.g. the video was made private or no longer exists
                        print("Failed To download")
               

        
        
        
            
            
            
'''____________________Downloading Tracks for Dataset_______________________'''


'''
We need to define the labels which the algorithm will search for. Some genres are closely related 
to the genre's we want to identify so they're also included.
'''

genreLabels = {
    "HipHop":["/m/06bxc","/m/0glt670"], 
    "Metal":["/m/03lty", "/m/05r6t"], 
    "Rock":["/m/06by7","/m/0dls3","/m/0dl5d","/m/07sbbz2","/m/05w3f"], 
    "EDM":["/m/02lkt","/m/03mb9","/m/07gxw","/m/07s72n","/m/0m0jc","/m/08cyft"], 
    "Pop":["/m/026z9","/m/064t9"]
}

'''
Genre Labels used:
Hip-Hop:
    Rapping - /m/06bxc
    Hip-Hop - /m/0glt670

Metal:
    Heavy Metal - /m/03lty
    Punk Rock - /m/05r6t

Rock:
    Rock Music - /m/06by7
    Grunge - /m/0dls3
    Progressive Rock - /m/0dl5d
    Rock and Roll - /m/07sbbz2
    Psychedelic Rock - /m/05w3f

EDM:
    Electronic music -/m/02lkt
    House music - /m/03mb9
    Techno - /m/07gxw
    Dubstep - /m/07s72n
    Electronica - /m/0m0jc
    Electronic dance music - /m/08cyft

Pop:
    Pop Music - /m/064t9
    Disco - /m/026z9
'''

#Create a new class for our CNN
genreDataSet = youtubeDataSet()
dataSet = 'unbalanced_train_segments.csv'
genreDataSet.importData(dataSet) #Import our initial data set
genreDataSet.filterDataSet(genreLabels) #Extracts tracks which only contain the labels we want.
genreDataSet.balanceDataSet(["Rock", "EDM"], 3500) #Balances the data set so that it is less skewed.
genreDataSet.downloadTracks("/home/spchee/CodeProjects") #Downloads each individual track and will sort into folder based on genres.





 
