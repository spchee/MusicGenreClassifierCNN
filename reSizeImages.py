import librosa
from PIL import Image
import os
from multiprocessing import Process
def genDataSet():
    dataSet = []
    for dirPath, dirNames, fileNames in os.walk("."):
            for dirName in dirNames: #Loops through every directory
                for file in os.listdir(dirName): #Loops through every single file within the current directory
                    #If the file ends with .wav then it will append the path of that file to the dataset
                    if file.endswith(".jpg"): 
                        dataSet.append(["./" + dirName + "/" + file, dirName])
    return dataSet

def splitData(dataSet, n): 
    #Finds the length of each individual dataset will be once it's split up
    length = int(len(dataSet)/n) 
    dataSets = []
    for i in range(n-1):
        dataSets += [dataSet[i*length : (i+1)*length]] 
    #We add on the last section of the dataset on seperately as else 
    #Certain tracks may be removed due to variable length being rounded 
    #down when converted to an integer
    dataSets.append(dataSet[(i+1)*length:])  
    return dataSets

'''This procedure will resize all the images within a given dataset and save them as png'''
def resizeImages(dataSet, width, height): #MAKE A BACKUP BEFORE YOU USE THIS AS IT'LL OVERWRITE EVERYTHING
    for image in genDataSet():
        try:
            img = Image.open(image[0]) #Opens the image
            img = img.resize((width,height), Image.ANTIALIAS) #Resizes the image
            img.save(image[0][:-3] + "png") #Saves the image
        except:
            print("failed")

os.chdir("./images")
dataSets = splitData(genDataSet(), 4) #Splits the data based on the number of cores the computer has.
cores = []
for i in range(4):
    cores.append(Process(target=resizeImages, args=(dataSets[i],120,90,))) #Makes a new process for each core containing different datasets
for core in cores: #Starts each individual process.
    core.start()
for core in cores:
    core.join()



