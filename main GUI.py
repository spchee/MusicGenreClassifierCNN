
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QDesktopWidget
import sys, os, shutil
import time
import zipfile
import numpy as np
import keras
from keras.models import Sequential,Input,Model, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import tensorflow as tf
from pydub import AudioSegment
from PIL import Image
import matplotlib
matplotlib.use('Agg') # No pictures displayed 
import pylab
import librosa
import librosa.display
import matplotlib.pyplot as plt
from math import floor
import taglib
import time

def splitIntoPixelSegments(trackPath):
    try:
        track = AudioSegment.from_file(trackPath, os.path.splitext(trackPath)[1][1:])
        trackSegments = []
        segmentLength = 10000
        pixelSegments = []
        
        for i in range(0,floor(len(track)/10000)):
            #Splits into 10 second segment and saves
            segment = track[i*10000:(i+1)*10000]
            segment.export("temp.wav", format = "wav")
            genMelSpectrogram("temp.wav", "temp.png")
            try:
                pixelSegments = np.vstack((pixelSegments, [imageToRGB("temp.png", True)]))
            except:
                pixelSegments = np.array([imageToRGB("temp.png", True)])
            try:
                os.remove("temp.wav")
                os.remove("temp.png")
            except:
                pass
        return pixelSegments
    
    except Exception as e:
        print(e)   
        
def genMelSpectrogram(inputFile, saveFile):
    print("gen")
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
    print("imagetoRGB")
    img = Image.open(inputFile) #Opens File
    img = img.resize((120,90), Image.ANTIALIAS) #Resizes the image
    pixels = (np.asarray(img)).astype("float32")[:,:,:3] #Converts it to a numpy array and discards alpha channel value
    if normalise: #This normalises it between the values of 0 and 1
        pixels= pixels/255.0
    img.close()
    return pixels

def editGenreTag(inputPath,outputPath, fileName, label, subFolders = True):

    if subFolders:
        if not os.path.exists(os.path.join(outputPath, label)): #If the directory doesn't exist for the genre, then make it.
            os.mkdir(os.path.join(outputPath, label))
        outputPath = os.path.join(outputPath, label, fileName) #If sorting into subfolders, set output path to include genre folder.

    else:
        outputPath = os.path.join(outputPath, fileName)  #Else dont bother with the genre folder.
        
    if (outputPath!= inputPath): 
        #Copies the initial file into it's output path if the input path isnt the same as the output path
        shutil.copy(inputPath, outputPath) 
            
    track = taglib.File(outputPath) 
    track.tags['GENRE'] = [label] #changes ID3 genre tag to the newly predicted genre.
    track.save() #save genre.
    
        

    
    
def predictGenre(pathToFile, fileName, outputPath, model, labels, sortIntoSubFolders):
    fullPath = os.path.join(pathToFile, fileName) #Full path to the file we are going to predict
    trackSegments = splitIntoPixelSegments(fullPath) #Converts it into 10 second melspectrogram RGB numpy arrays
    model = load_model(model) #Loads model
    predictions = np.mean(model.predict(trackSegments), axis = 0) #Predicts each segment and finds mean prediction.
    labelIndex = int(np.where(predictions == np.amax(predictions))[0][0]) #Finds the maximum value in the prediction
    
    #Edits the genre tag the track and moves it to it's designated location.
    editGenreTag(fullPath, outputPath, fileName, labels[labelIndex], sortIntoSubFolders) 
    return predictions


class Ui_MainWindow(object):
    
    def setupUi(self, MainWindow):
        

        global x
        global y
        
        #Main window
        MainWindow.setObjectName("MainWindow")
        MainWindow.setGeometry(x/4, y/4, x/2, y/2) #Size and geometry of the main window.
        
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        #Information icon button in the top left corner.
        self.informationIconBtn = QtWidgets.QPushButton(self.centralwidget)
        self.informationIconBtn.setGeometry(QtCore.QRect(x/2 - 10 - y/20, 10, y/20, y/20)) #Sets the position and geomtry
        self.informationIconBtn.setText("") #Sets text to blank
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("resources/informationIcon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off) #Sets the Icon image
        self.informationIconBtn.setIcon(icon)
        self.informationIconBtn.setIconSize(QtCore.QSize(y/25, y/25)) #Sets the size of the icon
        self.informationIconBtn.setObjectName("informationIconBtn")
        
        #Start button to open the password window
        self.startBtn = QtWidgets.QPushButton(self.centralwidget)
        self.startBtn.setGeometry(QtCore.QRect(x/8, 0.40 * y, x /4, y/16)) #Sets size and position of button
        font = QtGui.QFont()
        font.setPointSize(19)#Set font size
        self.startBtn.setFont(font)
        self.startBtn.setStyleSheet("background-color: green") #Sets colour of button
        self.startBtn.setObjectName("startBtn")
        
        #A container to contain all the input information to make it easier to manager.
        self.groupInput = QtWidgets.QGroupBox(self.centralwidget)
        self.groupInput.setGeometry(QtCore.QRect(x/14, y/21, 5 * x/14, y/9)) #Sets size and position of group.
        self.groupInput.setObjectName("groupInput")
        
        #Input folder button to open file explorer
        self.folderBtn = QtWidgets.QPushButton(self.groupInput)
        self.folderBtn.setGeometry(QtCore.QRect(31* x /100, y/26, x/35, y/30)) #Sets size and position
        self.folderBtn.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("resources/folderIcon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off) #Sets icon
        self.folderBtn.setIcon(icon1)
        self.folderBtn.setIconSize(QtCore.QSize(28, 28)) #Sets icon size
        self.folderBtn.setObjectName("folderBtn")
        
        #Input line for user to input text for folder location
        self.inpFolderLineEdit = QtWidgets.QLineEdit(self.groupInput)
        self.inpFolderLineEdit.setGeometry(QtCore.QRect(x/60, y/25, 9*x/31, y/33)) #Sets size and position
        self.inpFolderLineEdit.setObjectName("inpFolderLineEdit")
        self.inpFolderLineEdit.setPlaceholderText("Input Folder:") #Sets placeholder text
        
        
        #Check box to select option to search sub-directories
        self.searchSubChkBox = QtWidgets.QCheckBox(self.groupInput)
        self.searchSubChkBox.setGeometry(QtCore.QRect(x/35, y/13, 5 * x/27, 18)) #Sets size and position
        self.searchSubChkBox.setObjectName("searchSubChkBox")
        
        #A container to contain all the output information to make it easier to manager.
        self.groupOutput = QtWidgets.QGroupBox(self.centralwidget)
        self.groupOutput.setGeometry(QtCore.QRect(x/14, y/5, 5 * x/14, y/9)) #Sets size and position
        self.groupOutput.setObjectName("groupOutput")
        
        #Output folder button to open file explorer
        self.folderBtn_2 = QtWidgets.QPushButton(self.groupOutput)
        self.folderBtn_2.setGeometry(QtCore.QRect(31* x /100, y/26, x/35, y/30)) #Set size and position
        self.folderBtn_2.setText("")
        self.folderBtn_2.setIcon(icon1) #Sets icon
        self.folderBtn_2.setIconSize(QtCore.QSize(28, 28)) #Sets icon size
        self.folderBtn_2.setObjectName("folderBtn_2")
        
        #Output line for user to input text for output folder location
        self.outFolderLineEdit = QtWidgets.QLineEdit(self.groupOutput)
        self.outFolderLineEdit.setGeometry(QtCore.QRect(x/60, y/25, 9*x/31, y/33)) #Sets size and position
        self.outFolderLineEdit.setObjectName("outFolderLineEdit")
        self.outFolderLineEdit.setPlaceholderText("Output Folder:") #Sets placeholder Text
        
        #Check box for user to select whether they want tracks to be sorted into their identified genres.
        self.sortSubChkBox = QtWidgets.QCheckBox(self.groupOutput)
        self.sortSubChkBox.setGeometry(QtCore.QRect(x/35, y/13, 5 * x/27, 18)) #Sets size and position.
        self.sortSubChkBox.setObjectName("sortSubChkBox")
        
        
        MainWindow.setCentralWidget(self.centralwidget)

        #Sets up status bar
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        
        self.retranslateUi(MainWindow)
        
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        self.informationIconBtn.clicked.connect(self.showInformation) #Shows information menu when clicked
        
        self.startBtn.clicked.connect(self.openPasswordWindow) #Runs the openPasswordWindow function
        
        #Links the folder buttons with the showFileExplorer Function
        self.folderBtn.clicked.connect(lambda: self.showFileExplorer()) #Input folder button
        self.folderBtn_2.clicked.connect(lambda: self.showFileExplorer(False)) #Output Folder button
        
    #Opens file explorer and allows user to select directory
    def showFileExplorer(self, inputBox = True): 
        dir = QFileDialog()
        dirName = str(QFileDialog.getExistingDirectory(dir, "Select a directory", )) #Opens the file explorer
       
       #changes the text inside the input/output directory line edit
        if inputBox:
            self.inpFolderLineEdit.setText(str(dirName)) 
        elif not inputBox:
            self.outFolderLineEdit.setText(str(dirName))
             
        
    def showInformation(self): #Will display the information menu
        msg = QMessageBox() #Message box
        msg.setWindowTitle("Information") #Message Title
        msg.setText("This program will try to determine the genre of a song by just listening to the track! ") #Message text
        
        #Main message paragraph
        msg.setInformativeText("To use: \n 1. Type in the input folder destination containing the track(s). Or alternatively click the folder icon to open the folder exploerer to locate the folder. \n 2. Type in the output folder destination into hte scond box named 'Output' folder. Or click on the folder icon to open the folder explorer. If left blank it will be the same as the input folder. \n 3. Select each tickbox for different options. \n 4. Click the start button and enter the password. \n 5. Wait and after sometime it will be complete!")
        msg.setIcon(QMessageBox.Information) #Message icon
        
        x = msg.exec_()
        
    def openPasswordWindow(self): #Opens the password menu window
        inputPath = self.inpFolderLineEdit.text() #Gets the text from the input line
        outputPath = self.outFolderLineEdit.text() #Gets text from the output line
        searchSub = self.searchSubChkBox.isChecked() #Checks if the seach sub-folders chkbox is checked
        sortSub = self.sortSubChkBox.isChecked() #Checks if the sort into sub-folders chkbox is checked
        
        if not (os.path.isdir(inputPath) and os.path.isdir(outputPath)): #If the path doesn't exist
            msg = QMessageBox() #Message box
            msg.setWindowTitle("Error") #Message Title
            msg.setText("You have entered an invalid path, please enter a valid path") #Error Message text
        
        
            msg.setIcon(QMessageBox.Critical) #Message icon
        
            x = msg.exec_()
        
        else: #If it does exist, continue as normal
            MainWindow.hide() #Hides the main window
        
            #Opens the password menu and passes in parameters from previous.
            self.window = QtWidgets.QMainWindow() 
            self.ui = passwordMenu(self, inputPath = inputPath, outputPath = outputPath, searchSubFolders = searchSub, sortIntoFolders = sortSub)
            self.ui.setupUi(self.window)
       
            self.window.show() #Shows password menu.
        
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "genreIdentifier"))
        self.startBtn.setText(_translate("MainWindow", "Start"))
        self.groupInput.setTitle(_translate("MainWindow", ""))
        self.inpFolderLineEdit.setText(_translate("MainWindow", ""))
        self.searchSubChkBox.setText(_translate("MainWindow", "Search Sub-Folders"))
        self.groupOutput.setTitle(_translate("MainWindow", ""))
        self.outFolderLineEdit.setText(_translate("MainWindow", ""))
        self.sortSubChkBox.setText(_translate("MainWindow", "Sort into Sub-Folders by Genre"))

class progressMenu(object):
    def __init__(self,  parent, model,inputPath, outputPath, searchSubFolders, sortIntoFolders):
        self.inputPath = inputPath
        self.outputPath = outputPath
        self.searchSubFolders = searchSubFolders
        self.sortIntoFolders = sortIntoFolders
        self.parent = parent
        self.model = model
        
    
    def processTracks(self):
        fileCount = 0
        filesCompleted = 0
        QtCore.QCoreApplication.processEvents()
        
        #If searching all sub-folders
        if self.searchSubFolders:
            
            #Will loop through every single file within a folder including sub-folders.
            for root, dirs, files in os.walk(self.inputPath):
                for name in files:
                    #Checks if file name matches a music extension
                    if name.endswith((".flac", ".mp3", ".aac", ".wav", ".flv", ".ogg", ".raw")):
                        fileCount += 1 #To determine total number of tracks for the progressbar.
                        self.progressBar.setMaximum(fileCount) #Sets max to the number of tracks found.
                        
            QtCore.QCoreApplication.processEvents()                        
            #Will loop through every single file within a folder including sub-folders.          
            for root, dirs, files in os.walk(self.inputPath):
                for name in files:
                    QtCore.QCoreApplication.processEvents()#Will update the GUI, although it's glitchy
                    if name.endswith((".flac", ".mp3", ".aac", ".wav", ".flv", ".ogg", ".raw")): #Supported file extensions
                        predictGenre(root, name, self.outputPath,  self.model, ["EDM", "HipHop", "Metal", "Pop", "Rock"], self.sortIntoFolders) #Predicts the genre of a track
                        filesCompleted +=1 
                        self.progressBar.setValue(filesCompleted) #Adds 1 to completed file count
                        self.label.setText(str(filesCompleted) + "/" + str(fileCount))
                        print(filesCompleted)
                        
        #if not searching through folders            
        else:
            for file in os.listdir(self.inputPath): #loops through every file in a folder
                if file.endswith((".flac", ".mp3", ".aac", ".wav", ".flv", ".ogg", ".raw")):#Supported file extensions
                    fileCount += 1
                    self.progressBar.setMaximum(fileCount) #Sets progress bar max
                    
            for file in os.listdir(self.inputPath):#loops through every file in a folder
                if file.endswith((".flac", ".mp3", ".aac", ".wav", ".flv", ".ogg", ".raw")):#Supported file extensions
                    QtCore.QCoreApplication.processEvents()#updates gui
                    #Predicts the actual genre
                    predictGenre(self.inputPath, file, self.outputPath, self.model, ["EDM", "HipHop", "Metal", "Pop", "Rock"], self.sortIntoFolders)
                    filesCompleted +=1 
                    self.progressBar.setValue(filesCompleted) #Updates progress bar gui
                    self.label.setText(str(filesCompleted) + "/" + str(fileCount)) #Updates text gui
                    
        self.label.setText("Completed") #Sets text to completed when its done.
        
    def setupUi(self, MainWindow):
        global x,y
        
        #The actual progress menu
        MainWindow.setObjectName("MainWindow")
        MainWindow.setGeometry(x/4, 0.42*y, x/2, 0.095*y)#Sets the size/geometry of the actual window
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        #The progress bar which displays the progress of the algorithm
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(0.01 * x, 0.0225 * y, 0.48 * x, 0.04*y))#Sets size and geometry of progress bar 
        self.progressBar.setProperty("value", 0) #Sets the initial value
        self.progressBar.setTextDirection(QtWidgets.QProgressBar.TopToBottom)
        self.progressBar.setObjectName("progressBar")
        
        #The text which displays the number of tracks done
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0.01 * x, 0.00075*y, 300, 0.03 *y))#Sets geometry of text
        self.label.setObjectName("label")
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 707, 18))
        self.menubar.setObjectName("menubar")
        
        
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.retranslateUi(MainWindow)
        
        self.processTracks()
        

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Progress"))
        
class passwordMenu(object):
    def __init__(self,parent, inputPath = "", outputPath = "", searchSubFolders = False, sortIntoFolders = False):
        self.inputPath = inputPath
        self.outputPath = outputPath
        self.searchSubFolders = searchSubFolders
        self.sortIntoFolders = sortIntoFolders
        self.parent = parent
        
    def setupUi(self, MainWindow):
        global x,y
        
        #Main window
        MainWindow.setObjectName("MainWindow")
        MainWindow.setGeometry(x/4, 0.42*y, x/2, 0.095*y) #Sets size and geometry of main window
        
        
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        #Password input line
        self.passwordLineText = QtWidgets.QLineEdit(self.centralwidget)
        self.passwordLineText.setGeometry(QtCore.QRect(0.01 * x, 0.03 * y, 0.42 * x, 0.04*y))#Sets size and geomtetry
        self.passwordLineText.setObjectName("passwordLineText")
        
        #The text saying to "please enter password"
        self.enterPassText = QtWidgets.QLabel(self.centralwidget)
        self.enterPassText.setGeometry(QtCore.QRect(0.01 * x, 0.00075*y, 300, 0.03 *y)) #Sets geometry
        self.enterPassText.setObjectName("enterPassText")
        
        #Button to enter password
        self.enterPassBtn = QtWidgets.QPushButton(self.centralwidget)
        self.enterPassBtn.setGeometry(QtCore.QRect(0.455 * x, 0.0175 * y, 0.04 * x, 0.05*y)) #Sets size and geometry
        self.enterPassBtn.setObjectName("enterPassBtn")
        
        MainWindow.setCentralWidget(self.centralwidget)
        
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)


        
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        self.retranslateUi(MainWindow)
        
        #Links the enter button with the unencrypt model method.
        print(self.passwordLineText.text())
        self.enterPassBtn.clicked.connect(lambda: self.unencrptModel(password = self.passwordLineText.text()))
 
    def unencrptModel(self, password):
        try: 
            archive = zipfile.ZipFile('model.zip', 'r') #Opens the zipped file.
            
            #Tries to extract it with the inputted password
            print(password)
            model = archive.extract('model.h5', pwd = bytes(password, "utf-8"))
            
            #If successful, it will hide the window and open the progress menu
            self.parent.window.hide()
            self.openProgressMenu(model)
            os.remove("model.h5") #Deletes model file after compelte
            
            #If the password is incorrect, it will display an error message.
        except Exception as e:
            msg = QMessageBox()
            msg.setWindowTitle("Error") #Title
            msg.setText(str(e)) #Error message
            msg.setIcon(QMessageBox.Critical) #Error Icon
            x = msg.exec_()
            
    def openProgressMenu(self, model):
       
        
        self.window = QtWidgets.QMainWindow()
        self.window.show() #Opens progress menu
        self.ui = progressMenu(self, "model.h5", self.inputPath, self.outputPath, self.searchSubFolders, self.sortIntoFolders) #Passes in parameters
        self.ui.setupUi(self.window)
        MainWindow.hide() #Hides password menu
        
        

        
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.enterPassText.setText(_translate("MainWindow", "Please Enter Password:")) #Sets the text
        self.enterPassBtn.setText(_translate("MainWindow", "Enter"))      
        

  
if __name__ == "__main__":
    
    import sys
    app = QtWidgets.QApplication(sys.argv)
    screen = app.primaryScreen()
    
    global x, y
    x = screen.size().width()
    y = screen.size().height()    
    
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

