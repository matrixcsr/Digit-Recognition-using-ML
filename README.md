# Digit-Recognition-using-ML
An Image to text converter program that leverages ML and predicts digits drawn on Tkinter GUI and image that are uploaded through it.

Works best at 1920x1080.
Requires Python 3.7+ to run.
Requires necessary dependencies.

**pip or pip3 according to your pip version (pip --version)**

- pip install pyautogui             #Pyautogui for capturing frame and taking screenshot.
- pip install --upgrade tensorflow  #To load and run the tf model.
- pip install opencv-python         #For pre-processing of Image.
- pip install numpy                 #For array functions.
- pip install matplotlib            #For showing a graph on accuracy of predictions.
- pip install pandas                #Pandas required to make a data frame for graph.

**--OPTIONAL--**  
~~- pip install pickle~~                #Needed to load SKlearn model.(deprecated model)
- pip install silence_tensorflow    #To suppress Tensorflow debug code during compile.

GUI is based on Python Tkinter libraries, GUI has two modules 'Draw' and 'Upload'.

**1st Module:-**
In 'Draw' module, user will draw using the cursor inside the black canvas and the program will predict the digit when the 'Predict' button is pressed.

**2nd Module:-**
In 'Upload' module, user will upload an image of a digit(single) and the program will predict, If the prediction is wrong, User will be provided an alt method which changes some of the image pre-processing and will generate a different and more accurate output.
