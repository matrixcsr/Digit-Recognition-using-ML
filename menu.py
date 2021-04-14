''' 
IMAGE TO TEXT USING MACHINE LEARNING.
'''
##### BUNCH OF IMPORTS #####
from tkinter import *
from tkinter import filedialog, messagebox
import pyautogui
import pygetwindow
from silence_tensorflow import silence_tensorflow
silence_tensorflow()        #Disables unnecessary Warnings and Info messages from Tensorflow
import tensorflow as tf
from tensorflow import keras
import cv2 as cv
import numpy as np
from tkinter.tix import *
from PIL import Image, ImageTk
import pickle
import os.path
from os import path
from tkinter import ttk
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pandas import DataFrame
import webbrowser
#chrome = r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe %s'
##### GLOBAL VARIABLES #####
m = None
images_reference_list = []
g = []

##### GLOBAL PATHS #####
paths = {
    'logo': './Assets/logo.png',
    'imgpath': './Images/temp.png',
    'modelpath': './Models/newmodel'
}
##### ABOUT TAB DETAILS ###
about = {
    'title': 'Image to Text using ML',
    'desc': '''       An Image to Text converter program that leverages Machine Learning and predicts the digit drawn or uploaded image into machine readable format.''',
    'link': 'http://bit.ly/ITTUml',
    'cr': '''
Created by:

Rahul Soni    - B72
Aryan Soni    - B70
Moin Tirmizi  - B77'''
}

##### CHECKS FOR 'images' FOLDER IN PWD #####
if(path.exists('images') == True):
    print("directory exists:" + str(path.exists('images')))
else:
    os.mkdir('./images')

##### LOADING MODEL FOR 'UPLOAD TAB' ####
try:
    clf = pickle.load(open('./Models/saved.sav', 'rb'))

except FileNotFoundError:
    print('Could not load the pickle model.')

##### LOADING MODEL FOR 'DRAW' TAB #####
try:
    seq = keras.models.load_model(
        paths['modelpath'])

except OSError:
    print('Could not load the TF model')

##### IS CALLED DURING MOUSE PRESS EVENT #####


def paint(event):
    '''CAPTURES MOUSE DRAG EVENT AND DRAWS GRAPHICS IN CANVAS 'c1'. '''
    b2.config(state=NORMAL)
    color = 'white'
    # Tweak int value to increase/reduce brush size.
    x1, y1 = (event.x-10), (event.y-10)
    x2, y2 = (event.x+10), (event.y+10)
    c.create_oval(x1, y1, x2, y2, fill=color, outline=color)

##### IS CALLED WHEN 'Clear' BUTTON IS CLICKED in FRAME 'tab1'. #####


def clear():
    '''CLEARS THE CANVAS WHEN 'Clear' BUTTON IS CLICKED IN FRAME 'f1'. '''
    c.delete('all')
    b3.config(state=DISABLED)

##### IS CALLED WHEN 'Predict' BUTTON IS CLICKED IN FRAME 'tab1' #####


def predict():
    '''PREDICTS THE GRAPHICS DRAWN IN CANVAS 'c1'. '''
    global g
    # HELPS GET WINDOW COORDS.
    window = pygetwindow.getWindowsWithTitle('Image to Text')[0]
    # print(window)
    x1 = window.left + 15  # +10    adjusted +10  perfected +10
    y1 = window.top + 110  # +40    adjusted +40  perfected +75
    height = window.height - 160  # -50    adjusted -130
    width = window.width - 30  # -50    adjusted -20 perfected -20

    x2 = x1 + width
    y2 = y1 + height

    # TAKES A PRECISE SCREENSHOT AND SAVES IN MENTIONED PATH.
    pyautogui.screenshot(paths['imgpath'])

    im = Image.open(paths['imgpath'])
    im = im.crop((x1, y1, x2, y2))
    # ACCESS THE IMAGE AND PERFORMS CROP FUNCTION WITH THE PROVIDED COORDS.
    im.save(paths['imgpath'])
    # im.show(path)

    img = cv.imread(paths['imgpath'])  # ACCESS THE CROPPED IMAGE.
    # REMOVE COLOR CHANNELS FROM IMAGE.
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # RESIZE THE IMAGE TO 28X28.
    resized = cv.resize(gray, (28, 28), interpolation=cv.INTER_AREA)
    # DECREASES THE PIXEL VALUES FROM 0 - 255 TO 0 - 1 (PIXEL_VAL/255).
    newimg = tf.keras.utils.normalize(resized, axis=1)
    # RESIZES THE ARRAY TO FIT MODEL INPUT LAYER.
    newimg = np.array(newimg).reshape(-1, 28, 28, 1)
    b3.config(state=NORMAL)

    # PRINTS A LIST OF PROBABILITIES PASSED BY 'SOFTMAX' FUNCTION OF OUTPUT LAYER.
    prediction = seq.predict(newimg)

    # PICKS THE VALUE WITH MAX VALUE/MAX PROBABILITY IN LIST.
    print(np.argmax(prediction))
    messagebox.showinfo(
        'Predictions', 'Digit predicted is: ' + str(np.argmax(prediction)))

    # CONVERTING THE ARRAY TO LIST FROM 'ndarray' TYPE FOR GRAPH PURPOSES.
    prediction = prediction[0].tolist()
    g = prediction

# IS CALLED WHEN THE 'Upload' BUTTON IS CLICKED IN FRAME 'tab2'.


def upload():
    '''OPENS A DIALOG BOX WHICH HELPS SELECT A IMAGE FILE FROM FILE SYSTEM. '''
    for widget in placeholder.winfo_children():
        widget.destroy()  # CLEARS CANVAS IF THERE IS AN EXISTING IMAGE.

    bruh.file = filedialog.askopenfilename(initialdir='C', title='Select an Image file', filetypes=(
        ('PNG files', '*.png'), ('jpg files', '*.jpg'), ('JPEG files', '*.jpeg'), ('JPG files', '*.JPG'), ('All files', '*.*')))
    global m
    try:
        button2.config(state=NORMAL)
        m = bruh.file
        #print(m)  # GETS THE URL OF FILE SELECTED IN FILE DIALOG.
        img = Image.open(m)
        img = img.resize((150, 150), Image.ANTIALIAS)
        photo_image = ImageTk.PhotoImage(img)
        # UPLOADS THE SELECTED IMAGE TO CANVAS.
        label1 = Label(placeholder, image=photo_image)
        label1.pack()
        images_reference_list.append(photo_image)
    except AttributeError:
        messagebox.showerror('File Error', 'Select a file first!')
    except Exception:
        messagebox.showerror('File Error', 'Upload only a valid image file')

##### IS CALLED WHEN 'Predict' BUTTON IS CLICKED IN FRAME 'tab2'. #####


def predup():
    ''' HELPS PREDICT THE IMAGE PASSED FROM FILEDIALOG. '''
    try:
        img = cv.imread(m)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        thresh = cv.threshold(
            img, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
        #cv.imwrite('./icon1.png', thresh)
        imgr = cv.resize(thresh, (28, 28))
        a = np.array(imgr)
        res = a.flatten()
        # print(a)
        countw = -1
        countb = -1
        for i in res:
            if(i > 0):
                countw += 1

            if(i == 0):
                countb += 1

        if(countw > countb):
            if(v.get() == 0):
                pass
            else:
                thresh = (255-thresh)

        # print(countw)
        # print(countb)
        #plt.imshow(thresh)
       # cv.imwrite('./icon2.png', thresh)
        img1 = cv.resize(thresh, (28, 28), interpolation=cv.INTER_AREA)
        newimg = tf.keras.utils.normalize(img1, axis=1)
        #plt.imshow(newimg)
        #cv.imwrite('./icon3.png', newimg)
        newimg = np.array(newimg).reshape(-1, 28, 28, 1)

        # print(a.shape)
        # print(x_test[200].shape)
        # print(res)
        r = seq.predict([newimg])
        print(np.argmax(r))
        msg = messagebox.askquestion('Prediction','Is the Digit: '+ str(np.argmax(r))+'?',icon = 'warning')
        if msg == 'yes':
           pass
        else:
            thresh = (255-thresh)
            img1 = cv.resize(thresh, (28, 28), interpolation=cv.INTER_AREA)
            newimg = tf.keras.utils.normalize(img1, axis=1)
            newimg = np.array(newimg).reshape(-1, 28, 28, 1)
            r = seq.predict([newimg])
            messagebox.showinfo('Prediction', 'Prediction is: ' + str(np.argmax(r)))


    except Exception:
        messagebox.showerror('File Error', 'Select a file first!')


##### IS CALLED WHEN 'Show Graph' BUTTON IS CLICKED IN 'tab1'. #####
def showGraph():
    '''SHOWS A GRAPH OF ANALYSIS OF THE PAST IMAGE THAT WAS PREDICTED. '''
    graph = Tk()
    data1 = {'Digits': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
             'Accuracy': g
             }
    # USING PANDAS TO MAKE A DATAFRAME OBJECT.
    df1 = DataFrame(data1, columns=['Digits', 'Accuracy'])
    figure1 = plt.Figure(figsize=(10, 10), dpi=70)  # GRAPH DIMS.
    ax1 = figure1.add_subplot(111)
    # ADDING THE GRAPH TO 'graph' WINDOW.
    bar1 = FigureCanvasTkAgg(figure1, graph)
    bar1.get_tk_widget().pack(side=LEFT, fill=BOTH)
    df1 = df1[['Digits', 'Accuracy']].groupby('Digits').sum()
    df1.plot(kind='bar', legend=True, ax=ax1)
    ax1.set_title('Accuracy Scores')

    graph.mainloop()

##### IS CALLED WHEN LINK IS CLICKED IN 'about' TAB. #####


def browse(url):
    '''OPENS A LINK IN DEFAULT BROWSER. '''
    webbrowser.open_new(url)


##### BASIC TKINTER WINDOW PROPERTIES. #####
bruh = Tk()
bruh.title('Image to Text')
bruh.geometry('400x400')
bruh.resizable(0, 0)
icon = PhotoImage(file=paths['logo'])
bruh.iconphoto(False, icon)

##### TAB CUSTOM STYLING. #####
framebg = '#333333'
buttonblue = '#428bca'
taboff = "#1e81b0"
tabor = "#f57713"
style = ttk.Style()
style.theme_create("dark", parent="classic", settings={
    "TNotebook": {"configure": {"tabmargins": [2, 5, 2, 0]}},
    "TNotebook.Tab": {
        "configure": {"padding": [10, 5], "background": taboff},
        "map":       {"background": [("selected", tabor)],
                      "expand": [("selected", [1, 1, 1, 0])]}}})

style.theme_use("dark")


# Create Tab Control
tabControl = ttk.Notebook(bruh)
# Tab1
tab1 = ttk.Frame(tabControl)
tabControl.add(tab1, text='Draw')
# Tab2
tab2 = ttk.Frame(tabControl)
tabControl.add(tab2, text='Upload')
# Tab3
tab3 = ttk.Frame(tabControl)
tabControl.add(tab3, text='About')

tabControl.pack(expand=1, fill="both")

##############

f1 = Frame(tab1, bg=framebg)
c1 = Canvas(f1, height=200, width=200, bg='black')
f1.place(relheight=1, relwidth=1.0)
l1 = Label(f1, text='Draw a digit in Canvas',
           bg=framebg, fg=tabor, font=('Arial', 19))
l1.place(rely=0, relx=0)

# TAB 1
tip1 = Balloon(bruh)  # BUTTON HINTS.
tip2 = Balloon(bruh)
tip3 = Balloon(bruh)

b3 = Button(f1, text='Show Graph', state=DISABLED, command=showGraph)
b3.place(rely=0, relx=0.8, relheight=0.1, relwidth=0.2)
tip3.bind_widget(b3, balloonmsg='Click to see analysis of prediction')
c = Canvas(c1, width=200, height=200, bg='black')
c1.place(relheight=0.8, relwidth=1, rely=0.1)
f3 = Frame(f1)
b1 = Button(f3, text='Clear', background=buttonblue, command=clear)
b1.place(relwidth=0.5, relheight=1)
tip1.bind_widget(b1, balloonmsg='Press this to clear the canvas above')
b2 = Button(f3, text='Predict', background=buttonblue,
            command=predict, state=DISABLED)
b2.place(relwidth=0.5, relheight=1, relx=0.5)
tip2.bind_widget(b2, balloonmsg='Press this to predict the digit drawn.')
c.pack(expand=YES, fill=BOTH)
c.bind('<B1-Motion>', paint)
f3.place(relwidth=1, relheight=0.1, rely=0.9)

# TAB 2

tip4 = Balloon(bruh)
tip5 = Balloon(bruh)
v = IntVar(bruh, 1)
frame1 = Frame(tab2, bg=framebg)
frame1.place(relheight=1.0, relwidth=1.0)
bframe = Frame(frame1, bg='red')
bframe.place(relheight=0.1, relwidth=1, rely=0.9)
picframe = Frame(frame1, bg='black')
picframe.place(relwidth=0.8, relheight=0.7, rely=0.1, relx=0.1)
#ch = Checkbutton(picframe, text='Alt Method', var=v,
                # onvalue=0, offvalue=1, bg=framebg, fg=tabor)

#r1 = Radiobutton(picframe,text='Hack',var=v,value=1,bg='black',fg=tabor)
#r2 = Radiobutton(picframe,text='Method 2',var=v,value=2,tristatevalue=NONE,bg='black',fg=tabor)
#ch.place(rely=0.9, relx=0)
# r2.place(rely=0.9,relx=0.25)

placeholder = Frame(picframe, bg=framebg)
placeholder.place(relheight=0.5, relwidth=0.5, relx=0.25, rely=0.25)
button1 = Button(bframe, bg=buttonblue, text='Upload', command=upload)
button1.place(relwidth=0.5, relheight=1)
tip4.bind_widget(button1, balloonmsg='Click here an image from file system')
button2 = Button(bframe, bg=buttonblue, text='Predict',
                 command=predup, state=DISABLED)
button2.place(relwidth=0.5, relheight=1, relx=0.5)
tip5.bind_widget(
    button2, balloonmsg='Click here to predict the uploaded Image')

# TAB 3
Fabout = Frame(tab3, bg=framebg)
Fabout.place(relheight=1, relwidth=1)
t3f1 = Frame(tab3, bg=framebg)
t3f1.place(relheight=0.1, relwidth=1)
l3 = Label(t3f1, text=about['title'], bg=framebg, fg=tabor, font=('Ariel', 20))
l3.pack()
t3f2 = Frame(tab3, bg='red')
t3f2.place(relheight=0.3, relwidth=1, rely=0.1)
t1 = Text(t3f2, wrap=WORD, background=framebg,
          foreground='white', borderwidth=0, padx=10)
t1.insert('1.0', about['desc'])
t1.config(state=DISABLED, font=('Arial', 14))
t1.pack(side=LEFT)
t3f3 = Frame(tab3, bg=framebg)
t3f3.place(relheight=0.1, relwidth=1, rely=0.4)
l4 = Label(t3f3, text='Click here to access Github!',
           bg=framebg, fg=tabor, font=('Arial', 15))
l4.bind('<Button-1>', lambda e: browse(about['link']))
l4.pack(side=LEFT)
t3f4 = Frame(tab3, bg=framebg)
t3f4.place(relheight=0.4, relwidth=1, rely=0.5)
t3 = Text(t3f4, background=framebg, foreground='white', borderwidth=0, padx=10)
t3.insert('1.0', about['cr'])
t3.config(state=DISABLED, font=('Arial', 14))
t3.pack(side=LEFT)
bruh.mainloop()


# DOCUMENTATION WRITTED BY /B72.
