
from distutils.log import info
import email
from importlib.resources import contents
from multiprocessing import context
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login ,logout
from django.contrib import messages
from .forms import usernameForm

from datetime import datetime
from django.contrib.auth.decorators import login_required

# Create your views here.
from .forms import CreateUserForm
import face_recognition
import cv2
import numpy as np
from PIL import Image 
import pickle
import os
import pandas as pd
import csv

def loginPage(request):
 if request.user.is_authenticated:
        return redirect('home')
 else:

    if request.method=='POST':
       
        username=request.POST.get('username')
        password=request.POST.get('password')

        user = authenticate(request, username=username, password =password)
        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            messages.info(request,'Username or password is incorrect')
    page='login'
    context={}
    return render(request, 'facerecognition/loginPage.html',context)

def registerPage(request):
    if request.user.is_authenticated:
        return redirect('home')
    else:

        #default form
        form=CreateUserForm()
        page='register'
        
        # #validation
        if request.method=='POST':
            form=CreateUserForm(request.POST)
            if form.is_valid():
                form.save() 
            
                user= form.cleaned_data.get('username')
            
                messages.success(request,'Account was created for ' + user)
                

                return redirect('loginPage')
            
        context={'form': form }
        return render(request, 'facerecognition/registerPage.html', context)


@login_required(login_url='loginPage')
def dashboard(request):
    context={}
    return render(request,'facerecognition/dashboard.html',context)

def logoutUser(request):
    logout(request)
    return redirect('loginPage')

@login_required(login_url='loginPage')
def home(request):
    
    context={}
    return render(request,'facerecognition/home.html',context)



@login_required(login_url='loginPage')
def attendancereport(request):
  
    
        a = pd.read_csv("attendance.csv")
        
        
        htmltable=a.to_html(index=False)
        htmltable=htmltable.replace('border="1"','border="1" style="border-collapse:collapse"')
       


            
        htmlheader='''<html>
        <head>
        <style>
        table.dataframe {
        border: 1px solid #1C6EA4;
        background-color: #EEEEEE;
        text-align: center;
        border-collapse: collapse;
        width:60%;
        }

        table.dataframe td, table.dataframe th {
        border: 1px solid #AAAAAA;
        padding: 3px 2px;
        text-align: center;
        }

        table.dataframe tbody td {
        font-size: 16px;
        }

        table.dataframe tr:nth-child(even) {
        background: #D0E4F5;
        }

        table.dataframe thead {
        background: #000;
        
        border-bottom: 2px solid #444444;
        }
        table.dataframe thead th {
        font-size: 15px;
        font-weight: bold;
        color: #FFFFFF;
        border-left: 2px solid #D0E4F5;
        }
        table.dataframe thead th:first-child {
        border-left: none;
        }

        table.dataframe tfoot {
        font-size: 14px;
        font-weight: bold;
        color: #FFFFFF;
        background: #D0E4F5;
        
        border-top: 2px solid #444444;
        }
        table.dataframe tfoot td {
        font-size: 14px;
        }
       

        </style>
        '''

        final= htmlheader + htmltable

        
        with open('facerecognition/templates/facerecognition/Table.html', 'w') as file:
                                     
                file.write(final)               
                file.close()   
        return render(request,'facerecognition/attendancereport.html')
   
 
   
    
  
 
# def data(request):
   
#      return render(request, 'facerecognition/data.html')   
 




       


# @login_required(login_url='loginPage')
# def home_section(request):
    
#     context={}
#     return render(request,'facerecognition/home.html#second_container',context)








@login_required(login_url='loginPage')
def profile(request):
    
    context={}
    return render(request,'facerecognition/profile.html',context)





    


    

def add_photos(request):
    		#Create an object to hold reference to camera video capturing
        cam = cv2.VideoCapture(0)
        cam.set(3, 640) # set video width
        cam.set(4, 480) # set video height

        face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        with open('names.pkl', 'rb') as f:
                names = pickle.load(f)
        
        current_user=request.user
        name=  current_user

        names.append(name)
        id = names.index(name)
        print(name)
        print('''\n
            Look in the camera Face Sampling has started!.
            Try to move your face and change expression for better face memory registration.\n
            ''')
    
                # Initialize individual sampling face count
        count = 0

        while(True):
                
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray, 1.3, 5)

                for (x,y,w,h) in faces:

                    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
                    count += 1

                    # Save the captured image into the datasets folder
                    cv2.imwrite("dataset/"+ str(name) + '.' + str(id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

                    cv2.imshow('image', img)

                k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
                if k == 27:
                    break
                elif count >= 80: # Take 80 face sample and stop video
                    break

        with open('names.pkl', 'wb') as f:
                pickle.dump(names, f)

            # Do a bit of cleanup
        print("Your Face has been registered as {}\n\nExiting Sampling Program".format(str(name).upper()))
       
        cam.release()
        cv2.destroyAllWindows()
        return render(request,'facerecognition\home.html')
    

		
def facetraining(request):
    # Path for face image database
    path = 'dataset'

    recognizer = cv2.face.LBPHFaceRecognizer_create() 
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # function to get the images and label data
    def getImagesAndLabels(path):

        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
        faceSamples=[]
        ids = []

        for imagePath in imagePaths:

            PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
            img_numpy = np.array(PIL_img,'uint8')

            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)

            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)

        return faceSamples,ids

    print ("\nTraining for the faces has been started. It might take a while.\n")
    faces,ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))

    # Save the model into trainer/trainer.yml
    recognizer.write('trainer/trainer.yml') 

    # Print the numer of faces trained and end program
    print("{0} faces trained. Exiting Training Program".format(len(np.unique(ids))))  
    return  render(request,'facerecognition\home.html')



def markattendance(request):
    print('\nStarting Recognizer....')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath);

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Starting realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video widht
    cam.set(4, 480) # set video height

    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)

    with open('names.pkl', 'rb') as f:
        names = pickle.load(f)

    while True:

        ret, img =cam.read()

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
           )

        for(x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
           

            # Check if confidence is less them 100 ==> "0" is perfect match 
            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
               
                
                

              


                
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            counts = {}
            counts[id] = counts.get(id, 0) + 1
                #set name which has highest count
            global fname
            fname = max(counts, key=counts.get) 

            cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1) 

        
           
            
            
                    
        
            
        cv2.imshow('camera',img) 
        
        
        

        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k==27:
            break
        
        


    # Do a bit of cleanup
    print(fname)
    abc=request.user
    if(fname==abc):
        with open('attendance.csv', 'a') as file:
            writer = csv.writer(file)
            presenttime= datetime.now()

            #attendance_header=['Name','Date','Time','Day']
            #writer.writerow(attendance_header)
            attendance_data = [fname,presenttime.strftime('%Y-%m-%d'),presenttime.strftime('%H-%M-%S'),presenttime.strftime('%A')]
            
                                    
            writer.writerow(attendance_data)
                                
                            
            file.close()   
    print("\nExiting Recognizer.")
    cam.release()
    cv2.destroyAllWindows()
    return render(request,'facerecognition\home.html')




@login_required
def profile(request):
    context={}
    return render(request, 'facerecognition/profile.html', context)
 


 


     




