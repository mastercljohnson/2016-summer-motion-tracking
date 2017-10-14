#Import Libraries
#import numpy for ez array manipulation
import numpy as np
from numpy import array
from numpy import random

#imports opencv module, parsing, and date time for video manipulation and for processing tracking
import cv2
import argparse
import datetime

import imutils
import time
import threading

#imports scipy module for clustering of points
import scipy
from scipy.cluster.vq import vq,kmeans,whiten
#optional import in case I want to plot any of the points and used for debugging
from matplotlib import pyplot as plt
#python debugger module
import pdb

import re
#used to export excel spreadsheet for further analysis
import xlsxwriter
#used to make program more optimal by providing a file seeker for the video file
import tkinter as tk
from tkinter import filedialog

import os


def end():
    
    workbook=  xlsxwriter.Workbook('actualdataday31000.xlsx')
    worksheet= workbook.add_worksheet()
    worksheet.set_column('A:A',20)
    bold=workbook.add_format({'bold':True})
    worksheet.write('A1','Fish1x',bold)
    worksheet.write('B1','Fish1y',bold)
    worksheet.write('C1', 'Fish2x',bold)
    worksheet.write('D1','Fish2y',bold)
    worksheet.write('E1','Fish3x',bold)
    worksheet.write('F1','Fish3y',bold)
    worksheet.write('G1','fish4x',bold)
    worksheet.write('H1','fish4y',bold)
    worksheet.write('I1','fish5x',bold)
    worksheet.write('J1','fish5y',bold)
    worksheet.write('K1','z1',bold)
    worksheet.write('L1','z2',bold)
    worksheet.write('M1','z3',bold)
    worksheet.write('N1','z4',bold)
    worksheet.write('O1','z5',bold)
    row=1
        
    worksheet.write_column(row,0,fish1)
    worksheet.write_column(row,1,fish11)
    worksheet.write_column(row,2,fish2)
    worksheet.write_column(row,3,fish22)
    worksheet.write_column(row,4,fish3)
    worksheet.write_column(row,5,fish33)
    worksheet.write_column(row,6,fish4)
    worksheet.write_column(row,7,fish44)
    worksheet.write_column(row,8,fish5)
    worksheet.write_column(row,9,fish55)
    worksheet.write_column(row,10,z1array)
    worksheet.write_column(row,11,z2array)
    worksheet.write_column(row,12,z3array)
    worksheet.write_column(row,13,z4array)
    worksheet.write_column(row,14,z5array)
    workbook.close()

    camera.release()
    cv2.destroyAllWindows()


t= threading.Timer(7200.0,end)
t.start()


#parsing used to define the scope that contours are allowed to be found
ap = argparse.ArgumentParser()
ap.add_argument("-v","--video", help="coraldance.avi")
ap.add_argument("-a", "--min_area", type=int,default=1, help="minimum area")
ap.add_argument('-x','--max_area',type=int,default=500, help ='maximum area')
ap.add_argument('-b','--buffer', type=int, default=64, help =' max buffer size')
args = vars(ap.parse_args())
#this array and plotting code was added to help debugging by printing coordinates
pts = np.array(0)

fig=plt.figure()
ax1=fig.add_subplot(121)








#finds path of video input file
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()



#Function is defined to calculate distance between points
def distance(x1,x2,y1,y2):
    return((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))**.5

#x and y coordinates for each square center in each frame appended to these two arrays
i=np.array([])
u=np.array([])


#fish coordinates arrays
#fish x coordinates arrays
fish1=np.array([])
fish2=np.array([])
fish3=np.array([])
fish4=np.array([])
fish5=np.array([])
#fish y coordinates arrays
fish11=np.array([])
fish22=np.array([])
fish33=np.array([])
fish44=np.array([])
fish55=np.array([])

#For the first frame, values for x2?
#are used in order to randomly set the first centroids in the function
x2a=-50000000000
y2a=-50000000000
x2b=-50000000000
y2b=-50000000000
x2c=-50000000000
y2c=-50000000000
x2d=-50000000000
y2d=-50000000000
x2e=-50000000000
y2e=-50000000000


#These arrays are used to export the distance between closest centroids
#between frames into the excel spreadsheet
z1array=np.array([])
z2array=np.array([])
z3array=np.array([])
z4array=np.array([])
z5array=np.array([])

# This was used for debugging
#print(i)
#print(u)

#print('gah')


# Imports video file into program
if args.get("video", None) is None:
    camera= cv2.VideoCapture(file_path)
    time.sleep(0.25)
else:
    camera = cv2.VideoCapture(file_path)
 
     
#Used as a means to set up a reference frame for the analysis of motion
firstFrame= None


#Grabbing the frame from the video file
while True:
    (grabbed, frame)=camera.read()
    text="No fish"
    if not grabbed:
        break
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #filters tested and discarded during calibration of experimental environment
    #gray = cv2.GaussianBlur(gray, (21,21),0)
    #gray=cv2.bilateralFilter(gray,5,10,10)



    #Setting the first reference frame
    if firstFrame is None:
        firstFrame = gray
        continue
    


    #Takes frame and puts through a binary filter and is dilated
    #which takes away some of the noise from glare that may be detected as motion
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta,25, 255, cv2.THRESH_BINARY)[1]
        
    thresh = cv2.dilate(thresh, None, iterations =1)



    #Finding contours and drawing rectangles of detected motion

    (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
     
        if args['max_area'] < cv2.contourArea(c)<args['min_area'] :
            continue
        
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x,y), (x +w, y+h), (0, 255, 0), 2)
        text= "Has fish"


        #Used to define the center of rectangles drawn
        m= round(x+w/2)
        n= round(y+h/2)
        
        #Appends x and y coordinates of centers respectively to the arrays
        i=np.append(i,m)
        
        
        u=np.append(u,n)

        
        
        
        
        


       
        
        
        
        
        

        

            





        

    #This was used during debugging
    #This part was not discarded because
    #it helped track the processing progress of video
    cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.imshow("Fish Feed", frame)
    

    
    #This creates a matrix which is allowed to be clustered
    #using scipy library kmeans
    
    p=np.column_stack((i,u))
    #used for debugging
    #print(p)
    


    #No whitening was done because it distorts
    #the centroids generated since distance
    #is not distorted in the plane
    #Clustering using kmeans for scipy library
    kmeansp = scipy.cluster.vq.kmeans(p,5,iter=10,thresh=1,check_finite=True)
 
    
    
    

    
    

    

    

    
    



    
    
    
    #Extracting kmeans output
    #by splicing out the codebook
    #and converting to an array
    kmeansp= np.array(kmeansp)

    
    

    kmeansp = kmeansp[:1]
 
    kmeansp = np.array(kmeansp)

    
    kmeansp = kmeansp[0]


    



# Chooses random points in the extracted centroids
#Giving them an identity
    
    pick1 = kmeansp[0]

    pick2= kmeansp[1]

    pick3= kmeansp[2]

    pick4= kmeansp[3]

        
#Occasionally kmeans function is bugged
#and outputs one less point
#This allows the last point to exist
#It does not affect the actual results
#because the frame rate is high enough
#to prevent major distortion

    try:
        pick5= kmeansp[4]
        x1e=pick5[0]
        y1e=pick5[1]
    except IndexError:
        x1e=x2e
        y1e=y2e




    




#This extracts the coordinate x and y values
#giving identity
    x1a = pick1[0]
    y1a = pick1[1]
    x1b=pick2[0]
    y1b=pick2[1]
    x1c=pick3[0]
    y1c=pick3[1]
    x1d=pick4[0]
    y1d=pick4[1]



     

    

    

#This allows the first frame
#to randomly append a first centroid value
    
    if x2a<-40000000000 :
        x2a=x1a
    if x2b<-40000000000 :
        x2b=x1b
    if x2c<-40000000000 :
        x2c=x1c
    if x2d<-40000000000 :
        x2d=x1d
    if x2e<-40000000000 :
        x2e=x1e
    if y2a<-40000000000 :
        y2a = y1a
    if y2b<-40000000000 :
        y2b=y1b
    if y2c<-40000000000 :
        y2c=y1c
    if y2d<-40000000000 :
        y2d=y1d
    if y2e<-40000000000 :
        y2e=y1e





#Point is taken out
#Coordinates compared with previous centroid coordinates
#distance is calculated
#and the shortest distance gets appended into coresponding array
#Once a coordinates is used, its values is rewritten into
#the following coordinates therefore preventing repetition
#the value in current frame is reasigned
#to become the value of previous frame in
#next frame
#process is repeated 5 times for 5 centroids
#in each frame

    x2=x2a
    y2=y2a


    x1 =x1a
    y1 =y1a
    r11=distance(x1,x2,y1,y2)


    x1=x1b
    y1=y1b
    r12=distance(x1,x2,y1,y2)


    x1=x1c
    y1=y1c
    r13=distance(x1,x2,y1,y2)


    x1=x1d
    y1=y1d
    r14=distance(x1,x2,y1,y2)


    x1=x1e
    y1=y1e
    r15=distance(x1,x2,y1,y2)


    z1=min(r11,r12,r13,r14,r15)


    if z1==r11:
        x2a=x1a
        y2a=y1a
        x1a=x1b
        y1a=y1b
    if z1==r12:
        x2a=x1b
        y2a=y1b
        x1b=x1c
        y1b=y1c
    if z1==r13:
        x2a=x1c
        y2a=y1c
        x1c=x1d
        y1c=y1d
    if z1==r14:
        x2a=x1d
        y2a=y1d
        x1d=x1e
        y1d=y1e
        
    if z1==r15:
        x2a=x1e
        y2a=y1e
        x1e=x1a
        y1e=y1a
    fish1=np.append(fish1,x2a)
    fish11=np.append(fish11,y2a)



    

    x2=x2b
    y2=y2b

    x1=x1a
    y1=y1a
    r21=distance(x1,x2,y1,y2)


    x1=x1b
    y1=y1b
    r22=distance(x1,x2,y1,y2)


    x1=x1c
    y1=y1c
    r23=distance(x1,x2,y1,y2)


    x1=x1d
    y1=y1d
    r24=distance(x1,x2,y1,y2)


    x1=x1e
    y1=y1e
    r25=distance(x1,x2,y1,y2)


    z2=min(r21,r22,r23,r24,r25)

    if z2==r21:
        x2b=x1a
        y2b=y1a
        x1a=x1b
        y1a=y1b
    if z2==r22:
        x2b=x1b
        y2b=y1b
        x1b=x1c
        y1b=y1c
    if z2==r23:
        x2b=x1c
        y2b=y1c
        x1c=x1d
        y1c=y1d
    if z2==r24:
        x2b=x1d
        y2b=y1d
        x1d=x1e
        y1d=y1e
    if z2==r25:
        x2b=x1e
        y2b=y1e
        x1e=x1a
        y1e=y1a
    fish2=np.append(fish2,x2b)
    fish22=np.append(fish22,y2b)

    

    

    
    x2=x2c
    y2=y2c

    x1=x1a
    y1=y1a
    r31=distance(x1,x2,y1,y2)

    x1=x1b
    y1=y1b
    r32=distance(x1,x2,y1,y2)

    x1=x1c
    y1=y1c
    r33=distance(x1,x2,y1,y2)

    x1=x1d
    y1=y1d
    r34=distance(x1,x2,y1,y2)

    x1=x1e
    y1=y1e
    r35=distance(x1,x2,y1,y2)

    z3=min(r31,r32,r33,r34,r35)
    if z3==r31:
        x2c=x1a
        y2c=y1a
        x1a=x1b
        y1a=y1b
    if z3==r32:
        x2c=x1b
        y2c=y1b
        x1b=x1c
        y1b=y1c
    if z3==r33:
        x2c=x1c
        y2c=y1c
        x1c=x1d
        y1c=y1d
    if z3==r34:
        x2c=x1d
        y2c=y1d
        x1d=x1e
        y1d=y1e
    if z3==r35:
        x2c=x1e
        y2c=y1e
        x1e=x1a
        y1e=y1a
    fish3=np.append(fish3,x2c)
    fish33=np.append(fish33,y2c)

    

    



    x2=x2d
    y2=y2d

    x1=x1a
    y1=y1a
    r41=distance(x1,x2,y1,y2)

    x1=x1b
    y1=y1b
    r42=distance(x1,x2,y1,y2)

    x1=x1c
    y1=y1c
    r43=distance(x1,x2,y1,y2)

    x1=x1d
    y1=y1d
    r44=distance(x1,x2,y1,y2)

    x1=x1e
    y1=y1e
    r45=distance(x1,x2,y1,y2)

    z4=min(r41,r42,r43,r44,r45)
    if z4==r41:
        x2d=x1a
        y2d=y1a
        x1a=x1b
        y1a=y1b
    if z4==r42:
        x2d=x1b
        y2d=y1b
        x1b=x1c
        y1b=y1c
    if z4==r43:
        x2d=x1c
        y2d=y1c
        x1c=x1d
        y1c=y1d
    if z4==r44:
        x2d=x1d
        y2d=y1d
        x1d=x1e
        y1d=y1e
    if z4==r45:
        x2d=x1e
        y2d=y1e
        x1e=x1a
        y1e=y1a
    fish4=np.append(fish4,x2d)
    fish44=np.append(fish44,y2d)



    



    x2=x2e
    y2=y2e

    x1=x1a
    y1=y1a
    r51=distance(x1,x2,y1,y2)

    x1=x1b
    y1=y1b
    r52=distance(x1,x2,y1,y2)

    x1=x1c
    y1=y1c
    r53=distance(x1,x2,y1,y2)

    x1=x1d
    y1=y1d
    r54=distance(x1,x2,y1,y2)

    x1=x1e
    y1=y1e
    r55=distance(x1,x2,y1,y2)

    z5=min(r51,r52,r53,r54,r55)
    if z5==r51:
        x2e=x1a
        y2e=y1a
        x1a=x1b
        y1a=y1b
    if z5==r52:
        x2e=x1b
        y2e=y1b
        x1b=x1c
        y1b=y1c
    if z5==r53:
        x2e=x1c
        y2e=y1c
        x1c=x1d
        y1c=y1d
    if z5==r54:
        x2e=x1d
        y2e=y1d
    if z5==r55:
        x2e=x1e
        y2e=y1e
    fish5=np.append(fish5,x2e)
    fish55=np.append(fish55,y2e)



    z1array=np.append(z1array,z1)
    z2array=np.append(z2array,z2)
    z3array=np.append(z3array,z3)
    z4array=np.append(z4array,z4)
    z5array=np.append(z5array,z5)
    
#Used during debugging
    #print(r11)
  #  print(r12)
  #  print(r13)
  #  print(r14)
  #  print(r15)
  #  print(z1)
   # print('first point')
   # print(r21)
   # print(r22)
   # print(r23)
   # print(r24)
   # print(r25)
   # print(z2)
   # print('second point')
   # print(r31)
   # print(r32)
  #  print(r33)
  #  print(r34)
  #  print(r35)
   # print(z3)
  #  print('thrid point')
  #  print(r41)
  #  print(r42)
  #  print(r43)
  #  print(r44)
  #  print(r45)
   # print(z4)
  #  print('fourthframe')
  #  print(r51)
  #  print(r52)
  #  print(r53)
  #  print(r54)
   # print(r55)
    #print(z5)
   # print('fifth point')


    


    






    

    
   
#Arrays holding the current frames coordinates are refreshed
#for next frames use
    u=np.array([])
    i=np.array([])


    
    #Used for the calibration of experimental environment
    #cv2.imshow("Thresh", thresh)
    #cv2.imshow("Frame Delta", frameDelta)
    key = cv2.waitKey(1) & 0xFF




    #Allows for a manual breaking of loop
    #exports an excel spreadsheet
    #With coordinates paths of 5 centroids
    #using xlsxwriter
    if key == ord("q"):

        
        

        workbook=  xlsxwriter.Workbook('actualdataday21000.xlsx')
        worksheet= workbook.add_worksheet()
        worksheet.set_column('A:A',20)
        bold=workbook.add_format({'bold':True})
        worksheet.write('A1','Fish1x',bold)
        worksheet.write('B1','Fish1y',bold)
        worksheet.write('C1', 'Fish2x',bold)
        worksheet.write('D1','Fish2y',bold)
        worksheet.write('E1','Fish3x',bold)
        worksheet.write('F1','Fish3y',bold)
        worksheet.write('G1','fish4x',bold)
        worksheet.write('H1','fish4y',bold)
        worksheet.write('I1','fish5x',bold)
        worksheet.write('J1','fish5y',bold)
        worksheet.write('K1','z1',bold)
        worksheet.write('L1','z2',bold)
        worksheet.write('M1','z3',bold)
        worksheet.write('N1','z4',bold)
        worksheet.write('O1','z5',bold)
        row=1
        
        worksheet.write_column(row,0,fish1)
        worksheet.write_column(row,1,fish11)
        worksheet.write_column(row,2,fish2)
        worksheet.write_column(row,3,fish22)
        worksheet.write_column(row,4,fish3)
        worksheet.write_column(row,5,fish33)
        worksheet.write_column(row,6,fish4)
        worksheet.write_column(row,7,fish44)
        worksheet.write_column(row,8,fish5)
        worksheet.write_column(row,9,fish55)
        worksheet.write_column(row,10,z1array)
        worksheet.write_column(row,11,z2array)
        worksheet.write_column(row,12,z3array)
        worksheet.write_column(row,13,z4array)
        worksheet.write_column(row,14,z5array)
        workbook.close()
        


#The graphing was used for debugging
        ax1.scatter(i,u,color='blue',s=10,edgecolor='none')
        ax1.set_aspect(1./ax1.get_data_ratio())
        ax1.set_xlim([0,600])
        ax1.set_ylim([0,600])
        
        ax1.grid(True)
        plt.show()

        #This was also used for debugging
        print("this is bnlah")
        print(i)
        print(u)


        break
camera.release()
cv2.destroyAllWindows()
