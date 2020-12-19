def get_filepaths(directory,extension=''):
   import os.path
   file_paths =[]
   for (dirpath, dirnames, files) in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(dirpath, filename)
            if filename.endswith(extension) and not os.path.isfile(filepath+".csv") :
                file_paths.append((filepath,filename))  # Add it to the list.
   return file_paths;
   
def from_list_to_csv(_list,filepath):
    import csv
    with open(filepath,'w',newline='') as resultFile:
        wr = csv.writer(resultFile)
        wr.writerow(_list)
        
def from_arraylist_to_csv(_list,filepath):
    import csv
    with open(filepath,'w',newline='') as resultFile:
        wr = csv.writer(resultFile)
        for row in _list:
            wr.writerow(row)
        
def from_csv_to_list(filepath):
    import csv
    import re
    with open(filepath,'r',newline='') as resultFile:
        wr = csv.reader(resultFile)
        for row in wr:
            participant = re.sub(r"-.*", "", re.sub(r".*\\", "", filepath))
            pain = re.sub(r".*\\", "", filepath)
            pain = re.sub(r"-.*", "", re.sub(r".{11}-", "", pain))
            if pain == 'BL1':
                encode = 0.1
            elif pain == 'PA1':
                encode = 1
            elif pain == 'PA2':
                encode = 2
            elif pain in 'PA3':
                encode = 3
            else: #sad
                encode = 4
            filename = re.sub(r".mp4.csv", "", re.sub(r".*\\", "", filepath))
            row = [participant,encode,filename] + row
            #row.append(encode)
            return row
            
def get_features_headers():
    headers = [None] * 680
    for landmark_distance in range(0,68):
        headers[0 + landmark_distance*10] = 'Landmark #' + str(landmark_distance) + ' mean'
        headers[1 + landmark_distance*10] = 'Landmark #' + str(landmark_distance) + ' std'
        headers[2 + landmark_distance*10] = 'Landmark #' + str(landmark_distance) + ' amin'
        headers[3 + landmark_distance*10] = 'Landmark #' + str(landmark_distance) + ' amax'
        headers[4 + landmark_distance*10] = 'Landmark #' + str(landmark_distance) + ' median'
        headers[5 + landmark_distance*10] = 'Landmark #' + str(landmark_distance) + ' max-min'
        headers[6 + landmark_distance*10] = 'Landmark #' + str(landmark_distance) + ' iqr'
        headers[7 + landmark_distance*10] = 'Landmark #' + str(landmark_distance) + ' mad'
        headers[8 + landmark_distance*10] = 'Landmark #' + str(landmark_distance) + ' mad'
        headers[9 + landmark_distance*10] = 'Landmark #' + str(landmark_distance) + ' trim_mean'
    headers = ['Participant','Level','FileName']+headers
    return headers

def from_vidoe_to_features(filepath):
    import numpy as np
    import math
    import cv2
    import dlib # facial detection
    import imutils  # facial landmarks
    from imutils import face_utils
    cap = cv2.VideoCapture(filepath)
    facedetector = dlib.get_frontal_face_detector()
    landmarkspredictor = dlib.shape_predictor('VideoProcessing/detectors/shape_predictor_68_face_landmarks.dat')
    SignalDistancesPerFrame = []
    frames = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if (frame is not None):
            frame = imutils.resize(frame, width=500)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facedetector(gray, 1)
            for (i, face) in enumerate(faces):
            	landmarks = landmarkspredictor(gray, face)
            	landmarks = face_utils.shape_to_np(landmarks)
            	index = 0
            	frame_distance = np.zeros(68)
            	COG = [np.sum(landmarks[:,0])/68,np.sum(landmarks[:,1])/68]
            	for (x, y) in landmarks:
            		frame_distance[index] = math.sqrt((x - COG[0])**2 + (y - COG[1])**2)
            		index=index+1
            	SignalDistancesPerFrame.append(frame_distance)
            	frames = frames + 1
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    signal_distance_array = np.asarray(SignalDistancesPerFrame)
    features = np.zeros(680)
    from scipy.stats import iqr,stats
    from astropy.stats import median_absolute_deviation
    from statsmodels import robust
    for distance_index in range(0,68):
        features[0 + distance_index*10] = np.mean(signal_distance_array[:,distance_index])
        features[1 + distance_index*10] = np.std(signal_distance_array[:,distance_index])
        features[2 + distance_index*10] = np.amin(signal_distance_array[:,distance_index])
        features[3 + distance_index*10] = np.amax(signal_distance_array[:,distance_index])
        features[4 + distance_index*10] = np.median(signal_distance_array[:,distance_index])
        features[5 + distance_index*10] = np.max(signal_distance_array[:,distance_index]) - np.amin(signal_distance_array[:,distance_index])
        features[6 + distance_index*10] = iqr(signal_distance_array[:,distance_index])
        features[7 + distance_index*10] = median_absolute_deviation(signal_distance_array[:,distance_index])
        features[8 + distance_index*10] = robust.mad(signal_distance_array[:,distance_index])
        features[9 + distance_index*10] = stats.trim_mean(signal_distance_array[:,distance_index],0.1)
    return features

def GenerateFeatures():
    #Absolute path
    filepath='E:/Thesis/BioVid Heat Pain Dataset/biovid/download/PartB/video'
    Files= get_filepaths(filepath,".mp4")
    print("Total Files : " +str(len(Files)))
    index=0;
    for f in  Files:
        try:
            print("Path : " + str(f[0]))
            lst=from_vidoe_to_features(f[0]); # get your array Here.
            from_list_to_csv(lst,f[0]+".csv")
            index=index+1;
            print("Finish File    :   ************** " +str(index)+ "/"+str(len(Files)))
        except IndexError:
            index=index+1;
            
def CombineFeatures():
    filepath='LearningAlgorithm/features/BioVid Heat Pain - Part B - Facial Expressions'
    Files= get_filepaths(filepath,".csv")
    print("Total Files : " +str(len(Files)))
    index=0;
    aggregate_features = []
    for f in Files:
        try:
            print("Path : " + str(f[0]))
            aggregate_features.append(from_csv_to_list(f[0]))
            index=index+1;
            print("Finish File    :   ************** " +str(index)+ "/"+str(len(Files)))
        except IndexError:
            index=index+1;
    return aggregate_features
    
headers = get_features_headers()
aggregate_features = CombineFeatures()
aggregate_features = [headers] + aggregate_features
from_arraylist_to_csv(aggregate_features,'LearningAlgorithm/features/Pain Features Combined.csv')