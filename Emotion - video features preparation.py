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
            emotion = re.sub(r".mp4.*", "", re.sub(r".*\d*-\d*_\w_\d*-", "", filepath))
            if emotion == 'amusement':
                encode = 0.1
            elif emotion == 'anger':
                encode = 1
            elif emotion == 'disgust':
                encode = 2
            elif emotion in 'fear':
                encode = 3
            else: #sad
                encode = 4
            row = [encode] + row
            #row.append(encode)
            return row
            
def get_features_headers():
    headers = [None] * 860
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
    for landmark_distance in range(68,86):
        headers[0 + landmark_distance*10] = 'Distance #' + str(landmark_distance-68) + ' mean'
        headers[1 + landmark_distance*10] = 'Distance #' + str(landmark_distance-68) + ' std'
        headers[2 + landmark_distance*10] = 'Distance #' + str(landmark_distance-68) + ' amin'
        headers[3 + landmark_distance*10] = 'Distance #' + str(landmark_distance-68) + ' amax'
        headers[4 + landmark_distance*10] = 'Distance #' + str(landmark_distance-68) + ' median'
        headers[5 + landmark_distance*10] = 'Distance #' + str(landmark_distance-68) + ' max-min'
        headers[6 + landmark_distance*10] = 'Distance #' + str(landmark_distance-68) + ' iqr'
        headers[7 + landmark_distance*10] = 'Distance #' + str(landmark_distance-68) + ' mad'
        headers[8 + landmark_distance*10] = 'Distance #' + str(landmark_distance-68) + ' mad'
        headers[9 + landmark_distance*10] = 'Distance #' + str(landmark_distance-68) + ' trim_mean'
    headers = ['Emotion']+headers
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
            	frame_distance = np.zeros(86)
            	COG = [np.sum(landmarks[:,0])/68,np.sum(landmarks[:,1])/68]
            	for (x, y) in landmarks:
            		frame_distance[index] = math.sqrt((x - COG[0])**2 + (y - COG[1])**2)
            		index=index+1
            	frame_distance[68] = math.sqrt((landmarks[20][0] - landmarks[37][0])**2 + (landmarks[20][1] - landmarks[37][1])**2)
            	frame_distance[69] = math.sqrt((landmarks[22][0] - landmarks[28][0])**2 + (landmarks[22][1] - landmarks[28][1])**2)
            	frame_distance[70] = math.sqrt((landmarks[22][0] - landmarks[23][0])**2 + (landmarks[22][1] - landmarks[23][1])**2)
            	frame_distance[71] = math.sqrt((landmarks[22][0] - landmarks[40][0])**2 + (landmarks[22][1] - landmarks[40][1])**2)
            	frame_distance[72] = math.sqrt((landmarks[25][0] - landmarks[46][0])**2 + (landmarks[25][1] - landmarks[46][1])**2)
            	frame_distance[73] = math.sqrt((landmarks[23][0] - landmarks[28][0])**2 + (landmarks[23][1] - landmarks[28][1])**2)
            	frame_distance[74] = math.sqrt((landmarks[23][0] - landmarks[43][0])**2 + (landmarks[23][1] - landmarks[43][1])**2)
            	frame_distance[75] = math.sqrt((landmarks[32][0] - landmarks[40][0])**2 + (landmarks[32][1] - landmarks[40][1])**2)
            	frame_distance[76] = math.sqrt((landmarks[36][0] - landmarks[43][0])**2 + (landmarks[36][1] - landmarks[43][1])**2)
            	frame_distance[77] = math.sqrt((landmarks[37][0] - landmarks[49][0])**2 + (landmarks[37][1] - landmarks[49][1])**2)
            	frame_distance[78] = math.sqrt((landmarks[38][0] - landmarks[42][0])**2 + (landmarks[38][1] - landmarks[42][1])**2)
            	frame_distance[79] = math.sqrt((landmarks[39][0] - landmarks[41][0])**2 + (landmarks[39][1] - landmarks[41][1])**2)
            	frame_distance[80] = math.sqrt((landmarks[44][0] - landmarks[48][0])**2 + (landmarks[44][1] - landmarks[48][1])**2)
            	frame_distance[81] = math.sqrt((landmarks[45][0] - landmarks[47][0])**2 + (landmarks[45][1] - landmarks[47][1])**2)
            	frame_distance[82] = math.sqrt((landmarks[46][0] - landmarks[55][0])**2 + (landmarks[46][1] - landmarks[55][1])**2)
            	frame_distance[83] = math.sqrt((landmarks[49][0] - landmarks[55][0])**2 + (landmarks[49][1] - landmarks[55][1])**2)
            	frame_distance[84] = math.sqrt((landmarks[52][0] - landmarks[63][0])**2 + (landmarks[52][1] - landmarks[63][1])**2)
            	frame_distance[85] = math.sqrt((landmarks[58][0] - landmarks[67][0])**2 + (landmarks[58][1] - landmarks[67][1])**2)
            	SignalDistancesPerFrame.append(frame_distance)
            	frames = frames + 1
            	if cv2.waitKey(1) & 0xFF == ord('q'):
            		break
        else:
            break
    signal_distance_array = np.asarray(SignalDistancesPerFrame)
    print(signal_distance_array)
    signal = signal_distance_array
    features = np.zeros(860)
    from scipy.stats import iqr,stats
    from astropy.stats import median_absolute_deviation
    from statsmodels import robust
    for distance_index in range(0,86):
        features[0 + distance_index*10] = np.mean(signal[:,distance_index])
        features[1 + distance_index*10] = np.std(signal[:,distance_index])
        features[2 + distance_index*10] = np.amin(signal[:,distance_index])
        features[3 + distance_index*10] = np.amax(signal[:,distance_index])
        features[4 + distance_index*10] = np.median(signal[:,distance_index])
        features[5 + distance_index*10] = np.max(signal[:,distance_index]) - np.amin(signal[:,distance_index])
        features[6 + distance_index*10] = iqr(signal[:,distance_index])
        features[7 + distance_index*10] = median_absolute_deviation(signal[:,distance_index])
        features[8 + distance_index*10] = robust.mad(signal[:,distance_index])
        features[9 + distance_index*10] = stats.trim_mean(signal[:,distance_index],0.1)
    return features

def GenerateFeatures():
    #Absolute path
    filepath='E:/Thesis/videos_frontal/videos_frontal'
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
    filepath='LearningAlgorithm/features/BioVid EmoDB - Facial Expressions'
    Files= get_filepaths(filepath,".csv")
    print("Total Files : " +str(len(Files)))
    index=0;
    aggregate_features = []
    for f in  Files:
        try:
            print("Path : " + str(f[0]))
            aggregate_features.append(from_csv_to_list(f[0]))
            index=index+1;
            print("Finish File    :   ************** " +str(index)+ "/"+str(len(Files)))
        except IndexError:
            index=index+1;
    return aggregate_features

#GenerateFeatures()
headers = get_features_headers()
aggregate_features = CombineFeatures()
aggregate_features = [headers] + aggregate_features
from_arraylist_to_csv(aggregate_features,'LearningAlgorithm/features/Emotion Features Combined.csv')