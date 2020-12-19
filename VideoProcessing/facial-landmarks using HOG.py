import numpy as np
import math
import cv2
import dlib # facial detection
import imutils  # facial landmarks
from imutils import face_utils
''' -------------------Extracting Signals --------------------- '''
# To capture from Camera
#cap = cv2.VideoCapture(0)
# To play a video
cap = cv2.VideoCapture('E:/Thesis/BioVid Heat Pain Dataset/biovid/download/PartB/video/071309_w_21/071309_w_21-BL1-081.mp4')
# To read image
#cap = cv2.imread('images\example_01.jpg')
facedetector = dlib.get_frontal_face_detector()
landmarkspredictor = dlib.shape_predictor('detectors/shape_predictor_68_face_landmarks.dat')
SignalDistancesPerFrame = []
frames = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if (frame is not None):
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetector(gray, 1)   
        for (i, face) in enumerate(faces):
        	# determine the facial landmarks for the face region, then
        	# convert the facial landmark (x, y)-coordinates to a NumPy array
        	landmarks = landmarkspredictor(gray, face)
        	landmarks = face_utils.shape_to_np(landmarks)
        	# convert dlib's rectangle to a OpenCV-style bounding box
        	# [i.e., (x, y, w, h)], then draw the face bounding box
        	(x, y, w, h) = face_utils.rect_to_bb(face)
        	cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        	# show the face number
        	cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        	
        	index = 0
        	frame_distance = np.zeros(86)
        	# loop over the (x, y)-coordinates for the facial landmarks and draw them on the image
        	COG = [np.sum(landmarks[:,0])/68,np.sum(landmarks[:,1])/68]
        	for (x, y) in landmarks:
        		frame_distance[index] = math.sqrt((x - COG[0])**2 + (y - COG[1])**2)
        		cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
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
        	# show the output image with the face detections + facial landmarks
        	cv2.imshow("Output", frame)
        	if cv2.waitKey(1) & 0xFF == ord('q'):
        		break
    else:
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

''' -------------------Filter Signals --------------------- '''
signal_distance_array = np.asarray(SignalDistancesPerFrame)
signal = np.vstack(signal_distance_array)
N = len(signal[:,0])            # Number of samplepoints
Fs = 25          # Sample Frequency
Ts = 1.0 / Fs       # Sample Period
Nyq = 0.5 * Fs  # Nyquist Frequency
from scipy.signal import butter, lfilter
LowCutoff = 1           # desired low cutoff frequency of the filter, Hz
HighCutoff = 1          # desired high cutoff frequency of the filter, Hz
b, a = butter(N = 4, Wn = LowCutoff/Nyq, btype='low', analog=False)
signal_filtered = lfilter(b, a, signal[:,0])
import matplotlib.pyplot as plt
import numpy as np
t = np.linspace(0.0, N*Ts, N)
plt.title('Signals Visualisation')
plt.xlabel('Time [sec]')
plt.ylabel('Magnitude')
plt.grid()
plt.legend()
plt.plot(t[:200], signal[:200], 'b-', label='data')
plt.plot(t[:200], signal_filtered[:200], 'g-', label='data')
''' -------------------Extracting Features --------------------- '''
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
