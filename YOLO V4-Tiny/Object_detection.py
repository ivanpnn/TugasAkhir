from cProfile import label
from unittest import result
from matplotlib import colors, image
import cv2
import numpy as np
import os
import imutils
import time

NMS_THRESHOLD=0.4
MIN_CONFIDENCE=0.3



timeframe= time.time()
frame_id = 0
carCount = 0
frame_no = 1

carInFrame = 0
truckInFrame = 0
busInFrame = 0
othersInFrame = 0

def upper_line_threshold_for_detection(frame_shape):

	return int(frame_shape[0] / 2) + 2

def lower_line_threshold_for_detection(frame_shape):

	return int(frame_shape[0] / 2) - 2


def car_detection(image, model, layer_name, carID=2):
	global carCount, carInFrame, busInFrame, truckInFrame, othersInFrame
	(H, W) = image.shape[:2]
	results = []
	#Mengatur kualitas video
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (250, 250), swapRB=True, crop=False)
	model.setInput(blob)
	layerOutputs = model.forward(layer_name)

	boxes = []
	centroids = []
	confidences = []
	classIDs = []

	carInFrame = 0
	truckInFrame = 0
	busInFrame = 0

	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			if confidence > MIN_CONFIDENCE:
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				boxes.append([x, y, int(width), int(height)])
				centroids.append((centerX, centerY))
				confidences.append(float(confidence))
				classIDs.append(classID)

				if classID == carID:
					carInFrame += 1
				elif classID == 5:
					busInFrame += 1
				elif classID == 7:
					truckInFrame += 1
				else:
					othersInFrame += 1

				# Line to count car
				if (lower_line_threshold_for_detection(image.shape)) < y < (upper_line_threshold_for_detection(image.shape)):
					if classID == carID:
						carCount += 1

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idzs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)
	# ensure at least one detection exists
	if len(idzs) > 0:
		# loop over the indexes we are keeping
		for i in idzs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			confidence = confidences[i]

			# update our results list to consist of the person
			# prediction probability, bounding box coordinates,
			# and the centroid
			label = str(LABELS[classIDs[i]])
			cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y), font, 1, (0, 255, 0), 2)

			res = (confidences[i], (x, y, x + w, y + h), centroids[i])
			results.append(res)
	# return the list of results
	return results

labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

weights_path = "yolov4-tiny.weights"
config_path = "yolov4-tiny.cfg"
#weights_path = "yolov3.weights"
#config_path = "yolov3.cfg"

model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
'''
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
'''

layer_name = model.getLayerNames()
layer_name = [layer_name[i - 1] for i in model.getUnconnectedOutLayers()]
cap = cv2.VideoCapture("traffic.mp4")
writer = None
font = cv2.FONT_HERSHEY_PLAIN

while True:
	(grabbed, frame) = cap.read()
	frame_id += 1
	
	height, width, channels = frame.shape
		
	if not grabbed:
		break
	frame = imutils.resize(frame, width=700)
	results = car_detection(frame, model, layer_name, carID=LABELS.index("car"))

	for res in results:
		cv2.rectangle(frame, (res[1][0],res[1][1]), (res[1][2],res[1][3]), (0, 255, 0), 2)
	#FPS
	elapsed_time = time.time() - timeframe
	fps = frame_id / elapsed_time
	cv2.putText(frame, str(round(fps,2)), (10, 50), font, 2, (0, 255, 0), 2)
	cv2.putText(frame, "FPS", (120, 50), font, 2, (0, 255, 0), 2)

	#counter
	count1 = len(results) 
	cv2.putText(frame, "jumlah =" + " " + str(carCount), (10, 80), font, 2, (0, 255, 0))

	cv2.line(frame, (0, upper_line_threshold_for_detection(frame.shape)), (int(frame.shape[1]), upper_line_threshold_for_detection(frame.shape)), (0, 0, 200), 1)
	cv2.line(frame, (0, lower_line_threshold_for_detection(frame.shape)), (int(frame.shape[1]), lower_line_threshold_for_detection(frame.shape)), (0, 0, 200), 1)

	with open('log.txt', 'a') as f:
		f.write('Frame = ' + str(frame_id) + '\n')
		f.write('Car Total = ' + str(carCount) + '\n')
		f.write('Car in Frame = ' + str(carInFrame) + '\n')
		f.write('Bus in Frame = ' + str(busInFrame) + '\n')
		f.write('Truck in Frame = ' + str(truckInFrame) + '\n')
		f.write('Others in Frame = ' + str(othersInFrame) + '\n' + '\n')



	cv2.imshow("Detection", frame)

	key = cv2.waitKey(1)
	if key == 27:
		break

cap.release()
cv2.destroyAllWindows()


