import mouth_detector as md
import numpy as np
import cv
import cv2
import logistic
from PIL import Image

WIDTH, HEIGHT = 28, 10
dim = WIDTH * HEIGHT

def vectorize(filename):
	size = WIDTH, HEIGHT
	image = Image.open(filename)
	grey_image = image.resize(size, Image.ANTIALIAS).convert('L')
	return np.array(grey_image).reshape(1, size[0] * size[1])


img = cv2.imread('testn.jpg', 0)
mouth = md.detect_mouth(cv.fromarray(img))
cv.SaveImage('mouth.jpg', mouth)

lr = logistic.Logistic(dim)
lr.weights = np.load('smile_model.npy')

cap = cv2.VideoCapture(0)
while(True):
	ret, frame = cap.read()
	fmouth = md.detect_mouth(cv.fromarray(frame))
	if(fmouth == 0):		# no mouth detected
		continue
	cv.SaveImage('mouth.jpg', fmouth)
	cv2.imshow('frame',frame)
	print lr.predict(vectorize('mouth.jpg'))	
	if cv2.waitKey(1) == 27:	# ESC
		break

cap.release()



