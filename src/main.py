import numpy as np
import math
import webcolors
import colorsys
import cv2

face_cascade_file = "haarcascade_frontalface_default.xml"
haar_face_cascade = cv2.CascadeClassifier(face_cascade_file)
eye_cascade_file = "haarcascade_eye_tree_eyeglasses.xml"
haar_eye_cascade = cv2.CascadeClassifier(eye_cascade_file)

cam = cv2.VideoCapture(0)

#returns an image from the camera
def get_image():
	_, img = cam.read()
	if _:
		return img

#uses cascade detection to find faces in the image
def detect_face(_img):
	face = haar_face_cascade.detectMultiScale(_img, 1.1, 4)
	return face


#uses cascade detection to locate eyes in a face
def detect_eyes(_face):
	eyes = haar_eye_cascade.detectMultiScale(_face, 1.1, 4)
	return eyes

#isolates the iris, and then uses a color histogram to determine the person's eye color
#TODO: re-write using RGB values. Use the split method to get them. See https://docs.opencv.org/3.4/d8/dbc/tutorial_histogram_calculation.html for more info
def get_eye_color(_eyes, _img):
	for (x,y,w,r) in _eyes:

		#extract iris region
		_iris = _img[y:y+r, x:x+w]
		bgr_planes = cv2.split(_iris)

		hist_size = 256
		hist_range = (0, hist_size)

		accumulate = False

		#the image is in BGR instead of RGB because...idk I'd have to google it.
		b_hist = cv2.calcHist(bgr_planes, [0], None, [hist_size], hist_range, accumulate=accumulate)
		g_hist = cv2.calcHist(bgr_planes, [1], None, [hist_size], hist_range, accumulate=accumulate)
		r_hist = cv2.calcHist(bgr_planes, [2], None, [hist_size], hist_range, accumulate=accumulate)

		hist_h = 400
		cv2.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
		cv2.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
		cv2.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)

		#todo: extract bgr values with the max on the histogram

		return _col

#takes in HSV color code, and returns the closest HTML color name.
def name_eye_color(_col):

	#predefined dictionary for eye colors
	colors = {
		"brown" : (94, 49, 34),
		"blue" : (69, 96, 139),
		"green" : (129, 121, 31),
		"hazel" : (131, 122, 91),
		"gray" : (146, 146, 146)
	}

	min_distance = float("inf")

	closest_color = None
	for color, value in colors.items():
		distance = math.sqrt(((_col[0] + value[0]) ** 2) + ((_col[1] + value[1]) ** 2) + ((_col[2] + value[2]) ** 2))
		if distance < min_distance:
			min_distance = distance
			closest_color = color
	return closest_color


		

def draw_rects(_img, _face, _eyes):
	for (x,y,w,h) in _face:
		_eyes_img = _img[y:y+h, x:x+w]
		cv2.rectangle(_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
		for (ex,ey,ew,eh) in _eyes:
			cv2.rectangle(_eyes_img, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
	return _img



#infinite loop to read camera input
#TODO: this is temporarily getting video input from the webcam. In the final implementation we will use an image sent over from the app.
while True:

	img = get_image()
	_clone = img
	grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	face = detect_face(grayscale)
	eyes_gray = grayscale
	for (x,y,w,h) in face:
		eyes_gray = grayscale[y:y+h, x:x+w]
	eyes = detect_eyes(eyes_gray)
	col = get_eye_color(eyes, _clone)
	if col is not None:
		col_name = name_eye_color(col)
		print(col_name)

	img = draw_rects(img, face, eyes)
	text = "Face not detected"

	##cam.read returns a success boolean and the image.
	#_, img = cam.read()
#
	##default the text to say no face detected.
	#text = "No face detected"
#
	##doing this in grayscale makes processing faster
	#grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
	##detectMultiScale takes in the image, the scale factor (default 1.3), and the number of neighbors.
	##for the scale factor, higher values give faster execution at the cost of accuracy (minimum 1.0).
	##for the number of neighbors, higher values lead to fewer false positives, but can miss unclear faces
	#face = haar_face_cascade.detectMultiScale(grayscale, 1.1, 4)
	#
	##draw a rectangle around detected faces, and look for eyes.
	#for (x, y, w, h) in face:
	#	cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
	#	eyes_img = img[y:y+h, x:x+w]
	#	eyes_gray = grayscale[y:y+h, x:x+w]
	#	eyes = haar_eye_cascade.detectMultiScale(eyes_gray, 1.1, 4)
	#	for (ex, ey, ew, eh) in eyes:
	#		cv2.rectangle(eyes_img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
#
		
		
	
	if len(face) != 0:
		text = "Face Detected"

	#add the text to the image
	image = cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

	
	# display the output window 
	cv2.imshow("Face Detection", image)
	print(text)

	#press the escape key to exit
	key = cv2.waitKey(10)
	if key == 27:
		break

cam.release()
cv2.destroyAllWindows()