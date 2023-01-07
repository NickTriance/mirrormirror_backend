import numpy as np
import cv2

face_cascade_file = "haarcascade_frontalface_default.xml"
haar_face_cascade = cv2.CascadeClassifier(face_cascade_file)
eye_cascade_file = "haarcascade_eye.xml"
haar_eye_cascade = cv2.CascadeClassifier(eye_cascade_file)

cam = cv2.VideoCapture(0)

#infinite loop to read camera input
while True:

	#cam.read returns a success boolean and the image.
	_, img = cam.read()

	#default the text to say no face detected.
	text = "No face detected"

	#doing this in grayscale makes processing faster
	grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	#detectMultiScale takes in the image, the scale factor (default 1.3), and the number of neighbors.
	#for the scale factor, higher values give faster execution at the cost of accuracy (minimum 1.0).
	#for the number of neighbors, higher values lead to fewer false positives, but can miss unclear faces
	face = haar_face_cascade.detectMultiScale(grayscale, 1.1, 4)
	
	#draw a rectangle around detected faces, and look for eyes.
	for (x, y, w, h) in face:
		cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
		eyes_img = img[y:y+h, x:x+w]
		eyes_gray = grayscale[y:y+h, x:x+w]
		eyes = haar_eye_cascade.detectMultiScale(eyes_gray, )
		for (ex, ey, ew, eh) in eyes:
			cv2.rectangle(eyes_img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

		
		
	
	#add the text to the image
	image = cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

	if len(face) != 0:
		text = "Face Detected"
	
	# display the output window 
	cv2.imshow("Face Detection", image)
	print(text)

	#press the escape key to exit
	key = cv2.waitKey(10)
	if key == 27:
		break

cam.release()
cv2.destroyAllWindows()