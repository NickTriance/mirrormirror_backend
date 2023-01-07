import cv2

face_cascade_file = "haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(face_cascade_file)

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
	face = haar_cascade.detectMultiScale(grayscale, 1.1, 4)
	
	# draw a rectangle around the face and update the text to Face Detected
	for (x, y, w, h) in face:
		text = "Face Detected"
		cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
	
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