import cv2
import numpy as np 
# cap = cv2.VideoCapture("coin.mp4")
cap = cv2.VideoCapture(0)

while cap.read():
	ref,frame = cap.read()
	# roi = frame[:1080,0:1920]
	roi = frame

	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	gray_blur = cv2.GaussianBlur(gray,(15,15),0)
	thresh = cv2.adaptiveThreshold(gray_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,1)
	kernel = np.ones((2,2),np.uint8)
	closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel,iterations=2)

	result_img = closing.copy()
	contours,hierachy = cv2.findContours(result_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	counter = 0
	for cnt in contours:
		area = cv2.contourArea(cnt)
		if area<5000 or area>35000:
			continue
		ellipse = cv2.fitEllipse(cnt)
		cv2.ellipse(roi,ellipse,(0,255,0),2)
		counter += 1

	cv2.putText(roi,str(counter),(10,100),cv2.FONT_HERSHEY_SIMPLEX,4,(255,0,0),2,cv2.LINE_AA)
	cv2.imshow("Show",roi)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
# img = np.ones((800,800,3),dtype=np.uint8)*255
# cv2.ellipse(img,(400,400),(200,150),0,0,360,(255,0,0),10)
# cv2.imshow('image',img)
# cv2.waitKey()
# cv2.destroyAllWindows()
