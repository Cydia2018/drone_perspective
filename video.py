import numpy as np
import cv2

cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = frame
    ret2, thresh1 = cv2.threshold(gray, 55, 255, cv2.THRESH_BINARY)
    # ret2, th2 = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Display the resulting frame
    cv2.imshow('frame', thresh1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()