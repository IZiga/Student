import cv2
import numpy as np

# installer https://github.com/opencv/opencv/tree/4.x/data/haarcascades
faces = cv2.CascadeClassifier("models/face.xml")

capture = cv2.VideoCapture(0)
i = 0
while True:
    success, img = capture.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = faces.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=1)
    photo = np.zeros((480, 640), dtype='uint8')
    j = 0
    for (x, y, w, h) in results:
        square = cv2.rectangle(photo.copy(), (x, y), (x + w, y+h), 255, thickness=-1)
        photo = cv2.bitwise_or(photo, square) 
        resized = cv2.resize(img[y:(y+h), x: (x+w)], (64, 64), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f'Mymedia/t3{i}-{j}.png', resized)
        j += 1
    img = cv2.bitwise_and(img, img, mask=photo)
    cv2.imshow("Test", img)
    if cv2.waitKey(500) & 0xFF == ord('q'):
        break
    i += 1

