import cv2
import numpy as np
import requests
from io import BytesIO

image_url = "https://programmerhumor.io/wp-content/uploads/2022/11/programmerhumor-io-programming-memes-cfd575d0864fff4.jpg"
response = requests.get(image_url)
image = cv2.imdecode(np.frombuffer(response.content, np.uint8), 1)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
output_path = "output_image.jpg"
cv2.imwrite(output_path, image)
cv2.imshow('Detected Patterns', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
