import random
import time

import cv2
import joblib
import warnings

warnings.filterwarnings("ignore")

count = 0
clf = joblib.load("model.sav")
# clf = cv2.ml.SVM_load("alphabet_classifier2.xml")
r = list(range(122))
random.shuffle(r)

for i in r:
    img = cv2.imread("Testing/" + str(i + 1) + ".jpg")
    print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    retval, img = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img = cv2.dilate(img, (3, 3))
    img = img.reshape(1, -1)


    prediction = clf.predict(img)
    print("For " + str(i + 1) + ".jpg:")
    if i + 1 <= 88:
        if prediction == 2:
            count = count + 1
            print('A')
    if i + 1 >= 89:
        if prediction == 3:
            print('C')
            count = count + 1
    print(prediction)

print("Accuracy: ")
print(count / 122)
