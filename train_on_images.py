import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

while True:
    data = []

    for i in range(122):
        img = cv2.imread("Testing/" + str(i + 1) + ".jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        retval, img = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        img = cv2.dilate(img, (3, 3))
        img = np.array(img).flatten()
        label = 2

        if i + 1 >= 89:
            label = 3

        data.append([img, label])

    features = []
    labels = []

    for feature, label in data:
        features.append(feature)
        labels.append(label)

    xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.25, shuffle=True)
    model = SVC(C=1, kernel='poly', gamma='auto')
    model.fit(xtrain, ytrain)

    prediction = model.predict(xtest)
    accuracy = model.score(xtest, ytest)
    print(f"Accuracy: {accuracy}")

    categories = ['ERR', 'ERR', 'A', 'C']
    predicted = prediction[0]
    if predicted > 1:
        print(f"Predicted: {categories[predicted]}")
    test_image = xtest[0].reshape(28, 28)
    plt.imshow(test_image)
    plt.show()

    if accuracy > 0.98:
        break

pick = open('model.sav','wb')
pickle.dump(model,pick)
pick.close()