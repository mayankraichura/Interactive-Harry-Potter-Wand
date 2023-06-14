from PIL import Image
import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Loading the prpiocessed last frame form Desktop
img = Image.open("/Users/mayankraichura/Desktop/lastframe.jpg")

# Loading the SVM classifier
clf = joblib.load("alphabet_classifier.joblib")

# Converting image to numpy array
img = np.array(img)
# Converting the numpy array to 1-Dimensional array
img = img.reshape(1, -1)


prediction = clf.predict(img)
print(prediction)
