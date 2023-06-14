import os
import warnings
from os import listdir
from os.path import isfile, join, isdir
import cv2
import numpy as np

warnings.filterwarnings("ignore")


TrainingFolderName = "Training"
TrainingResolution = 50
TrainingNumPixels = TrainingResolution * TrainingResolution
IsTraining = True
nameLookup = {}
LastSpell = "None"



def InitClassificationAlgo():
    """
    Create and Train k-Nearest Neighbor Algorithm
    """
    global knn, nameLookup
    labelNames = []
    labelIndexes = []
    trainingSet = []
    numPics = 0
    dirCount = 0
    scriptpath = os.path.realpath(__file__)
    trainingDirectory = join(os.path.dirname(scriptpath), TrainingFolderName)

    # Every folder in the training directory contains a set of images corresponding to a single spell.
    # Loop through all folders to train all spells.
    for d in listdir(trainingDirectory):
        if isdir(join(trainingDirectory, d)):
            nameLookup[dirCount] = d
            dirCount = dirCount + 1
            for f in listdir(join(trainingDirectory, d)):
                f_name = join(trainingDirectory, d, f)
                if isfile(f_name) and cv2.imread(f_name) is not None:
                    labelNames.append(d)
                    labelIndexes.append(dirCount - 1)
                    trainingSet.append(f_name);
                    numPics = numPics + 1

    print("Trained Spells: ")
    print(nameLookup)

    samples = []
    for i in range(0, numPics):
        img = cv2.imread(trainingSet[i])
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        samples.append(gray)
        npArray = np.array(samples)
        shapedArray = npArray.reshape(-1, TrainingNumPixels).astype(np.float32);

    # Create KNN and Train
    knn = cv2.ml.KNearest_create()
    knn.train(shapedArray, cv2.ml.ROW_SAMPLE, np.array(labelIndexes))


def ClassifyImage(img):
    """
    Classify input image based on previously trained k-Nearest Neighbor Algorithm
    """
    global knn, nameLookup, args

    if (img.size <= 0):
        return "Error"

    size = (TrainingResolution, TrainingResolution)
    test_gray = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

    imgArr = np.array(test_gray).astype(np.float32)
    sample = imgArr.reshape(-1, TrainingNumPixels).astype(np.float32)
    ret, result, neighbours, dist = knn.findNearest(sample, k=5)
    print(ret, result, neighbours, dist)

    if IsTraining:
        filename = "char" + str(time.time()).replace(".","") + ".png"
        cv2.imwrite(join(TrainingFolderName, filename), test_gray)

    if nameLookup[ret] is not None:
        print("Match: " + nameLookup[ret])
        return nameLookup[ret]
    else:
        return "error"

InitClassificationAlgo()

while True:
    # For image processing
    import pickle
    import time
    from datetime import datetime

    from sklearn.svm import SVC

    # initializing Picamera
    vid = cv2.VideoCapture(0)

    fgbg = cv2.createBackgroundSubtractorMOG2()

    # Define parameters for the required blob
    params = cv2.SimpleBlobDetector_Params()

    # setting the thresholds
    params.minThreshold = 150
    params.maxThreshold = 250

    # filter by color
    params.filterByColor = 1
    params.blobColor = 255

    # filter by circularity
    params.filterByCircularity = 1
    params.minCircularity = 0.68

    # filter by area
    params.filterByArea = 1
    params.minArea = 30
    # params.maxArea = 1500

    # creating object for SimpleBlobDetector
    detector = cv2.SimpleBlobDetector_create(params)

    flag = 0
    points = []
    lower_blue = np.array([255, 255, 0])
    upper_blue = np.array([255, 255, 0])

    last_found_at = None


    def prepare_svm():
        import pandas as pd
        from sklearn.svm import SVC
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        # print("....Reading DataSet and Creating Pandas DataFrame....")
        alphabet_data = pd.read_csv("A_ZHandwrittenData.csv")
        # print("...DataFrame Created...")

        # print("...Slicing and creating initial training and testing set...")
        # Dataset of letter A containing features
        X_Train_A = alphabet_data.iloc[:13869, 1:]
        # Dataset of letter A containing labels
        Y_Train_A = alphabet_data.iloc[:13869, 0]
        # Dataset of letter C containing features
        X_Train_C = alphabet_data.iloc[22537:45946, 1:]
        # Dataset of letter C containing labels
        Y_Train_C = alphabet_data.iloc[22537:45946, 0]
        # Joining the Datasets of both letters
        X_Train = pd.concat([X_Train_A, X_Train_C], ignore_index=True)
        Y_Train = pd.concat([Y_Train_A, Y_Train_C], ignore_index=True)
        # print("...X_Train and Y_Train created...")

        train_features, test_features, train_labels, test_labels = train_test_split(X_Train, Y_Train, test_size=0.25,
                                                                                    random_state=0)
        # print(type(test_features))

        # SVM classifier created
        clf = SVC(kernel='linear')
        # print("")
        # print("...Training the Model...")
        clf.fit(train_features, train_labels)
        # print("...Model Trained...")

        labels_predicted = clf.predict(test_features)
        # print(test_labels)
        # print(labels_predicted)
        accuracy = accuracy_score(test_labels, labels_predicted)

        # print("")
        # print("Accuracy of the model is:  ")
        # print(accuracy)
        return clf


    def load_data():
        import joblib
        return joblib.load("alphabet_classifier.joblib")


    def validate_clf(clf, i, expected):
        img = cv2.imread("Testing/" + str(i + 1) + ".jpg")
        # print(img.shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        retval, img = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        img = cv2.dilate(img, (3, 3))
        img = img.reshape(1, -1)
        prediction = clf.predict(img)
        # print("OK" if prediction == expected else "NOT OK")


    clf = load_data()

    validate_clf(clf, 1, 0)
    validate_clf(clf, 90, 2)


    # Function for Pre-processing
    def last_frame(img):
        # cv2.imshow("LF", img)
        cv2.imwrite("/Users/mayankraichura/Desktop/lastframe1.jpg", img)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        cv2.imwrite("/Users/mayankraichura/Desktop/lastframe2.jpg", img)
        retval, img = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
        cv2.imwrite("/Users/mayankraichura/Desktop/lastframe3.jpg", img)
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        cv2.imwrite("/Users/mayankraichura/Desktop/lastframe4.jpg", img)
        img = cv2.dilate(img, (3, 3))
        cv2.imwrite("/Users/mayankraichura/Desktop/lastframe.jpg", img)
        # Converting the numpy array to 1-Dimensional array
        img = img.reshape(1, -1)
        output = clf.predict(img)
        print(f"{datetime.now()}: {output}")

        if output == 0:
            print("Alohamora!!")
            print("Box Opened!!")
        if output == 2:
            print("Close!!")
            print("Box Closed!!")


    def last_frame_v2(img):
        cv2.imwrite("/Users/mayankraichura/Desktop/lastframe1.jpg", img)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        cv2.imwrite("/Users/mayankraichura/Desktop/lastframe2.jpg", img)
        retval, img = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
        cv2.imwrite("/Users/mayankraichura/Desktop/lastframe3.jpg", img)
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        cv2.imwrite("/Users/mayankraichura/Desktop/lastframe4.jpg", img)
        img = cv2.dilate(img, (3, 3))
        cv2.imwrite("/Users/mayankraichura/Desktop/lastframe.jpg", img)
        # Converting the numpy array to 1-Dimensional array
        # data = np.array(img).flatten()
        data = img.reshape(1, -1)

        pick = open('model.sav', 'rb')
        model: SVC = pickle.load(pick)
        pick.close()

        output = model.predict(data)
        # print(output)

        if output == 2:
            print("Alohamora!!")
            print("Box Opened!!")
        if output == 3:
            print("Close!!")
            print("Box Closed!!")


    time.sleep(0.1)

    while True:
        # Capture the video frame
        # by frame
        ret, image = vid.read()

        # Mirror the frame along the y-axis
        image = cv2.flip(image, 1)

        fgmask = fgbg.apply(image)

        frame = image.copy()
        frame = cv2.resize(frame, (frame.shape[1] // 3, frame.shape[0] // 3))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # cv2.imshow("original", frame)

        # Subtract Background
        fgmask = fgbg.apply(frame, learningRate=0.001)
        f_no_background = cv2.bitwise_and(frame, frame, mask=fgmask)
        # cv2.imshow("RemovedBAckground", f_no_background)

        # detecting keypoints
        keypoints = detector.detect(frame)

        frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255),
                                                 cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # starting and ending circle
        # frame_with_keypoints = cv2.circle(frame_with_keypoints, (140, 70), 6, (0, 255, 0), 2)
        # frame_with_keypoints = cv2.circle(frame_with_keypoints, (190, 140), 6, (0, 0, 255), 2)
        #
        #
        points_array = cv2.KeyPoint_convert(keypoints)

        if len(points_array):
            last_found_at = datetime.now()
            if flag != 1:
                print("Start tracing")
                flag = 1

        if len(points_array) > 0:
            if flag == 1:
                # Get coordinates of the center of blob from keypoints and append them in points list
                points.append(tuple(map(int, points_array[0])))

                # Draw the path by drawing lines between 2 consecutive points in points list
                for i in range(1, len(points)):
                    cv2.line(frame_with_keypoints, points[i - 1], points[i], (255, 255, 0), 10)

        if flag == 1 and len(points_array) == 0 and (
                datetime.now() - last_found_at).total_seconds() > 1:
            # if int(points_array[0][0]) in range(185, 195) and int(points_array[0][1]) in range(135, 145):
            print("Tracing Done!!")

            if len(points) > 0:
                # Draw the path by drawing lines between 2 consecutive points in points list
                for i in range(1, len(points)):
                    cv2.line(frame_with_keypoints, points[i - 1], points[i], (255, 255, 0), 10)

            try:
                masked = cv2.inRange(frame_with_keypoints, lower_blue, upper_blue)
                contours, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                cropped = masked[y:y + h, x:x + w]

                # cv2.imshow("Final Trace", masked)
                # last_frame(masked)
                ClassifyImage(cropped)
            except:
                pass

            break
            # break

        cv2.imshow("video", frame_with_keypoints)
        # cv2.imshow("video 2", frame)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            vid.release()
            cv2.destroyAllWindows()
            exit(0)

vid.release()
cv2.destroyAllWindows()
