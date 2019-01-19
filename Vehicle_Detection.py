import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from random import randint
from sklearn import svm
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import svm, grid_search, datasets
import pickle
import os
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
import threading
from collections import deque

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(32, 32), xy_overlap=(0.5, 0.5)):
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

def predict(inputimg, model, X_scaler):
    img = cv2.resize(inputimg,(64,64))
    X = extractFeatures(img)
    scaled_X = X_scaler.transform(X)
    prediction = model.predict(scaled_X)
    return prediction

def loadTrainingData():
    vehicle_images_udacity = glob.glob('test_data/vehicles/*/*.png')
    nonvehicle_images_udacity = glob.glob('test_data/non-vehicles/*/*.png')
    cars = vehicle_images_udacity
    notcars = nonvehicle_images_udacity
    return cars,notcars

def prepareTrainingData(cars,notcars):
    print('Number of Car examples',len(cars))
    print('Number of Not-Car examples',len(notcars))
    X=[]
    y=[]

    for car in cars:
        temp = cv2.imread(car)
        car_image = cv2.resize(temp,(64,64))
        X_temp = extractFeatures(car_image)
        y_temp = 1
        X.append(X_temp)
        y.append(y_temp)
    for notcar in notcars:
        temp = cv2.imread(notcar)
        notcar_image = cv2.resize(temp,(64,64))
        X_temp = extractFeatures(notcar_image)
        y_temp = 0
        X.append(X_temp)
        y.append(y_temp)

    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)
    return scaled_X,y,X_scaler

def plotRandomPair(cars,notcars):
    notcar_ind = randint(0, len(notcars))
    car_ind = randint(0, len(cars))
    car_image = mpimg.imread(cars[car_ind])
    notcar_image = mpimg.imread(notcars[notcar_ind])

    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(car_image)
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(notcar_image)
    plt.title('Example Not-car Image')
    plt.savefig('output_images/example_images.png')


def splitTestData(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return X_train,X_test,y_train,y_test

def extractRAW(img, size=(32, 32)):
    small_img = cv2.resize(img, size).ravel()
    return raw

def extractHOG(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:      
        #Parameter Hardcoded for final version
        hog = cv2.HOGDescriptor((64,64), (8,8), (8,8), (8,8), 9)
        features = hog.compute(img)
        return np.ravel(features)

def extractHOC(img, nbins=32, bins_range=(0, 256)):
    cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    hhist = np.histogram(img[:,:,0], bins=32, range=(0, 256))
    shist = np.histogram(img[:,:,1], bins=32, range=(0, 256))
    vhist = np.histogram(img[:,:,2], bins=32, range=(0, 256))
    hist_features = np.concatenate((hhist[0], shist[0], vhist[0]))
    return hist_features

def extractFeatures(img):
    features = []
    orient = 9
    pix_per_cell = 4
    cell_per_block = 2

    YCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    featuresHOGY = extractHOG(YCrCb[:,:,0],orient,pix_per_cell,cell_per_block,False,True)
    featuresHOGCr = extractHOG(YCrCb[:,:,1],orient,pix_per_cell,cell_per_block,False,True)
    featuresHOGCb = extractHOG(YCrCb[:,:,2],orient,pix_per_cell,cell_per_block,False,True)
    featuresHOC = extractHOC(img, nbins=32, bins_range=(0, 256))

    output = np.concatenate((featuresHOGY,featuresHOGCr,featuresHOGCb, featuresHOC)).astype(np.float64)  
    return output

def plotHOGParameterset(cars,notcars,orient, pix_per_cell, cell_per_block):
    notcar_ind = randint(0, len(notcars))
    car_ind = randint(0, len(cars))
    car_image = mpimg.imread(cars[car_ind])
    notcar_image = mpimg.imread(notcars[notcar_ind])

    car_image_gray = cv2.cvtColor(car_image, cv2.COLOR_RGB2GRAY)
    notcar_image_gray = cv2.cvtColor(notcar_image, cv2.COLOR_RGB2GRAY)

    features,car_image_hog = extractHOG(car_image_gray, orient, pix_per_cell, cell_per_block, True, False)
    features,notcar_image_hog = extractHOG(notcar_image_gray, orient, pix_per_cell, cell_per_block, True, False)

    fig = plt.figure()
    plt.subplot(221)
    plt.imshow(car_image, cmap='gray')
    plt.title('Image')

    plt.subplot(222)
    plt.imshow(car_image_hog, cmap='gray')
    plt.title('HOG')

    plt.subplot(223)
    plt.imshow(notcar_image, cmap='gray')
    plt.title('Image')

    plt.subplot(224)
    plt.imshow(notcar_image_hog, cmap='gray')
    plt.title('HOG')

    plt.savefig('output_images/example_hog.png')

def trainClassifier(X,y):
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    svr = svm.SVC()
    clf = grid_search.GridSearchCV(svr, parameters)
    clf.fit(X, y)
    return clf

def predictClass(X,clf):
    y = clf.predict(X)
    #proba = clf.predict_prob(X)
    #print(proba)
    #if(proba>0.5):
    #    return 0
    #if(proba>0.5):
    #    return 1
    return y

def evaluateModel(X_test,y_test,clf):
    y_predict = clf.predict(X_test)
    accuracy = accuracy_score(y_predict, y_test)
    return accuracy


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def vehicleWindowDetect(img,model, X_scaler,x_start_stop=[700, 1200], y_start_stop=[375, 500], xy_window=(70, 70), xy_overlap=(0.90, 0.90)):
    detectionBoxes = []
    windows = slide_window(img, x_start_stop, y_start_stop, xy_window, xy_overlap)
    for window in windows:
        window_extract = img[window[0][1]:window[1][1],window[0][0]:window[1][0]]
        prediction = predict(window_extract,model,X_scaler)
        if prediction==1:
            detectionBoxes.append(window)
    return detectionBoxes

def detectVehicles(img, model, X_scaler, previousDetections):
    detectionBoxes = []
    #Sliding Window
    window3 = vehicleWindowDetect(img,model, X_scaler,x_start_stop=[700, None], y_start_stop=[375, 700], xy_window=(90, 90), xy_overlap=(0.90, 0.8))
    window4 = vehicleWindowDetect(img,model, X_scaler,x_start_stop=[700, None], y_start_stop=[375, 700], xy_window=(100, 100), xy_overlap=(0.9, 0.8))
    window5 = vehicleWindowDetect(img,model, X_scaler,x_start_stop=[700, None], y_start_stop=[375, 700], xy_window=(110, 110), xy_overlap=(0.9, 0.8))

    for box in window3:
        detectionBoxes.append(box)
    for box in window4:
        detectionBoxes.append(box)
    for box in window5:
        detectionBoxes.append(box)

    return detectionBoxes

def getBoxSize(box):
    width=box[1][1]-box[0][1]
    height=box[1][0]-box[0][0]
    return height, width, ratio

def showSearchpattern(img):
    detectionBoxes = []
    #Sliding Window
    #Horizon Region
    windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[375, 500], xy_window=(90, 90), xy_overlap=(0.90, 0.80))
    for window in windows:
        detectionBoxes.append(window)

    windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[375, 700], xy_window=(100, 100), xy_overlap=(0.80, 0.80))
    for window in windows:
        detectionBoxes.append(window)

    windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[375, 700], xy_window=(110, 110), xy_overlap=(0.70, 0.80))
    for window in windows:
        detectionBoxes.append(window)

    outputImage = draw_detections(img, detectionBoxes)
    return outputImage

def estimateBoundingBox():
    return boundingBoxes

def draw_detections(img, bboxes, color=(0, 0, 255), thick=6):
    draw_img = np.copy(img)

    for box in bboxes:
        cv2.rectangle(draw_img, box[0], box[1], color, thick)
    return draw_img

def markVehicles(image, detections):
    return result_image

def detect_vehicles_video(img):
    previousDetections = []
    detections = detectVehicles(img, clf, X_scaler, previousDetections)
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    heat = add_heat(heat, detections)

    heatmaps.append(heat)
    combined = np.sum(heatmaps, axis=0)

    heat = apply_threshold(combined, 4)
    heatmap = np.clip(heat, 0, 255)

    labels = label(heatmap)
    label_image = draw_labeled_bboxes(np.copy(img), labels)
    return label_image


heatmaps = deque(maxlen = 3)

cars,notcars = loadTrainingData()
plotRandomPair(cars,notcars)
X,y,X_scaler = prepareTrainingData(cars,notcars)
X_train,X_test,y_train,y_test = splitTestData(X,y)
#plotHOGParameterset(cars,notcars,12, 8, 4)

if not os.path.isfile('finalized_model.sav'):
    print('No Model saved yet - fitting')
    clf = trainClassifier(X_train,y_train)
    pickle.dump(clf, open('finalized_model.sav', 'wb'))
    print('Determine Accuracy')
    print(evaluateModel(X_test,y_test,clf))
else:
    print('Load model...')
    clf = pickle.load(open('finalized_model.sav', 'rb'))
#detections = detectVehicles(image, clf, previousDetections)
#output_image = markVehicles(image, detections)

print('Pipeline-Test')
vehicle_temp = glob.glob('test_data/vehicles/*/*.png')
testimg = cv2.imread(vehicle_temp[0])
print(predict(testimg, clf, X_scaler))

process_images = True

if process_images==True:
    test_images = glob.glob('test_images/*.jpg')
    for image_name in test_images:
        print('Processing '+image_name)
        image = cv2.imread(image_name)
        #Plot Searchpattern
        print('Show searching pattern')
        searchpattern = showSearchpattern(image)
        cv2.imshow('searchpattern',searchpattern)
        outfilename = image_name.split('.')[0]+'_searchpattern.jpg'
        outfilename = os.getcwd()+'/output_images/'+outfilename.split('/')[1]
        print(outfilename)
        cv2.imwrite(outfilename, searchpattern)
        cv2.waitKey(1000)
        #Detect
        print('Detect vehicle')
        previousDetections = []
        detections = detectVehicles(image, clf, X_scaler, previousDetections)
        #Draw Heatmap
        print('Generate Heatmap')
        heat = np.zeros_like(image[:,:,0]).astype(np.float)
        heat = add_heat(heat, detections)
        heat = apply_threshold(heat, 4)
        
        heatmap = np.clip(heat, 0, 255)
        cv2.imshow('heatmap',heatmap)
        outfilename = image_name.split('.')[0]+'_heatmap.jpg'
        outfilename = os.getcwd()+'/output_images/'+outfilename.split('/')[1]
        cv2.imwrite(outfilename, heatmap)
        cv2.waitKey(1000)
        #Draw Detections
        processed_image = draw_detections(image, detections)
        cv2.imshow('processedimage',processed_image)
        outfilename = image_name.split('.')[0]+'_processed.jpg'
        outfilename = os.getcwd()+'/output_images/'+outfilename.split('/')[1]
        cv2.imwrite(outfilename, processed_image)
        cv2.waitKey(1000)
        #Draw Filtered
        labels = label(heatmap)
        label_image = draw_labeled_bboxes(np.copy(image), labels)
        outfilename = image_name.split('.')[0]+'_labels.jpg'
        outfilename = os.getcwd()+'/output_images/'+outfilename.split('/')[1]
        cv2.imwrite(outfilename, label_image)
        cv2.imshow('labels',label_image)
        cv2.waitKey(5000)

#Load test_video.mp4
#load project_video.mp4
filename = 'test_video.mp4'
print("Processing: "+filename)
white_output = filename
white_output = "output_videos/"+white_output.split(".")[0]+"_processed.mp4"
print(white_output)
clip = VideoFileClip(filename)
white_clip = clip.fl_image(detect_vehicles_video)
white_clip.write_videofile(white_output, audio=False)

filename = 'project_video.mp4'
print("Processing: "+filename)
white_output = filename
white_output = "output_videos/"+white_output.split(".")[0]+"_processed.mp4"
print(white_output)
clip = VideoFileClip(filename)
white_clip = clip.fl_image(detect_vehicles_video)
white_clip.write_videofile(white_output, audio=False)
