#
#   FINAL PROJECT - SCC0251: Image Processing
#   ICMC-USP, Sao Carlos, 30-06-2017
#
#   Kelvin Guilherme de Oliveira - 9293286
#   Mauricio Caetano da Silva - 9040996
#

from PIL import Image
import cv2
import numpy as np
import sys
import pytesseract

#
# Function convert the image color system
#
def convertToHSV(img):
	# Backup of the original image 
	cpImg = img.copy()

	# Converting the RGB color system to HSV color system
	hsv = cv2.cvtColor(cpImg, cv2.COLOR_BGR2HSV)
	return hsv

#
# Function will search for a price tag on a given image
#
def findPriceTag(img):
	hsv = convertToHSV(img)
	
	cv2.imshow("HSV", hsv)

	# Values to find yellow and green price tags
	# intensities between 15 and 60 will select colors between green and yellow
	lower = np.array([15,100,100], dtype = np.uint8)
	upper = np.array([58,255,255], dtype = np.uint8)

	# Mask used to find regions with the desired color
	mask = cv2.inRange(hsv, lower, upper)

	# Enhancement of the mask borders
	mask = cv2.dilate(mask, (5,5), iterations = 1)
	mask = cv2.erode(mask, (5,5), iterations = 1)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, (3,3))
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, (3,3))
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (3,3))
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (3,3))

	# Extracting edges from the result mask
	edges = cv2.Canny(mask, 100, 250, apertureSize = 3)

	im2,cnts,hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	# Sorting contours by area
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:200]

	# Going through the sorted contours in order to find those that are closest to the price tag
	priceTag = None
	for c in cnts:
	    perimeter = cv2.arcLength(c,True)
	    approx = cv2.approxPolyDP(c, 0.02*perimeter, True)
	    area = cv2.contourArea(approx)

	    # Parameters who defines if a contour will be considered a price tag or not
	    if (len(approx) == 4 and area >= 14000 and area <= 70000 and perimeter >= 620 and perimeter <= 1400):
    	            # Contour approximate in a rectangle
		    (tagX, tagY, tagWidth, tagHeight) = cv2.boundingRect(approx)
		    # Cropped price tag
		    priceTag = img[tagY:tagY+tagHeight, tagX:tagX+tagWidth].copy()
		    # Resizing the price tag to enhance results of tesseract analysis
		    priceTag = cv2.resize(priceTag, (600, 250))

        # End of function
	cv2.destroyAllWindows()
	return priceTag

#
#   Function that prepares the image to the future data extraction.
#
def prepareImage(img):
    
    # Standardizes the image size
    img_resized = cv2.resize(img, (800,400))

    # Applies the threshold on the image to improve it
    thresh = 127
    maxValue = 255
    (ret, img_prepared) = cv2.threshold(img_resized, thresh, maxValue, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Returns the prepared image
    return img_prepared

#
#   Function that gets the contours of an image, returning the contoured image and the contours places.
#
def getContouredImage(img):

    # Creates a black image
    img_contoured = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

    # Finds the contours of the image
    (img_aux, contours, hierarchy) = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Creates a contoured image (green contours) with the contours found
    cv2.drawContours(img_contoured, contours, -1, (0, 255, 0), 3)

    # Returns the contoured image and the contours places
    return (img_contoured, contours)

#
#   Function that separates the main and the discarded contours, following one threshold passed by parameter.
#
def getMainContours(contours, minThreshold = 3000, maxThreshold = 7000, greater = False):
    
    # Creates auxiliar arrays
    mainContours = []
    discardedContours = []
    shapeAreas = []
    
    # Finds the areas inside each contours
    for i in range(len(contours)):
        # Computes the contours[i] area
        shapeAreas.append(cv2.contourArea(contours[i]))
        
    # Finds the largest area
    larger = np.amax(shapeAreas)

    # Applies the threshold in the area, separating the main and discarded contours
    for i in range(len(shapeAreas)):
        if(shapeAreas[i] > minThreshold):
            if(((greater == False and shapeAreas[i] < larger) or (greater == True)) and shapeAreas[i] < maxThreshold):
                mainContours.append(contours[i])
        else: 
            discardedContours.append(contours[i])

    # Returns the main and discarded contours
    return (mainContours, discardedContours)

#
#   Function that finds the main part of the image (correspondig to the price part)
#
def getMainContouredImage(img):

    # Finds the contours of the image
    (img_contoured, contours) = getContouredImage(img)

    # Selects the main contours found
    (mainContours, discardedContours) = getMainContours(contours, 2000)

    # Creates an image with the mainContours
    img_mainContoured = np.ones((img.shape[0], img.shape[1], 3), np.uint8)
    img_mainContoured = np.multiply(img_mainContoured, 255)
    cv2.drawContours(img_mainContoured, mainContours, -1, (0, 0, 0), -1)
    cv2.drawContours(img_mainContoured, discardedContours, -1, (255, 255, 255), -1)

    # Returns the image that contain the main contours
    return (img_mainContoured, mainContours, discardedContours)

#
#   Function that returns the limits of a contour.
#
def getContourLimits(contour):

    # Initializes the max/min values
    maxLin = maxCol = 0
    minLin = minCol = 4000

    # Finds the max/min values
    for i in range(len(contour)):
        if(contour[i][0, 1] > maxLin):
            maxLin = contour[i][0, 1]
        if(contour[i][0, 1] < minLin):
            minLin = contour[i][0, 1]
        if(contour[i][0, 0] > maxCol):
            maxCol = contour[i][0, 0]
        if(contour[i][0, 0] < minCol):
            minCol = contour[i][0, 0]

    # Returns the height and width of the component
    return (minLin, maxLin, minCol, maxCol)

#
#   Function that computes the dimension (height, width) of a component inside a contour
#
def computeContourDimension(contour):
    
    (minLin, maxLin, minCol, maxCol) = getContourLimits(contour)

    # Returns the height and width of the component
    return ((maxLin - minLin), (maxCol - minCol))

#
#   Function that shorts the contours by column.
#
def sortContours(contours):
    
    # Copies the array
    sortedContours = contours.copy()

    # Executes a bubble-sort algorithm
    for i in range(len(sortedContours)):
        for j in range(len(sortedContours)):
           if(i != j):
                a = sortedContours[i][0][0, 0]
                b = sortedContours[j][0][0, 0]
                if(a < b):
                    aux = sortedContours[j].copy()
                    sortedContours[j] = sortedContours[i].copy()
                    sortedContours[i] = aux.copy()

    # Returns the sorted contours array
    return sortedContours

#
#   Function that gets a focused image.
#
def getFocusedImage(img, contours):
    
    # Initializes the parameters
    aMin = bMin = 4000
    aMax = bMax = 0

    # Gets the contours limits parameters
    for i in range(len(contours)):
        (minLin, maxLin, minCol, maxCol) = getContourLimits(contours[i])
        if(minLin < aMin):
            aMin = minLin
        if(minCol < bMin):
            bMin = minCol
        if(maxLin > aMax):
            aMax = maxLin
        if(maxCol > bMax):
            bMax = maxCol

    # Computes the focused image
    img_focused = img[(aMin - 20):(aMax + 20), (bMin - 50):(bMax + 50)]

    # Returns the focused image
    return img_focused

#
#   Function that finds the separation point of the price.
#
def findPointPlace(contours):

    # Initializes the parameters
    diffMax = 0
    point = -1

    # Finds the point place (place with the larger distance between two components)
    for i in range(len(contours)-1):
        (minLinA, maxLinA, minColA, maxColA) = getContourLimits(contours[i])
        (minLinB, maxLinB, minColB, maxColB) = getContourLimits(contours[i+1])
        if((minColB - maxColA) > diffMax):
            diffMax = (minColB - maxColA)
            point = i
    
    # Returns the separation point
    return point;

#
#   Function that computes the distance parameter of an contour to the others contours
#
def computeDistanceParameter(contours, index):
    
    # Initializes the parameters
    dist = 0

    # Gets the parameters of the desired contour
    (minLin, maxLin, minCol, maxCol) = getContourLimits(contours[index])
    
    # Computes the dist parameter
    for i in range(len(contours)):
        if(i != index):
            (minLinB, maxLinB, minColB, maxColB) = getContourLimits(contours[i])
            if(minColB < minCol):
                dist = dist + ((minCol - maxColB)**2)
            else:
                dist = dist + ((minColB - maxCol)**2)

    # Returns the dist parameter
    return dist

#
#   Computes the distance between all contours.
#
def computeDistanceVector(contours):
    
    # Initializes the array
    dist = []

    # Condition
    if(len(contours) == 1):
        dist.append(0)
        return dist
    
    # Computes the distance array
    for i in range(len(contours)):
        distParameter = computeDistanceParameter(contours, i)
        dist.append(distParameter)
    
    # Returns it
    return dist

#
#   Function that performs the last filters on the image, in order to find an image with just the price.
#
def extractPrice(img, mainContours, discardedContours):

    # Creates an auxiliar array
    priceContours = []
    definitiveContours = []
    dist = []
    height = []
    width = []

    # Filters the contours
    for i in range(len(mainContours)):
        perimeter = cv2.arcLength(mainContours[i], True)
        if((perimeter < 700 and perimeter > 300)):
            (h, w) = computeContourDimension(mainContours[i])
            if(h > 80 and w > 15):
                priceContours.append(mainContours[i])
                height.append(round(h, -1))
                width.append(round(w, -1))

    # Computes the median of the dimensions, in order to crop noises
    if(len(priceContours) <= 5):
        medianHeight = np.median(height)
        medianWidth = np.median(width)

    # Computes the median of the dimensions more significants, if there are many contours
    else:
        aux = np.sort(height)
        aux = aux[(len(aux) - 5):(len(aux)-1)]
        medianHeight = np.median(aux)
        aux = np.sort(width)
        aux = aux[(len(aux) - 5):(len(aux)-1)]
        medianWidth = np.median(aux)

    # Filters the contours based on the dimensions of each component
    for i in range(len(height)):
        var = np.square(abs(height[i] - medianHeight) + abs(width[i] - medianWidth))
        if(var < 1000):
            definitiveContours.append(priceContours[i])
    
    # Computes the distances parameter
    dist = computeDistanceVector(definitiveContours)
    
    # Filters the contours based on the dimensions of each component 
    while(np.average(dist) > 10000):
        argmax = np.argmax(dist)
        definitiveContours = np.delete(definitiveContours, argmax)
        dist = computeDistanceVector(definitiveContours)
        
    # Sorts the contours to facilitate the resize of the image
    definitiveContours = sortContours(definitiveContours)

    # Creates an image with the extracted price
    img_extractedPrice = np.ones((img.shape[0], img.shape[1], 3), np.uint8)
    img_extractedPrice = np.multiply(img_extractedPrice, 255)
    cv2.drawContours(img_extractedPrice, definitiveContours, -1, (0, 0, 0), -1)
    cv2.drawContours(img_extractedPrice, discardedContours, -1, (255, 255, 255), -1)
    
    # Focus the price on image
    img_extractedPrice = getFocusedImage(img_extractedPrice, definitiveContours)

    # Finds the point place of the price
    point = findPointPlace(definitiveContours)

    # Returns the image with the price on focus 
    return (img_extractedPrice, point, definitiveContours)

#
#   Function that reads the price contained in image using an OCR algorithm
#
def readPrice(img, separatePoint):

    # Convert the array to image
    tess_img = Image.fromarray(img)

    # Uses the OCR Tesseract
    string_img = pytesseract.image_to_string(tess_img)
    
    # Creates a new string
    price = ""

    # Validates the string found
    if(len(string_img) >= 3):
        # Copies the first part of the price
        i = 0
        while(i <= separatePoint):
            price = price + string_img[i]
            i = i + 1
        # Inserts the point separator
        price = price + "."
        # Copies the last part of the price
        while(i < len(string_img)):
            price = price + string_img[i]
            i = i + 1

    # Insert a error message if the string is not valid
    else:
        price = 'Price not found'

    # Returns the string read
    return price

#
#   Function that tries to get the price, handling the image with others approaches. 
#
def tryGetPrice(img, point, definitiveContours, discardedContours):
    
    # Validates the parameters
    if(len(definitiveContours) < 3):
        return 'Price not found'

    # First Attempt - Erosion
    img_extractedPrice = cv2.erode(img, np.ones((3,3), np.uint8), iterations = 1)
    price = readPrice(img_extractedPrice, point)
    if(price != 'Price not found'):
        return price

    # Second Attempt - Dilation
    img_extractedPrice = cv2.dilate(img, np.ones((3,3), np.uint8), iterations = 1)
    price = readPrice(img_extractedPrice, point)
    if(price != 'Price not found'):
        return price

    # Third Attempt - Closing
    img_extractedPrice = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    price = readPrice(img_extractedPrice, point)
    if(price != 'Price not found'):
        return price
    
    # Returns "Price not found"
    return price

#
#   Shows the images on screen according to the flag
#
def showImages(flag, img_input, priceTag = None, img_prepared = None, img_contoured = None, img_extractedPrice = None):
  
    # Resize the input image to show on screen
    img_input = cv2.resize(img_input, (800, 600))

    # Shows all images
    if(flag == "ALL"):
        cv2.imshow("INPUT IMAGE", img_input)
        cv2.waitKey(0)
        cv2.imshow("PRICE TAG", priceTag)
        cv2.waitKey(0)
        cv2.imshow("PRE PROCESSED IMAGE", img_prepared)
        cv2.waitKey(0)
        cv2.imshow("EXTRACTED PRICE IMAGE", img_extractedPrice)
        cv2.waitKey(0)

    # Shows just the input image
    elif(flag == "INPUT"):
        cv2.imshow("INPUT IMAGE", img_input)
        cv2.waitKey(0)

    # Shows just the input and price tag image
    elif(flag == "PRICE_TAG_EXTRACTION"):
        cv2.imshow("INPUT IMAGE", img_input)
        cv2.waitKey(0)
        cv2.imshow("PRICE TAG", priceTag)
        cv2.waitKey(0)

    # Shows just the price image
    elif(flag == "PRICE"):
        cv2.imshow("EXTRACTED PRICE IMAGE", img_extractedPrice)
        cv2.waitKey(0)

    # Shows just the price image
    elif(flag == "PRICE_EXTRACTION"):
        cv2.imshow("PRICE TAG", priceTag)
        cv2.waitKey(0)
        cv2.imshow("PRE PROCESSED IMAGE", img_prepared)
        cv2.waitKey(0)
        cv2.imshow("CONTOURED IMAGE", img_contoured)
        cv2.waitKey(0)
        cv2.imshow("EXTRACTED PRICE IMAGE", img_extractedPrice)
        cv2.waitKey(0)

    # Shows the input/price image by default ()
    elif(flag != "NONE"):
        cv2.imshow("INPUT IMAGE", img_input)
        cv2.waitKey(0)
        cv2.imshow("EXTRACTED PRICE IMAGE", img_extractedPrice)
        cv2.waitKey(0)

    # Shows nothing if "NONE"
    else:
        return

    # Destroy the windows and return
    cv2.destroyAllWindows()
    return

#
#   Main Function
#
def main():
    
    # Reads the input image
    img_input = cv2.imread(sys.argv[1])

    # Finds the price tag (STEP 1)
    priceTag = findPriceTag(img_input)
    
    # If did not find the price tag, prints a message and terminates the program execution
    if (priceTag is None):
        showImages("INPUT", img_input)
        print('Price tag not found')
        return
    
    # Converts the image to GrayScale
    priceTag = cv2.cvtColor(priceTag, cv2.COLOR_BGR2GRAY)

    # Prepares the image to find the price
    img_prepared = prepareImage(priceTag)

    # Gets the main contours of the image
    (img_contoured, mainContours, discardedContours) = getMainContouredImage(img_prepared)

    # Selects just the price place of the image
    (img_extractedPrice, separatePoint, definitiveContours) = extractPrice(img_contoured, mainContours, discardedContours) 

    # Uses the OCR to find the price placed on image
    price = readPrice(img_extractedPrice, separatePoint)

    # Shows the image
    if(len(sys.argv) > 2):
        showImages(sys.argv[2], img_input, priceTag, img_prepared, img_contoured, img_extractedPrice)
    
    # If price is found, Prints it on screen
    if(price != 'Price not found'):
        print(price)

    # Else, tries to find the price, handling the image
    else:
        print(tryGetPrice(img_extractedPrice, separatePoint, definitiveContours, discardedContours))
    
    # End of execution
    return

# Calls the main function
main()
