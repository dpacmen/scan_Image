from transform.transform import four_point_transform
from transform import imutils
from skimage.filters import threshold_adaptive
import numpy as np
import argparse
import cv2

#construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
args = vars(ap.parse_args())

# load the image, clone it, and resize it
image = cv2.imread(args["image"])
ratio = image.shape[0] / 600.0
original = image.copy()
image = imutils.resize(image, height = 600)

# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

#canny edge detection method
detect_edge = cv2.Canny(gray, 75, 200)

# show the original image and the edge detected image
print("Edge Detection")
cv2.imshow("Image", image)
cv2.imshow("Edged", detect_edge)
cv2.waitKey(0)
cv2.destroyAllWindows()

# find the contours
(edge_contours, _) = cv2.findContours(detect_edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

#reverse sort and select max four contours
edge_contours = sorted(edge_contours, key = cv2.contourArea, reverse = True)[:5]

# loop over the contours
for c in edge_contours:
	# approximate the contour
	perimeter = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)

	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	if len(approx) == 4:
		screen_contours = approx
		break

# show the contour (outline) of the piece of paper
print "STEP 2: Find contours of paper"
cv2.drawContours(image, [screen_contours], -1, (255, 0, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# apply the four point transform to obtain a top-down
warped = four_point_transform(orig, screen_contours.reshape(4, 2) * ratio)

# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

warped = threshold_adaptive(warped, 251, offset = 10)
warped = warped.astype("uint8") * 255

# show the original and scanned images
print "STEP 3: Apply perspective transform"
cv2.imshow("Original", imutils.resize(original, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.waitKey(0)
cv2.destroyAllWindows()