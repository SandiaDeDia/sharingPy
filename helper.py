import cv2 as cv
import numpy as np


def sharpen(my_image):
    if cv.is_grayscale(my_image):
        height, width = my_image.shape
    else:
        my_image = cv.cvtColor(my_image, cv.CV_8U)
        height, width, n_channels = my_image.shape
    result = np.zeros(my_image.shape, my_image.dtype)
    
    for j in range(1, height - 1):
        for i in range(1, width - 1):
            if cv.is_grayscale(my_image):
                sum_value = 5 * my_image[j, i] - my_image[j + 1, i] - my_image[j - 1, i] \
                            - my_image[j, i + 1] - my_image[j, i - 1]
                result[j, i] =cv.saturated(sum_value)
            else:
                for k in range(0, n_channels):
                    sum_value = 5 * my_image[j, i, k] - my_image[j + 1, i, k]  \
                                - my_image[j - 1, i, k] - my_image[j, i + 1, k]\
                                - my_image[j, i - 1, k]
                    result[j, i, k] = cv.saturated(sum_value)
    
    return result


#shift the image to one side
def  shiftimage(image,shift):
    for i in range(image.shape[1] -1, image.shape[1] - shift, -1):
        image = np.roll(image, -1, axis=1)
        image[:, -1] = 0
        return image
   
    
#Example using fast
def myFast(img,img2):
    # Initiate FAST object with default values
    fast = cv.FastFeatureDetector_create()
    # find and draw the keypoints
    kp = fast.detect(img,None)
    
    #img2 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
    # Print all default params
    print( "Threshold: {}".format(fast.getThreshold()) )
    print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
    print( "neighborhood: {}".format(fast.getType()) )
    print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )
    cv.imwrite('fast_true.png',img2)
    
    # Disable nonmaxSuppression
    
    fast.setNonmaxSuppression(0)
    kp = fast.detect(img,None)
    print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )
    img3 = cv.drawKeypoints(img, kp, None, color=(255,0,0))

    cv.imshow('image2', img2)
    cv.imshow('image', img3)
    cv.waitKey()