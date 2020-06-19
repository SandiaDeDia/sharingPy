import cv2
import numpy as np
import helper as helper



path="D:/0WORK/opencv/"
img = cv2.imread(path+'pic1_a.png')
image= cv2.imread(path+'pic1_a.png')

helper.myFast(img, image)

image=cv2.blur(image,(5,5))
image=cv2.blur(image,(5,5))

for i in range(0,1,1):
    image=image-helper.shiftimage(image,2)
    image=cv2.blur(image,(2,2))

# image=shiftimage(image,2)
# image=cv2.blur(image,(15,15))
# image=image-img

cv2.imshow('image', image)
cv2.waitKey()


cv2.fastNlMeansDenoisingColored

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)

lines = cv2.HoughLines(edges,1,np.pi/180,200)
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imwrite(path+'houghlines3.jpg',img)

