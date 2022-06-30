import cv2

#Input image




input = cv2.imread('')
 
#Get input size
 
#height, width = input.shape[:2]

#Desired "pixelated" size

#print (height, width) 
#w, h = (256, 256)
 
#Resize input to "pixelated" size
 
#temp = cv2.resize(input, (w, h), interpolation=cv2.INTER_LINEAR)

#Initialize output image
 
#output = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
cv2.imwrite("test.png", input)
