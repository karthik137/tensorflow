import cv2

import numpy as np 
from scipy import misc 
import matplotlib.pyplot as plt 

imgdata = misc.ascent()

#print(data[0])

### Use python library to draw images
plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(imgdata)
#plt.show()

'''
Transform the image 
'''
#print(imgdata)

transformed_data = np.copy(imgdata)
print(transformed_data.shape)

size_x = transformed_data.shape[0]
size_y = transformed_data.shape[1]

### Now we will create a filter of 3 x 3 array

filter = [[-1,-2,-1],
          [0,0,0],
          [1,2,1]]
weight = 1



'''
3x3 grid can be mentioned like this -->
              
 (2-1, 2-1)   (2-1,2)       (2-1,2+1)

 (2,2-1)        22          (2,2+1)

(2+1, 2-1)    (2+1,2)       (2+1,2+1)
'''


for x in range(1, size_x-1):
    for y in range(1, size_y-1):
        convolution = 0.0
        convolution = convolution + (imgdata[x - 1, y-1] * filter[0][0])
        convolution = convolution + (imgdata[x, y-1] * filter[0][1])
        convolution = convolution + (imgdata[x + 1, y-1] * filter[0][2])
        convolution = convolution + (imgdata[x-1, y] * filter[1][0])
        convolution = convolution + (imgdata[x, y] * filter[1][1])
        convolution = convolution + (imgdata[x+1, y] * filter[1][2])
        convolution = convolution + (imgdata[x-1, y+1] * filter[2][0])
        convolution = convolution + (imgdata[x, y+1] * filter[2][1])
        convolution = convolution + (imgdata[x+1, y+1] * filter[2][2])
        convolution = convolution * weight
        if(convolution<0):
            convolution=0
        if(convolution>255):
            convolution=255
        transformed_data[x, y] = convolution



### Plot the image
'''
plt.gray()
plt.grid(False)
plt.imshow(transformed_data)
#plt.axis('off')
plt.show()

'''

### Pooling on image

'''

The following code will show (2,2) pooling.The idea here is to iterate over the image, and look at the pixel and it's immediate neighbors to the right, beneath, and right-beneath. 
Take the largest of them and load it into the new image. Thus the new image will be 1/4 the size of the old -- with the dimensions on X and Y being halved by this process.

'''

new_x = int(size_x/2)
new_y = int(size_y/2)
newImage = np.zeros((new_x, new_y)) # Returns a new array with all zeros

for x in range(0, size_x, 2):
    for y in range(0, size_y, 2):
        pixels = []
        pixels.append(transformed_data[x,y])
        pixels.append(transformed_data[x+1, y])
        pixels.append(transformed_data[x,y+1])
        pixels.append(transformed_data[x+1, y+1])
        newImage[int(x/2),int(y/2)] = max(pixels)

# Plot the image
plt.gray()
plt.grid(False)
plt.imshow(newImage)
plt.show()









