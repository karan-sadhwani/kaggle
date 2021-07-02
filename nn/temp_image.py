import numpy as np
import scipy
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
import cv2


image = np.zeros([3509,2480,3],dtype=np.uint8)
image.fill(255) # or img[:] = 255
cv2.imwrite("./data/output/error.png", image)


# fname = './data/raw/1428_7xa25s6vg35s8ki1_University_Hospitals_31Mar19_0.png'
# image = cv2.imread(fname)
# print(image.shape)

# image = np.array(ndimage.cv2.imread(fname, flatten=False))
# print(image)
# my_image = scipy.misc.imresize(image, size=(64,64))
# # my_image = my_image.reshape((1, 64*64*3)).T
# my_image = flatten_images(my_image)
# image = normalise_image(my_image)

image = Image.open('./data/output/error.png')
draw = ImageDraw.Draw(image)
font = ImageFont.truetype('./data/raw/Arial Bold.ttf', 100)
# font = ImageFont.load_default()
# font = ImageFont.ImageFont.getsize('font.ttf', 'hello')
x, y = 50, 100
message = "Error processing document:"
color = 'rgb(0, 0, 0)' # black color
draw.text((x, y), message, fill=color, font=font)

x, y = 50, 300
error = 'Karan'
color = 'rgb(0, 0, 0)' # w color
draw.text((x, y), error, fill=color, font=font)
 
# save the edited image
 
image.save('./data/output/error_message.png')