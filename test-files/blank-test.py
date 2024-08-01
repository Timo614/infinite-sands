import numpy as np
import cv2
import cv2_fullscreen
import time

# Create an instance of the FullScreen class
screen = cv2_fullscreen.FullScreen()

# Create a white image (255 for all pixels in all channels)
white_image = np.ones((1080, 1920, 3), dtype=np.uint8) * 255

# Display the white image in full screen
screen.imshow(white_image)

# Keep the white screen displayed for 60 seconds
time.sleep(60)

# Close the full screen display
screen.close()