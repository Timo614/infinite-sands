import numpy as np
import cv2
import cv2_fullscreen
import time 

screen = cv2_fullscreen.FullScreen()

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard((12, 8), 0.015, 0.012, aruco_dict)
board_image = cv2.aruco.CharucoBoard.generateImage(board, (1920, 1080), marginSize=0)

board_image_path = '/home/nineso/Downloads/board-projection.jpg'
cv2.imwrite(board_image_path, board_image)

screen.imshow(board_image)

time.sleep(360)