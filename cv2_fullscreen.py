# https://github.com/elerac/fullscreen/blob/master/fullscreen/cv2_fullscreen.py
#
# MIT License
#
# Copyright (c) 2019 Ryota MAEDA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import cv2
import screeninfo  # https://github.com/rr-/screeninfo

class FullScreen:
    """Full-screen with OpenCV High-level GUI backend"""

    delay: int = 1  # internal delay time after imshow

    def __init__(self, screen_id: int = 0):
        self.monitor = screeninfo.get_monitors()[screen_id]
        self.width = self.monitor.width
        self.height = self.monitor.height
        self.x = self.monitor.x
        self.y = self.monitor.y
        self.name = str(screen_id)

        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.name, self.x, self.y)
        cv2.resizeWindow(self.name, self.width, self.height)
        cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # set initial image
        img_gray = np.full((self.height, self.width), 127, dtype=np.uint8)
        self.imshow(img_gray)
        cv2.waitKey(750)  # first imshow require long delay time

    @property
    def shape(self):
        return self.height, self.width, 3

    def imshow(self, image: np.ndarray):
        cv2.imshow(self.name, image)
        cv2.waitKey(self.delay)
        cv2.waitKey(self.delay)  # magic

    def destroyWindow(self):
        cv2.destroyWindow(self.name)