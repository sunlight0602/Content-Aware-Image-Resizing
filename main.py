#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 17:14:50 2020

Read and show image
"""

import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('./img.png')
if img is None:
    sys.exit("no img")
    
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
