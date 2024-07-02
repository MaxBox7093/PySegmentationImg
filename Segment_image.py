import numpy as np
import cv2 as cv
import matplotlib
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog
matplotlib.use('TkAgg') 

def Canny():
    img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)

    if img is None:
        print ('Error opening image')
        print ('Program Arguments: [image_name -- default img.test2.jpeg]')
        return -1

    edges = cv.Canny(img,100,200)

    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()

def Laplacian():
    ddepth = cv.CV_16S
    kernel_size = 3
    window_name = "Laplace Demo"

    imageName = file_path
    src = cv.imread(cv.samples.findFile(imageName), cv.IMREAD_COLOR)
    img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
    
    if src is None:
        print ('Error opening image')
        print ('Program Arguments: [image_name -- default img.test2.jpeg]')
        return -1
    
    src = cv.GaussianBlur(src, (3, 3), 0)
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)
    dst = cv.Laplacian(src_gray, ddepth, ksize=kernel_size)
    abs_dst = cv.convertScaleAbs(dst)

    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(122),plt.imshow(abs_dst,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()

def Sobel():
    img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)

    if img is None:
        print ('Error opening image')
        print ('Program Arguments: [image_name -- default img.test2.jpeg]')
        return -1

    sobelx = cv.Sobel(img,cv.CV_8U,1,0,ksize=5)
    sobely = cv.Sobel(img,cv.CV_8U,0,1,ksize=5)

    plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,3,2),plt.imshow(sobelx,cmap = 'gray')
    plt.title('SobelX'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,3,3),plt.imshow(sobely,cmap = 'gray')
    plt.title('SobelY'), plt.xticks([]), plt.yticks([])

    plt.show()

def Morf_grad():
    img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
    
    if img is None:
        print ('Error opening image')
        print ('Program Arguments: [image_name -- default img.test2.jpeg]')
        return -1

    kernel = np.ones((5,5),np.uint8)
    gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)
    tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
    blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)

    plt.subplot(2,2,1), plt.imshow(img, cmap="gray")
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(2,2,2), plt.imshow(gradient, cmap="gray")
    plt.title('Gradient'), plt.xticks([]), plt.yticks([])

    plt.subplot(2,2,3), plt.imshow(tophat, cmap="gray")
    plt.title('Tophat'), plt.xticks([]), plt.yticks([])

    plt.subplot(2,2,4), plt.imshow(blackhat, cmap="gray")
    plt.title('Blackhat'), plt.xticks([]), plt.yticks([])

    plt.show()

def load_image():
    file_path = filedialog.askopenfilename()
    return file_path

file_path = load_image()

root = tk.Tk()
root.title('Градиентная сегментация')

canny_button = tk.Button(root, text='Canny', command=Canny)
canny_button.pack()

canny_button = tk.Button(root, text='Laplacian', command=Laplacian)
canny_button.pack()

canny_button = tk.Button(root, text='Sobel', command=Sobel)
canny_button.pack()

canny_button = tk.Button(root, text='Morf_grad', command=Morf_grad)
canny_button.pack()

root.mainloop()