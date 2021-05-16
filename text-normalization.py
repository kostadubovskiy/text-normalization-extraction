from PIL import Image
import pytesseract
import numpy as np
import cv2

filename = '/Users/kosta/Coding/OCR/Tesseract Experimentation/Tesseract Sample.jpg'
img1 = np.array(Image.open(filename))
text = pytesseract.image_to_string(img1)

print(text)

filename2 = '/Users/kosta/Coding/OCR/Tesseract Experimentation/Tesseract Sample with Noise.png'
img2 = np.array(Image.open(filename2))

img3 = Image.fromarray(img2, 'RGB')
img3.show()

norm_img = np.zeros((img2.shape[0], img2.shape[1]))
img2 = cv2.normalize(img2, norm_img, 0, 255, cv2.NORM_MINMAX)
img2 = cv2.threshold(img2, 100, 255, cv2.THRESH_BINARY)[1]
img2 = cv2.GaussianBlur(img2, (1, 1), 0)

text2 = pytesseract.image_to_string(img2)

img4 = Image.fromarray(img2, 'RGB')
img4.show()

print(text2)
