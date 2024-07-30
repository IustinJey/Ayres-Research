import cv2
import pytesseract
from PIL import Image
import numpy as np
import io

def recognize_number_plate(image_array, crop_box):
    
    """
    image_array este un array din numpy care reprezinta imaginea
    Acum functia utilizeaza poza rezultata in urma crop-ului
    Daca trimitem in aceasta functie toata poza, trebuie decomentate cele 3 linii de cod
    si pe linia gray=... sa se inlocuiasca image_array cu cropped_image 
    """

    #pil_image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    #cropped_img = pil_image.crop(crop_box)
    #cropped_image = np.array(cropped_img)

   
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    
    binary_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)

   
    denoised_image = cv2.medianBlur(binary_image, 3)

   
    height, width = denoised_image.shape
    optimal_height = 1000  
    aspect_ratio = width / height
    optimal_width = int(optimal_height * aspect_ratio)
    resized_image = cv2.resize(denoised_image, (optimal_width, optimal_height))

   
    pil_image = Image.fromarray(resized_image)
    byte_io = io.BytesIO()
    pil_image.save(byte_io, format='PNG')

    byte_io.seek(0)
    image_for_ocr = Image.open(byte_io)

   
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(image_for_ocr, config=custom_config)

    return text
