import cv2
import easyocr

reader = easyocr.Reader(['en'])

def OCR_reader(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image_thresh = cv2.threshold(image_gray, 64, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow("Processed_license_plate", image_thresh)
    # cv2.waitKey(0)
    detections = reader.readtext(image_thresh)
    for detection in detections:
        _, text, score = detection
        text = text.upper().replace(' ', '')
        return text, score
    return "", 0
