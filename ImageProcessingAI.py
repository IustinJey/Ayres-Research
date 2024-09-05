import pytesseract
import cv2
from ultralytics import YOLO
import configparser
import torch

from DetectionAI import DetectionAI

CONFIDENCE_THRESHOLD = 0.5

class ImageProcessingAI:
    def __init__(self):
        config = configparser.ConfigParser()
        config.read('ConfigFile.properties')
        self.__vehicle_model = YOLO(config['Path']['vehicle_model'])
        self.__license_plate_model = YOLO(config['Path']['license_plate_model'])
        self.__orientation_model = torch.hub.load('yolov5', 'custom', path=config['Path']['orientation_model'], source='local')
        pytesseract.pytesseract.tesseract_cmd = config['Path']['OCR_reader']

    def process_image(self, image):
        """
        Takes an image and performs object detection, detecting vehicles and license plates,
        and performs OCR recognition on license plates.
        :param image: numpy array
        :return: DetectionAI class containing necessary information
        """

        vehicle_labels = {
            2: 'car',
            3: 'car',
            5: 'car',
            7: 'car'
        }

        orientation_labels = {
            0: 'rear',
            10: 'front',
        }

        vehicle_detector = self.__vehicle_model(image)[0]
        x1, y1, x2, y2, confidence, class_id = self.__reduce(vehicle_detector.boxes.data.tolist())
        result = DetectionAI()
        if int(class_id) in vehicle_labels:
            if confidence > CONFIDENCE_THRESHOLD:
                if (vehicle_labels.get(int(class_id), 'unknown') == 'car'):
                    result.set_car()  #TODO for truck and emergency vehicles
                cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
                #cv2.imshow('Detection', cropped_image)
                # cv2.waitKey(0)
                orientation_detector = self.__orientation_model(cropped_image)
                orientation_detector = orientation_detector.xyxy[0]
                _, _, _, _, confidence, class_id = self.__reduce(orientation_detector)
                if (confidence > CONFIDENCE_THRESHOLD):
                    if (orientation_labels.get(int(class_id), 'unknown') == 'front'):
                        result.set_front()
                    elif (orientation_labels.get(int(class_id), 'unknown') == 'rear'):
                        result.set_rear()
                license_plate_detections = self.__license_plate_model(cropped_image)[0]
                license_plate_detection_list = license_plate_detections.boxes.data.tolist()
                x1_plate, y1_plate, x2_plate, y2_plate, confidence, _ = self.__reduce(license_plate_detection_list)
                if (confidence > CONFIDENCE_THRESHOLD):
                    cropped_plate = image[int(y1_plate)+int(y1):int(y2_plate)+int(y1), int(x1_plate)+int(x1):int(x2_plate)+int(x1)]
                    height, width, _ = cropped_plate.shape
                    cropped_plate = cv2.resize(cropped_plate, (4*width, 4*height))
                    plate_string = self.__OCR_processing(cropped_plate, r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                    result.set_license_plate(plate_string)
        return result



    def __OCR_processing(self, image, custom_config):
        """
        performs OCR recognition on image
        :param image: numpy array
        :return: string
        """
        #TODO

        return pytesseract.image_to_string(image, config=custom_config)

    def __reduce(self, list):
        """
        Reduces the list of detections to only one detection having best confidence threshold
        :param list: list
        :return: tuple
        """

        if len(list) == 0:
            return (0,0,0,0,0,0)
        x1, y1, x2, y2, confidence, class_id = list[0]
        for arg in list[1:]:
            l_x1, l_y1, l_x2, l_y2, l_confidence, l_class_id = arg
            if l_confidence > confidence:
                x1, y1, x2, y2, confidence, class_id = l_x1, l_y1, l_x2, l_y2, l_confidence, l_class_id
        return (x1, y1, x2, y2, confidence, class_id)