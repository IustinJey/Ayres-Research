from ultralytics import YOLO
import cv2

#pre-trained
model = YOLO('yolov8n.pt')

if __name__ == '__main__':

    cap = cv2.VideoCapture("[appsgolem.com][00-08-07][00-10-16]_Driving_Downtown_-_New_York_Ci.mp4")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_path = R"C:/Users/rusva/Desktop/out_video.mp4"

    fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    ret = True
    while ret:
        ret, frame = cap.read()
        results = model(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                label = box.cls[0]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label_text = f'{model.names[int(label)]} {confidence:.2f}'
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)
