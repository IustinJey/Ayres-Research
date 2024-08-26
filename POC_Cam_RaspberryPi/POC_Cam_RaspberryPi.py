import socket
import struct
import threading
import cv2


# returns frame from camera
def capture_frame(cam):
    res, frame = cam.read()
    return frame if res else None


# shows frame to desktop (optional)
def show_frame(name, frame):
    cv2.imshow(f"Source: {name}", frame)
    cv2.waitKey(1)


# creates data packet from given image frame
def create_frame_packet(frame, name: str):
    _, img = cv2.imencode('.jpg', frame)
    img_bytes = img.tobytes()
    encoded_name = name.encode('utf-8')

    img_length = len(img_bytes)
    name_length = len(encoded_name)

    header = struct.pack(">LLL", img_length + name_length + 8, img_length, name_length)
    return header + img_bytes + encoded_name


def capture_loop(cam, name, socket):
    while True:
        frame = capture_frame(cam)
        if frame is not None:
            show_frame(name, frame)
            socket.sendall(create_frame_packet(frame, name))


if __name__ == '__main__':
    ip, port = '127.0.0.1', 2220
    cams = [('Front', cv2.VideoCapture(0)), ('Rear', cv2.VideoCapture(1))]
    socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket.connect((ip, port))
    for name, cam in cams:
        thread = threading.Thread(target=capture_loop, args=(cam, name, socket))
        thread.start()
        