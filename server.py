import os
import socket
import struct
import threading
import datetime

import cv2
import numpy as np

# Logging setup


print('Server Started!')

# Server IP and port
server_ip = '0.0.0.0'#'192.168.1.4' #'192.168.1.129'
server_port = 2200 #4778      # server port is 4778
    
def handle_client(client_socket):
    def receive_image(sock):
        try:

            # Receive the size of the image
            header_data = sock.recv(12)
            if not header_data:
                return None

            # Unpack the image size from the received data
            total_length, img_size, name_size = struct.unpack(">LLL", header_data)
            # Receive the actual image bytes
            img_data = b''

            while len(img_data) < img_size:
                packet = sock.recv(img_size - len(img_data))
                if not packet:
                    return None
                img_data += packet

            name_data = b''
            while len(name_data) < name_size:
                packet = sock.recv(name_size - len(name_data))
                if not packet:
                    return None
                name_data += packet

            # Convert the bytes to a numpy array and decode the image
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            name = name_data.decode('utf-8')
            return img, name

        except Exception as e:
            print(f'Error receiving image: {e}')
            return None

    frame_count = 0

    try:
        while True:
            try:
                img , name = receive_image(client_socket)
                if img is None:
                    print('Failed to receive rear image.')
                    break
                cv2.imshow(f'Src: {name}', img)
                cv2.waitKey(1)
                frame_count += 1

                #save img in folder
                date = datetime.datetime.now()
                save_path = f'./images/{name}{date.strftime("%Y")}_{date.strftime("%b")}_{date.strftime("%d")}_{date.strftime("%H")}_{date.strftime("%M")}_{date.strftime("%S")}.jpg' #_{date.strftime("%f")}
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                cv2.imwrite(save_path, img)
                print(f"Image saved successfully at {save_path}")

            except Exception as e:
                print(f'An error occurred during image processing or saving: {e}')
                break

    except Exception as e:
        print(f'An unexpected error occurred: {e}')

    finally:
        try:
            client_socket.close()
            print('Client socket closed.')
        except Exception as e:
            print(f'An error occurred during cleanup: {e}')


# Create a socket object
try:
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((server_ip, server_port))
    server_socket.listen(5)
    print(f'Server listening on {server_ip}:{server_port}')
except Exception as e:
    print(f'Error setting up server: {e}')
    raise SystemExit("Exiting due to server setup failure.")

try:
    while True:
        try:
            # Accept a client connection
            client_socket, client_address = server_socket.accept()
            print(f'Connection from {client_address} has been established.')

            # Start a new thread for handling the client
            client_thread = threading.Thread(target=handle_client, args=(client_socket,))
            client_thread.start()

        except Exception as e:
            print(f'Error accepting client connection: {e}')

finally:
    try:
        server_socket.close()
        print('Server socket closed.')
    except Exception as e:
        print(f'An error occurred during cleanup: {e}')
