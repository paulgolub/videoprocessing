## GNU Affero General Public License v3.0
##
## Python 3.10

import cv2
import asyncio
import multiprocessing
import time
import socket
import websockets
import mediapipe as mp
import json
import jwt
from datetime import datetime, timedelta
import math

fingerDif = 8
coordPoiList = [[50, 150], [150, 150], [200, 200],[300,100]]
radius = 20

port = "8765"

class VideoStream:
    def __init__(self, width=1280, height=720):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.mp_hands = mp.solutions.hands.Hands()

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        self.cap.release()

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, 
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon, 
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):    
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

        return lmList, bbox

def video_process(shared_list, lock):
    video_stream = VideoStream()
    pTime = 0
    fingerPoint = [4, 8, 12, 16, 20]
    detector = HandDetector()

    while True:
        frame = video_stream.read_frame()
        if frame is None:
            continue

        frame = detector.findHands(frame)
        lmList, bbox = detector.findPosition(frame)
        if len(lmList) != 0:
            txDateList = []
            for i in detector.tipIds:
                txDateList.append(lmList[i])

            global_data = [item for sublist in txDateList for item in sublist]

            with lock:
                #shared_list[:] = global_data[:]
                shared_list[:] = []
                if len(lmList) != 0:
                    global_data = []
                    for i in fingerPoint:
                        global_data.extend(lmList[i])
                    shared_list[:] = global_data

            print(f'Shared list from OpenCV: {shared_list[:]}')
            print(f'Shared list from OpenCV: {txDateList[2]}')
            poiIndex = poiDetector(shared_list[:], fingerDif)
            print(f'poiIndex: {poiIndex}')

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        xyDetection = poi_parser()
        try:
            next(xyDetection)

        except StopIteration:
            print("The end ;)")

        ###
        # for i, finger in enumerate(fingerPoint):
        #     x = 100
        #     y = 200
        #     i = 1
        #     cv2.putText(frame, f"Finger {i+1}: ({x}, {y})", (70, 30 + 30*i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        ###
        conturePoiDrow(frame)

        cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_stream.release()
    cv2.destroyAllWindows()

def conturePoiDrow(frame):

    gen = poi_parser()
    for index, poi in enumerate(gen):
        centerCoordinates = (poi[0], poi[1])
        color = (255, 120, 60) #BGR
        thickness = 3 # in px
        cv2.putText(frame, f"{index}", (poi[0]-10, poi[1]+9), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.circle(frame, centerCoordinates, radius, color, thickness)
        #cv2.rectangle(frame,(300,200),(350,400),(0,255,0),1)

    return frame

def poiDetector(txDateList, fingerDif):
    if (fingerDif == 4):
        x, y = txDateList[1], txDateList[2]
        fingerStr = "police"
        print(f"The finger {fingerStr} are selected")
    elif (fingerDif == 8):
        x = txDateList[4]
        y = txDateList[5]
        fingerStr = "indice"
        print(f"The finger {fingerStr} is selected")
    elif (fingerDif == 12):
        x, y = txDateList[7], txDateList[8]
        fingerStr = "medio"
        print(f"The finger {fingerStr} is selected")
    elif (fingerDif == 16):
        x, y = txDateList[10], txDateList[11]
        fingerStr = "anulare"
        print(f"The finger {fingerStr} is selected")
    elif (fingerDif == 20):
        x, y = txDateList[13], txDateList[14]
        fingerStr = "mignolo"
        print(f"The finger {fingerStr} is selected")
    else:
        print("The finger is not selected")
    poiIndex = is_point_in_circle(radius, x, y)
    if poiIndex is not None:
        print(f"POI index: {poiIndex}")
    else:
        print("Non detected")

    return poiIndex

def is_point_in_circle(r, x, y):
    gen = poi_parser()
    for index, poi in enumerate(gen):
        cx = poi[0]
        cy = poi[1]
        distance = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        if distance <= r:
            return index
    return None

def poi_parser():
    for i in coordPoiList:
        yield i

async def handle_connection(websocket, path, shared_list, lock):
    while True:
        try:
            message = await websocket.recv()
            print(f"Received from client: {message}")

            with lock:
                shared_data = shared_list[:]

            if not shared_data:
                json_data = {
                    "PX": 0, "PY": 0,
                    "IX": 0, "IY": 0,
                    "MX": 0, "MY": 0,
                    "AX": 0, "AY": 0,
                    "MX": 0, "MY": 0
                }
                
            else:
                json_data = {
                    "PX": shared_data[1],
                    "PY": shared_data[2],
                    "IX": shared_data[4],
                    "IY": shared_data[5],
                    "MX": shared_data[7],
                    "MY": shared_data[8],
                    "AX": shared_data[10],
                    "AY": shared_data[11],
                    "MX": shared_data[13],
                    "MY": shared_data[14]
                }

            response = json_data

            await websocket.send(json.dumps(response))

            # chack the last send data
            if isinstance(json_data, dict):
                last_sent_data = json_data
            else:
                last_sent_data = "Non detected"
            
        except websockets.ConnectionClosed:
            print("Connection closed")
            break

        json_data = {"PX": 0, "PY": 0,
                     "IX": 0, "IY": 0,
                     "MX": 0, "MY": 0,
                     "AX": 0, "AY": 0,
                     "MX": 0, "MY": 0
                    }

def ws_server(shared_list, lock):
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"Host IP address: {local_ip}")
    # local_ip = "localhost"
    start_server = websockets.serve(lambda ws, path: handle_connection(ws, path, shared_list, lock), local_ip, port)
    asyncio.get_event_loop().run_until_complete(start_server)
    print(f"Server was started. ws://{local_ip}:{port}")
    asyncio.get_event_loop().run_forever()

if __name__ == "__main__":
    manager = multiprocessing.Manager()
    shared_list = manager.list()
    lock = manager.Lock()

    video_proc = multiprocessing.Process(target=video_process, args=(shared_list, lock))
    ws_srv = multiprocessing.Process(target=ws_server, args=(shared_list, lock))

    video_proc.start()
    ws_srv.start()

    video_proc.join()
    ws_srv.join()
