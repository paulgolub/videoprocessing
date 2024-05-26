## GNU Affero General Public License v3.0
##
## Python 3.10
## pip install opencv-python-headless websockets mediapipe PyJWT aiohttp

import cv2                                  # pip install opencv-python-headless
import asyncio
import multiprocessing
import time
import socket
import websockets                           # pip install websockets
import mediapipe as mp                      #pip install mediapipe
import json
import jwt
from datetime import datetime, timedelta
import math
import jwt                                  # pip install PyJWT
from urllib.parse import urlparse, parse_qs
from aiohttp import web                     # pip install aiohttp
import ctypes

fingerDif = 8
fingerDef = 8
coordPoiList = [[50, 150], [150, 150], [200, 200],[500,200]]
radius = 50
SECRET_KEY = 'secret-key'
TOKEN_EXPIRATION = 3600

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
        
    def findHands(self, img, hand_detected_flag, hand_detected_lock, draw=True):    
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
            with hand_detected_lock:
                hand_detected_flag.value = 1
        else:
            with hand_detected_lock:
                hand_detected_flag.value = 0
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

def video_process(shared_list, lock, finger_def, poi_index, finger_def_lock, poi_index_lock, hand_detected_flag, hand_detected_lock):
    video_stream = VideoStream()
    pTime = 0
    fingerPoint = [4, 8, 12, 16, 20]
    detector = HandDetector()
    while True:
        frame = video_stream.read_frame()
        if frame is None:
            continue
        frame = detector.findHands(frame, hand_detected_flag, hand_detected_lock)
        lmList, bbox = detector.findPosition(frame)
        if len(lmList) != 0:
            txDateList = []
            for i in detector.tipIds:
                txDateList.append(lmList[i])
            global_data = [item for sublist in txDateList for item in sublist]
            with finger_def_lock:
                fingerDif = finger_def.value
            print(f'Finger Dif{fingerDif}')
            with lock:  # Защищаем доступ к разделяемому списку
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
            print(f'Finger: {fingerDif}')
            with poi_index_lock:
                poi_index.value = poiIndex
            time.sleep(0.03)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        xyDetection = poi_parser()
        try:
            next(xyDetection)
        except StopIteration:
            print("The end ;)")
        conturePoiDrow(frame)
        cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_stream.release()
    cv2.destroyAllWindows()

def conturePoiDrow(frame):
    gen = poi_parser()
    # Draw a circle in the frame
    for index, poi in enumerate(gen):
        centerCoordinates = (poi[0], poi[1])
        color = (255, 120, 60) #BGR
        thickness = 3 # in px
        cv2.putText(frame, f"{index}", (poi[0]-10, poi[1]+9), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.circle(frame, centerCoordinates, radius, color, thickness)
        #cv2.rectangle(frame,(300,200),(350,400),(0,255,0),1)
    return frame

def poiDetector(txDateList, fingerDif):
    #               0   1    2   3   4    5   6    7    8    9   10   11  12   13   14
    # txDateList = [4, 327, 236, 8, 291, 210, 12, 308, 228, 16, 310, 243, 20, 305, 258]
    if (fingerDif == 4): # 4, 8, 12, 16, 20
        x, y = txDateList[1], txDateList[2]
        fingerStr = "police"
        print(f"The finger {fingerStr} are selected. {fingerDif}")
    elif (fingerDif == 8):
        x = txDateList[4]
        y = txDateList[5]
        fingerStr = "indice"
        print(f"The finger {fingerStr} is selected. {fingerDif}")
    elif (fingerDif == 12):
        x = txDateList[7]
        x = txDateList[8]
        fingerStr = "medio"
        print(f"The finger {fingerStr} is selected. {fingerDif}")
    elif (fingerDif == 16):
        x = txDateList[10]
        x = txDateList[11]
        fingerStr = "anulare"
        print(f"The finger {fingerStr} is selected. {fingerDif}")
    elif (fingerDif == 20):
        x = txDateList[13]
        y = txDateList[14]
        fingerStr = "mignolo"
        print(f"The finger {fingerStr} is selected. {fingerDif}")
    else:
        print("The finger is not selected.")
    # print(f'Finger: {fingerDif}')
    #print(f'Type of txDataList: {type(txDateList)}')
    #print(f'txDataList[1]: {txDateList[1]}')
    # is_point_in_circle(cx, cy, r, x, y):
    # cx, cy - центр круга
    # r - радиус
    # x, y - координаты точки
    poiIndex = is_point_in_circle(radius, x, y)
    if poiIndex != 999:
        print(f"Попало в круг с индексом: {poiIndex}")
    else:
        print("Не попало ни в один круг")
    return poiIndex

def is_point_in_circle(r, x, y):
    gen = poi_parser()
    for index, poi in enumerate(gen):
        cx = poi[0]
        cy = poi[1]
        distance = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        if distance <= r:
            return index
    return 999
    
def poi_parser():
    for i in coordPoiList:
        yield i
async def handle_connection(websocket, path, shared_list, lock, finger_def, poi_index, finger_def_lock, poi_index_lock, hand_detected_flag, hand_detected_lock):
    # close_connection = asyncio.Future()
    # async def timeout_handler():
    #     await asyncio.sleep(30)  # Таймаут 30 секунд
    #     close_connection.set_result(True)
    # asyncio.create_task(timeout_handler())
    # async for message in websocket:
    #     print(f"Received message: {message}")
    #     if message == "close":
    #         print("Closing connection by client request")
    #         break
    #     while True:
    #         try:
    #             if close_connection.done():
    #                 print("Closing connection by timeout")
    #                 break
    #             await websocket.send("This is a cyclic message.")
    #             await asyncio.sleep(1)  # sending message every second
    #         except websockets.ConnectionClosed:
    #             print("Connection closed")
    #             break
    #     await websocket.close()
    print(f'Inside handle_connection')
    try:
        query = urlparse(path).query
        params = parse_qs(query)
        token = params.get('token', [None])[0]
        with finger_def_lock:
            fingerDif = finger_def.value
        print(f'Query {query}')
        print(f'Params {params}')
        print(f'Token {token}')
        if token is None:
            await websocket.close(code=4001, reason='Unauthorized')
            return
        try:
            decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            print(f"Token valid, user: {decoded['username']}")
        except jwt.ExpiredSignatureError:
            await websocket.close(code=4001, reason='Token expired')
            return
        except jwt.InvalidTokenError:
            await websocket.close(code=4001, reason='Invalid token')
            return
        message = await websocket.recv()
        print(f"Received from client: {message}")
        while True:
            if message != None:
                with lock:
                    shared_data = shared_list[:]
                with poi_index_lock:
                    poiIndex = poi_index.value
                with hand_detected_lock:
                    hand_detected_flag = hand_detected_flag.value
                print(f'POI index (ws): {poiIndex}')
                print(f'Finger (ws): {fingerDif}')
                if hand_detected_flag == 1:
                    if poiIndex != 999:
                        json_data = {
                            "POI index": poiIndex,
                        }
                    else:
                        json_data = {
                            "POI index": "None",
                        }
                else:
                    json_data = {
                            "POI index": "Hand not detected",
                        }
                response = json_data
                await websocket.send(json.dumps(response))
            else:
                await websocket.send(f"Echo: {message}")
            await asyncio.sleep(1)
    except websockets.ConnectionClosed:
        print(f'Connection closed.')
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if not websocket.closed:
            await websocket.close(code=1000, reason="Session ended")
            print("WebSocket connection closed")
async def generate_token(request):
    data = await request.json()
    username = data.get('username')
    finger_def = data.get('finger')
    if not finger_def:
        return web.json_response({'error': 'Finger index is required'}, status=400)
    elif(finger_def != 8 and finger_def != 12):
        return web.json_response({'error': 'Invalid finger index'}, status=400)
    else:
        with finger_def_lock:
            finger_def.value = finger_def
    
    if not username:
        return web.json_response({'error': 'Username is required'}, status=400)

    payload = {
        'username': username,
        'exp': int(time.time()) + TOKEN_EXPIRATION
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')
    return web.json_response({'token': token})

def ws_server(shared_list, lock, finger_def, poi_index, finger_def_lock, poi_index_lock, hand_detected_flag, hand_detected_lock):

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"Host IP address: {local_ip}")

    local_ip = 'localhost'

    porthttp = "8088"
    app = web.Application()
    app.router.add_post('/token', generate_token)
    runner = web.AppRunner(app)
    asyncio.get_event_loop().run_until_complete(runner.setup())
    site = web.TCPSite(runner, local_ip, porthttp)
    asyncio.get_event_loop().run_until_complete(site.start())
    print(f"HTTP server started at http://{local_ip}:{porthttp}")
    with finger_def_lock:
        finger_def.value = 8
    portws = "8765"
    start_server = websockets.serve(lambda ws, path: handle_connection(ws, path, shared_list, lock, finger_def, poi_index, finger_def_lock, poi_index_lock, hand_detected_flag, hand_detected_lock), local_ip, portws)
    asyncio.get_event_loop().run_until_complete(start_server)
    print(f"WS server was started. ws://{local_ip}:{portws}")
    asyncio.get_event_loop().run_forever()

if __name__ == "__main__":
    manager = multiprocessing.Manager()

    shared_list = manager.list()
    finger_def = multiprocessing.Value('i', 0)
    poi_index = multiprocessing.Value('i', 0)
    hand_detected_flag = multiprocessing.Value('i', 0)
    hared_string = multiprocessing.Array(ctypes.c_char, 128)
        
    lock = manager.Lock()
    finger_def_lock = multiprocessing.Lock()
    poi_index_lock = multiprocessing.Lock()
    hand_detected_lock = multiprocessing.Lock()

    video_proc = multiprocessing.Process(target=video_process, args=(shared_list, lock, finger_def, poi_index, finger_def_lock, poi_index_lock, hand_detected_flag, hand_detected_lock))
    ws_srv = multiprocessing.Process(target=ws_server, args=(shared_list, lock, finger_def, poi_index, finger_def_lock, poi_index_lock, hand_detected_flag, hand_detected_lock))
    

    video_proc.start()
    ws_srv.start()

    video_proc.join()
    ws_srv.join()
