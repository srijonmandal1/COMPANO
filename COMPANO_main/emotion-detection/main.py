import argparse
import time
import os
from pathlib import Path
import numpy as np

from datetime import datetime
import re

# from playsound import playsound
from pydub import AudioSegment
from pydub.playback import play

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from emotion import detect_emotion, init

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, save_one_box, create_folder
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

import pyttsx3
import micro_operator
import detect_word

from gtts import gTTS

from pymongo import MongoClient

from twilio.rest import Client

from dotenv import load_dotenv

load_dotenv()

device_ID = os.getenv('DEVICE_ID')

def speak_text(text_msg):
    tts = gTTS(text=text_msg, lang='en')
    tts.save("text.mp3")
    os.system("mpg321 text.mp3")
    os.remove("text.mp3")


account_sid = "ACdc3e995b4baec5d0ce6f902600088aac"
auth_token = "c949c558247574a69110fd49299028ed"
client = Client(account_sid, auth_token)

def MongoDB_user():
    client = MongoClient("mongodb+srv://Access1:passedaccess@mongoeval.2h7cybx.mongodb.net/?retryWrites=true&w=majority")
    db = client.get_database('COMPANO')
    user_info = db.userinfo
    return user_info


user_info = MongoDB_user()

def MongoDB_preference():
    client = MongoClient("mongodb+srv://Access1:passedaccess@mongoeval.2h7cybx.mongodb.net/?retryWrites=true&w=majority")
    db = client.get_database('COMPANO')
    user_preference = db.userpreference
    return user_preference


user_preference = MongoDB_preference()

associated_name = user_info.find_one({'device_id':device_ID})['name']

emergency_to_name = user_info.find_one({'device_id':device_ID})["emergency_contact"]

emergency_number = user_info.find_one({'device_id':device_ID})["phone_number"]

overall_preferences = user_preference.find()

label = ""

sad_mood1 = False
angry_mood1 = False
happy_mood1 = False

time1er = datetime.today().strftime("%H:%M %p")

hour = int(re.split('[-:]',time1er)[0])


if hour >= 0 and hour <= 12:
    txt_hr = f"Good morning {associated_name}! Have a fantastic day today."
    speak_text(txt_hr)
if hour > 12 and hour <= 14:
    txt_hr = f"Good afternoon {associated_name}! I hope you are having a wonderful day."
    speak_text(txt_hr)
if hour > 14 and hour <= 18:
    txt_hr = f"Good evening {associated_name}! I hope you are having a great day."
    speak_text(txt_hr)
if hour > 18:
    # Have a good night
    txt_hr = f"Hello {associated_name}. I hope your day went well!"
    speak_text(txt_hr)


# engine.runAndWait()

# del engine

def detect(opt):
    global label
    global happy_mood1
    global sad_mood1
    global angry_mood1
    global overall_preferences
    global account_sid
    global auth_token
    global client
    global associated_name
    global emergency_to_name
    global emergency_number
    global speak_text


    source, view_img, imgsz, nosave, show_conf, save_path, show_fps = opt.source, not opt.hide_img, opt.img_size, opt.no_save, not opt.hide_conf, opt.output_path, opt.show_fps
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    # Directories
    create_folder(save_path)

    # Initialize
    set_logging()
    device = select_device(opt.device)
    init(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load("weights/yolo.pt", map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = ((0,52,255),(121,3,195),(176,34,118),(87,217,255),(69,199,79),(233,219,155),(203,139,77),(214,246,255))

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                images = []
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = xyxy
                    images.append(im0.astype(np.uint8)[int(y1):int(y2), int(x1): int(x2)])
                
                if images:
                    emotions = detect_emotion(images,show_conf)
                # Write results
                i = 0
                for *xyxy, conf, cls in reversed(det):


                    if view_img or not nosave:  
                        # Add bbox to image with emotions on 
                        label = emotions[i][0]
                        colour = colors[emotions[i][1]]
                        i += 1
                        plot_one_box(xyxy, im0, label=label, color=colour, line_thickness=opt.line_thickness)

                        # Code to get emotion
                        x = str(label)
                        y = x.split( )[0]

                        # Text to Speech Code
                        # engine = pyttsx3.init()
                        # engine.setProperty('volume', 1)
                        # engine.setProperty('rate', 138)
                        # engine.setProperty('voice', 'english+f2') 

                        timetime1 = datetime.today().strftime("%H:%M %p")

                        exact_time = timetime1.split(" ")[0]

                        # TO DO - reminder few minutes earlier

                        for preference in overall_preferences:
                            if preference['time'] == exact_time:
                                print("entered condition to say reminder")
                                reminder_txt = f"Hello {associated_name}. A quick reminder: {preference['text']}"
                                speak_text(reminder_txt)


                        if y == "sad" and sad_mood1 != True:
                            emotion_txt = "It looks like you are not in a very good mood today. I would like to cheer you up. What can I do?"
                            speak_text(emotion_txt)
                            micro_operator.tester()
                            sad_mood1 = True
                            y = ""
                            time.sleep(3)

                        if y == "sad" and sad_mood1 == True:
                            emotion_txt = "It looks like you are not feeling well again. Should I play some music again?"
                            speak_text(emotion_txt)
                            micro_operator.tester()
                            sad_mood2 = True
                            y = ""
       
                        if y == "anger" and angry_mood1 != True:
                            # engine.say("Do you want to talk to your near one?")
                            emotion_txt = "Looks like something is bothering you today, so let me play some calm music."
                            speak_text(emotion_txt)
                            song = AudioSegment.from_wav("new_calmer.wav")
                            print('playing sound using pydub')
                            play(song)
                            emotion_txt = f"Lets relax a bit and have a chat with {emergency_to_name}."
                            speak_text(emotion_txt)

                            print(emergency_number)
                            # Now I need to add a recording where it captures their message and sends it over
                            call = client.calls.create(
                                url="http://demo.twilio.com/docs/voice.xml",
                                to=f"+1{emergency_number}",
                                from_="+18333270330"
                            )

                                # Add dynamic name
                                # message = client.messages.create(
                                #     body=f"Hello {emergency_to_name}.{associated_name} would like to talk to you.",
                                #     to=f"+1{emergency_number}",
                                #     from_="+18333270330"
                                # )

                            prompt_txt = "How else can I help you?"
                            speak_text(prompt_txt)
                            micro_operator.tester()
                            angry_mood1 = True
                            y = ""
                            time.sleep(3)
                        
                        if y == "anger" and angry_mood1 == True:
                            emotion_txt = "You look bothered again today. Let me help you."
                            speak_text(emotion_txt)
                            micro_operator.tester()
                            angry_mood2 = True
                            y = ""


                        if y == "happy" and happy_mood1 != True:
                            emotion_txt = "You look to be in a good mood. I am glad you are doing well."
                            speak_text(emotion_txt)
                            happy_mood1 = True
                            y = ""
                            time.sleep(3)


                        if y == "happy" and happy_mood1 == True:
                            emotion_txt = "It is great to see you doing well again. If you are feeling down, I am here for you."
                            speak_text(emotion_txt)
                            happy_mood2 = True
                            y = ""

            # Stream results
            if view_img:
                display_img = cv2.resize(im0, (im0.shape[1]*2,im0.shape[0]*2))
                resized_img = cv2.resize(display_img,(400,400),interpolation = cv2.INTER_AREA)
                # cv2.imshow("Emotion Detection",display_img)
                cv2.imshow("Emotion Detection",resized_img)
                cv2.waitKey(1)  # 1 millisecond
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            if not nosave:
                # check what the output format is
                ext = save_path.split(".")[-1]
                if ext in ["mp4","avi"]:
                    # Save results (image with detections)
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
                elif ext in ["bmp", "pbm", "pgm", "ppm", "sr", "ras", "jpeg", "jpg", "jpe", "jp2", "tiff", "tif", "png"]:
                    # save image
                    cv2.imwrite(save_path,im0)
                else:
                    # save to folder
                    output_path = os.path.join(save_path,os.path.split(path)[1])
                    create_folder(output_path)
                    cv2.imwrite(output_path,im0)

        if show_fps:
            # calculate and display fps
            print(f"FPS: {1/(time.time()-t0):.2f}"+" "*5,end="\r")
            t0 = time.time()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='face confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--hide-img', action='store_true', help='hide results')
    save = parser.add_mutually_exclusive_group()
    save.add_argument('--output-path', default="output.mp4", help='save location')
    save.add_argument('--no-save', action='store_true', help='do not save images/videos')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--show-fps', default=False, action='store_true', help='print fps to console')
    opt = parser.parse_args()
    check_requirements(exclude=('pycocotools', 'thop'))
    with torch.no_grad():
        detect(opt=opt)
