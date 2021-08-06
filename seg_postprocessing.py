import json
import time
import cv2
import paho.mqtt.client as paho
from matplotlib import pyplot as plt, patches
import numpy as np
from PIL import Image, ImageDraw, ImageColor
disp=False
classes = [line.rstrip('\n') for line in open('labels.txt')]


def display_objdetect_image(image, boxes, labels, scores, masks, score_threshold=0.7):
    # Resize boxes
    ratio = 800.0 / min(image.size[0], image.size[1])
    boxes /= ratio

    _, ax = plt.subplots(1, figsize=(12,9))
    image = np.array(image)

    for mask, box, label, score in zip(masks, boxes, labels, scores):
        # Showing boxes with score > 0.7
        if score <= score_threshold:
            continue

        # Finding contour based on mask
        mask = mask[0, :, :, None]
        int_box = [int(i) for i in box]
        mask = cv2.resize(mask, (int_box[2]-int_box[0]+1, int_box[3]-int_box[1]+1))
        mask = mask > 0.5
        im_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        x_0 = max(int_box[0], 0)
        x_1 = min(int_box[2] + 1, image.shape[1])
        y_0 = max(int_box[1], 0)
        y_1 = min(int_box[3] + 1, image.shape[0])
        mask_y_0 = max(y_0 - box[1], 0)
        mask_y_1 = mask_y_0 + y_1 - y_0
        mask_x_0 = max(x_0 - box[0], 0)
        mask_x_1 = mask_x_0 + x_1 - x_0
        im_mask[y_0:y_1, x_0:x_1] = mask[
            mask_y_0 : mask_y_1, mask_x_0 : mask_x_1
        ]
        im_mask = im_mask[:, :, None]

        # OpenCV version 4.x
        contours, hierarchy = cv2.findContours(
            im_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        image = cv2.drawContours(image, contours, -1, 25, 3)

        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='b', facecolor='none')
        ax.annotate(classes[label] + ':' + str(np.round(score, 2)), (box[0], box[1]), color='w', fontsize=12)
        ax.add_patch(rect)

    ax.imshow(image)
    plt.show()



def on_message(clientname, userdata, message):
    time.sleep(1)
    data = json.loads(message.payload.decode('utf-8'))
    global choice
    choice = data['choice']
    global boxes_seg, labels_seg, scores_seg, masks_seg, disp
    if choice == 1:
        boxes_seg = np.array(data['data1']).astype('float32')
        labels_seg= np.array(data['data2']).astype('int64')
        scores_seg= np.array(data['data3']).astype('float32')
        masks_seg= np.array(data['data4']).astype('float32')
        print(masks_seg.shape)
        disp=True


def on_connect(mqtt_client, obj, flags, rc):
    if rc==0:
        client.subscribe("seg_inference_out", qos=0)
        print("connected")
    else:
        print("connection refused")


broker = "127.0.0.1"
client = paho.Client("seg_postprocessor")
client.on_message=on_message
client.on_connect=on_connect
client.connect(broker)
client.loop_start()
while 1:
    if disp:
        if choice == 1:
            display_objdetect_image(Image.open("demo.jpg"), boxes_seg, labels_seg, scores_seg, masks_seg)
            disp=False
