import numpy as np
import json

import paho.mqtt.client as paho
from PIL import Image

published=False

def on_connect(mqtt_client, obj, flags, rc):
    if rc==0:
        print("connected")
    else:
        print("connection refused")

def on_publish(client, userdata, mid):
    print("published data")
    global published
    published=True


def preprocess(image):
    # Resize
    ratio = 800.0 / min(image.size[0], image.size[1])
    image = image.resize((int(ratio * image.size[0]), int(ratio * image.size[1])), Image.BILINEAR)

    # Convert to BGR
    image = np.array(image)[:, :, [2, 1, 0]].astype('float32')

    # HWC -> CHW
    image = np.transpose(image, [2, 0, 1])

    # Normalize
    mean_vec = np.array([102.9801, 115.9465, 122.7717])
    for i in range(image.shape[0]):
        image[i, :, :] = image[i, :, :] - mean_vec[i]

    # Pad to be divisible of 32
    import math
    padded_h = int(math.ceil(image.shape[1] / 32) * 32)
    padded_w = int(math.ceil(image.shape[2] / 32) * 32)

    padded_image = np.zeros((3, padded_h, padded_w), dtype=np.float32)
    padded_image[:, :image.shape[1], :image.shape[2]] = image
    image = padded_image

    return image


broker = "127.0.0.1"
client = paho.Client("seg_preprocessor")
client.on_connect=on_connect
client.on_publish=on_publish
client.connect(broker)
choice = 1
client.loop_start()

if choice == 1:
    img_path = 'demo.jpg'

    img = preprocess(Image.open(img_path))
    print(img.shape)
    data = {'choice': choice,'data': img.tolist()}
    payload = json.dumps(data)
    client.publish("seg_preprocess_out", payload, qos=0)

while not published:
    pass