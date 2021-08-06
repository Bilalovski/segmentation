import json

import onnxruntime
import paho.mqtt.client as paho
import numpy as np


def on_message(clientName, userdata, message):
    print("message received")
    data = json.loads(message.payload.decode('utf-8'))
    choice = data['choice']
    if choice ==1:
        session = onnxruntime.InferenceSession("MaskRCNN-10.onnx")
        img_data = np.array(data['data']).astype('float32')
        print(img_data.shape)
        input_name = session.get_inputs()[0].name
        result = session.run(None, {input_name: np.array(img_data)})
        data1 = result[0].tolist()
        data2 = result[1].tolist()
        data3 = result[2].tolist()
        data4 = result[3].tolist()

        data = {'choice':choice, 'data1': data1, 'data2': data2, 'data3': data3, 'data4': data4}
        payload = json.dumps(data)
        client.publish("seg_inference_out", payload)
        print("published data")
    # display_objdetect_image(Image.open("demo.jpg"), result[0], result[1], result[2])


def on_connect(mqtt_client, obj, flags, rc):
    if rc==0:
        client.subscribe("seg_preprocess_out", qos=0)
        print("connected")
    else:
        print("connection refused")



broker = "127.0.0.1"
client = paho.Client("seg_inference_node")
client.on_connect = on_connect
client.on_message = on_message
client.connect(broker)
client.loop_start()

while 1:
    pass