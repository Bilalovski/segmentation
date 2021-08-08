import time

from matplotlib import pyplot as plt
from pydust import core

from PIL import Image
import pickle
import json
display = False
def receive(arg):
    print("image received")
    global img
    img = pickle.loads(arg)
    global display
    display = True

def main():
    # initialises the core with the given block name and the directory where the modules are located (default "./modules")
    global display
    dust = core.Core("seg_sub", "./modules")

    # start a background thread responsible for tasks that shouls always be running in the same thread
    dust.cycle_forever()
    # load the core, this includes reading the libraries in the modules directory to check addons and transports are available
    dust.setup()
    # set the path to the configuration file
    dust.set_configuration_file("configuration.json")
    # connects all channels
    dust.connect()
    time.sleep(1)
    # add a message listener on the subscribe-tcp channel. The callback function takes a bytes-like object as argument containing the payload of the message
    dust.register_listener("check_output", receive)
    #dust.register_listener("subscribe-mqtt", lambda payload: print("Received payload with %d bytes" % len(payload)))

    while True:
        if display:
            print("will display image")
            _, ax = plt.subplots(1, figsize=(12, 9))
            ax.imshow(img)
            plt.show()
            display = False
        time.sleep(1)


if __name__ == "__main__":
    main()