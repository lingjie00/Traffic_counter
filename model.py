# Imports
from helper import *

# for transfer learning
import tensorflow_hub as hub

# download model from tfhub
# two models: mobilenet which is fast but less accurate,
# inception resnet which is bigger model but more accurate
# we import mobilenet here due to local computational limitation
module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"  #@param ["https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1", "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"]
detector_model = hub.load(module_handle)
detector = detector_model.signatures['default']

# update all the camera image
traffic_cameras = update_camera()

tuas = run_detector_car(detector, traffic_cameras['View from Second Link at Tuas'])

# build Flask
# https://flask.palletsprojects.com/en/1.1.x/quickstart/#a-minimal-application