# Imports
import joblib
from helper import *

# for transfer learning
import tensorflow as tf
import tensorflow_hub as hub

# download model from tfhub
# two models: mobilenet which is fast but less accurate,
# inception resnet which is bigger model but more accurate
# we import mobilenet here due to local computational limitation
module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"  #@param ["https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1", "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"]
detector = hub.load(module_handle).signatures['default']
joblib.dump(detector, 'detector.pkl')

# the list of highways in Singapore
woodlands = 'https://www.onemotoring.com.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras/woodlands.html#trafficCameras'
kje = 'https://www.onemotoring.com.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras/kje.html#trafficCameras'
sle = 'https://www.onemotoring.com.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras/sle.html#trafficCameras'
tpe = 'https://www.onemotoring.com.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras/tpe.html#trafficCameras'
bke = 'https://www.onemotoring.com.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras/bke.html#trafficCameras'
aye = 'https://www.onemotoring.com.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras/aye.html#trafficCameras'
cte = 'https://www.onemotoring.com.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras/cte.html#trafficCameras'
mce = 'https://www.onemotoring.com.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras/mce.html#trafficCameras'
ecp = 'https://www.onemotoring.com.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras/ecp.html#trafficCameras'
pie = 'https://www.onemotoring.com.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras/pie.html#trafficCameras'
stg = 'https://www.onemotoring.com.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras/stg.html#trafficCameras'
urls = [woodlands, kje, sle, tpe, bke, aye, cte, mce, ecp, pie, stg]

# update all the camera image
traffic_cameras = update_camera(urls)