# Imports
from helper import *

# importer detector
import tensorflow_hub as hub

# building flask
from flask import Flask
from flask import request
import json

def update_camera():
    """
    get all the latest traffic camera photos
    """

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

    traffic_cameras = {}
    for url in urls:
        traffic_cameras.update(get_url(url))
    return traffic_cameras

def run_detector_car(detector, img_link, plot = False):
    """
    detect the number of car in an given image url
    """
    img = get_img(img_link)
    img_np = np.array(img)
    converted_img = tf.convert_to_tensor(img_np)
    converted_img = tf.image.convert_image_dtype(converted_img, tf.float32)[tf.newaxis, ...]
    start_time = time.time()
    result = detector(converted_img)
    end_time = time.time()

    result = {key: value.numpy() for key, value in result.items()}

    car_index = (np.isin(np.array(result["detection_class_entities"], dtype='str'),
                         ['Car', 'Vehicle', 'Land vehicle', 'Motorcycle'])) & (  # identify select types
                            result["detection_scores"] >= 0.1)  # min score

    # if printing the results on screen
    # print("Found %d cars." % len(result["detection_scores"][car_index]))
    # print("Inference time: ", end_time-start_time)

    if plot:
        title = 'Found {} cars.\nInference time: {}'.format(len(result["detection_scores"][car_index]),
                                                            end_time - start_time)

        image_with_boxes = draw_boxes(
            np.array(img_np, np.int32), result["detection_boxes"][car_index],
            result["detection_class_entities"][car_index], result["detection_scores"][car_index],
            title
        )

        # if to display the image
        # display_image(image_with_boxes)
        # if to return image as well
        return {'num_cars': len(result["detection_scores"][car_index]), 'time_taken': end_time - start_time,
                'img': image_with_boxes}
    else:
        return {'num_cars': len(result["detection_scores"][car_index]), 'time_taken': end_time - start_time}

# download model from tfhub
# two models: mobilenet which is fast but less accurate,
# inception resnet which is bigger model but more accurate
# we import mobilenet here for faster loading time
module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"  #@param ["https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1", "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"]
detector_model = hub.load(module_handle)
detector = detector_model.signatures['default']

# update all the camera image
traffic_cameras = update_camera()

# initiate flask
app = Flask(__name__)

@app.route('/')
def welcome():
    return """
    usage: select a traffic camera and model will return the number of cars
    """

@app.route('/detect', methods=['POST'])
def detect():
    cam = request.form['cam']
    cam_link = traffic_cameras[cam]
    result = run_detector_car(detector, cam_link)
    return json.dumps(result)

@app.route('/cameras', methods=['GET'])
def get_cameras():
    return json.dumps(list(traffic_cameras.keys()))

@app.route('/refresh', methods=['POST'])
def refresh():
    global traffic_cameras  # update global variable
    traffic_cameras = update_camera()
    return 'list of camera refreshed'

# run a simple test
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5080)