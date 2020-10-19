# Imports
import numpy as np
from six import BytesIO

# for transfer learning
import tensorflow as tf

# plotting traffic photos
import matplotlib.pyplot as plt
from PIL import Image, ImageColor, ImageDraw, ImageFont

# measuring inference time
import time

# downloading traffic photos
import requests
from bs4 import BeautifulSoup

# User functions
def display_image(image):
    """
    takes in an image file and plots it
    """
    fig = plt.figure(figsize=(20, 15))  # create a figure
    plt.grid(False)  # switch off the grids
    plt.imshow(image)  # show the image
    return fig


def draw_title(image, font, title):
    """
    takes in an image, desired font and title text and
    plots the title to the image
    """
    # initiate
    im_width, im_height = image.size
    draw = ImageDraw.Draw(image)
    w, h = font.getsize(title)
    # draw background box
    draw.rectangle((
        10, im_height - 50,  # draw on x = 10, y = - 50
        10 + w, im_height - 50 + h * 2.5  # draw the box as long and as high as the title
    ), fill='black'
    )
    # draw title
    draw.multiline_text(
        (10, im_height - 50),  # position on x = 10, y = - 50
        title,
        font=font,
        fill=(255, 255, 255)
    )
    return None


def draw_bounding_box_on_image(
        image, ymin, xmin, ymax, xmax,
        color, font, display_str_list=()
):
    """
    add boxing box to an image
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    draw.line([
        (left, top), (left, bottom), (right, bottom), (right, top), (left, top)
    ], width=4, fill=color)

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height
        # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)],
                       fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                  display_str,
                  fill="black",
                  font=font)
        text_bottom -= text_height - 2 * margin
    return None


def draw_boxes(image, boxes, class_names, scores, title, max_boxes=10, min_score=0.1):
    """
    Overlay labeled boxes on an image with formatted scores and label names
    """
    colors = list(ImageColor.colormap.values())

    try:
        font = ImageFont.truetype("/Users/lingjie/Library/Fonts/JosefinSansLight-ZVEll.ttf", 20)
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                           int(100 * scores[i]))
            color = colors[hash(class_names[i]) % len(colors)]
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color, font, display_str_list=[display_str])
            np.copyto(image, np.array(image_pil))

    image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
    draw_title(image_pil, font, title)
    np.copyto(image, np.array(image_pil))

    return image


def get_url(url):
    """
    input an url from onemotoring.com.sg and extract the image urls
    """
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    title = soup.title.string
    links = soup.find_all('img')
    img_links = {}
    for link in links:
        if 'View from' in link.get('alt'):
            img_links[link.get('alt')] = 'http://' + link.get('src').replace('//', '')
    return img_links


def get_img(img_url):
    """
    input an image url and return the image file
    """
    img = requests.get(img_url, stream=True)
    image_data = BytesIO(img.content)
    pil_image = Image.open(image_data)
#     pil_image = ImageOps.fit(pil_image, (512, 512), Image.ANTIALIAS) # resize photo
    return pil_image


def run_detector_car(detector, img_link):
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

    title = 'Found {} cars.\nInference time: {}'.format(len(result["detection_scores"][car_index]),
                                                        end_time - start_time)

    image_with_boxes = draw_boxes(
        np.array(img_np, np.int32), result["detection_boxes"][car_index],
        result["detection_class_entities"][car_index], result["detection_scores"][car_index],
        title
    )

    # if to display the image
    # display_image(image_with_boxes)
    return {'num_cars': len(result["detection_scores"][car_index]), 'time_taken': end_time - start_time,
            'img': image_with_boxes}


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

