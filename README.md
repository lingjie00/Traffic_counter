# Traffic_counter

Using object detection to count the number of vehicle on Singapore roads, based on LTA traffic camera data

Understanding traffic condition is useful for route planning and to avoid congested roads. Modern navigation platforms
such as Google maps shows the traffic condition using color codes: red means congested while green means fast lane.\
However, drivers sometimes hope to have a detailed understanding of the road conditions. For example, Singaporeans
often travel to Malaysia for weekend trip, but travelling in peak hours could take up to 3 hours just to clear the
immigration checkpoint. Therefore, it is important to know exactly the traffic condition and to avoid the peak hours.\
However, a simple red color code is not able to tell us the exact traffic condition.

This model aims to solve this problem by providing user the number of cars on the road
in real time. This model could potentially by built in a pipeline to enhance the existing navigation apps. \
For example, Google maps might be able to show the number of cars along with the color codes.

The current product is an API where users can POST request to retrieve the number of cars on the road in real time.

An illustration of the model with image output as shown.

A sample usage is found in ```sample.ipynb```

 Sample output
 ![](https://github.com/lingjie00/Traffic_counter/blob/main/traffic_cam.jpg)
