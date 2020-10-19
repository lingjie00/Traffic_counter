# Traffic_counter

Using object detection to count the number of vehicle on Singapore roads, based on LTA traffic camera data

Understanding traffic condition is useful for route planning and to avoid congested roads. Modern navigation platforms
such as Google maps shows the traffic condition using color codes: red means congested while green means fast lane.\
However, drivers sometimes hope to have a detailed understanding of the road conditions. For example, Singaporeans
often travel to Malaysia for weekend trip, but travelling in peak hours could take up to 3 hours just to clear the
immigration checkpoint. Therefore, it is important to know exactly the traffic condition and to avoid the peak hours.\
However, a simple red color code is not able to tell us the exact traffic condition.

This model aims to solve this problem by providing user the traffic camera photo and the number of cars on the road
in real time. This model could potentially by built in a pipeline to enhance the existing navigation apps. \
For example, Google maps might be able to show the number of cars along with the color codes and interested users
can view the traffic camera photo as well.

The current product will showcase this idea by launching a webpage where users can select any highway in Singapore to know the number 
of cars currently on the road.

I have included the first prototype (initial Jupyter notebook) in subtree: prototype\
where I played around with exporting every single traffic camera in Singapore in prototype. Sample output as shown.

 Sample output
 ![](https://github.com/lingjie00/Traffic_counter/blob/main/prototype/output/traffic_cam.jpg)
