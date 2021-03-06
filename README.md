# FindMe
**Hidden Object Finder**

Finds hidden objects in a big crowded picture based on a template object.
Note that the application uses opencv-python's matchTemplate module and currently works best on grayscale images and unrotated objects.

![Sample Image](/input/ShakeBreak.png)

***

### Step 1

Upload the big picture where we need to find the hidden objects

![Big Picture](/input/ShakeBreak_image.png)

### Step 2

Upload an image that looks similar to the hidden objects

![Hidden Object](/input/ShakeBreak_template2.png)

### Step 3

Select a value for the threshold

![Threshold](/input/Threshold.png)

***

### Result

![Object(s) Found](/input/ShakeBreak_result.png)

The results may not be very accurate when the hidden objects are of different sizes, colors, backgrounds, and rotations compared to the template image.
This application was deployed at [FindMe](https://share.streamlit.io/jonathanadamrico/findme/main/main.py) hosted by Streamlit.

### Thank you!





***





# Acknowledgments

* Object Detection on Python using Template Matching [blog](https://towardsdatascience.com/object-detection-on-python-using-template-matching-ab4243a0ca62) by Ravindu Senaratne 
* Shake Break visual puzzle by David Helton 




