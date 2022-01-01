import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
import cv2
from tempfile import NamedTemporaryFile


if __name__ == "__main__":

    st.title("FindMe")
    
    st.write('''
    #**Find look-alike objects**
    ''')

    #st.set_option('deprecation.showfileUploaderEncoding', False)

    #buffer = st.file_uploader("Choose an image...", type='png')
    #temp_file = NamedTemporaryFile(delete=False)
    #if buffer:
    #    temp_file.write(buffer.getvalue())
    #    st.write(load_img(temp_file.name))
    
    try:
        image_src = r'input/image.png'
        template_src = r'input/template.png'
        img_rgb = cv2.imread(image_src)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        template = cv2.imread(template_src, 0)
    except:
        img_gray = data.coins()
        template = img_gray[170:220, 75:130]

    height, width = template.shape[::]
    
    res = cv2.matchTemplate(img_gray, template, cv2.TM_SQDIFF)
    plt.imshow(res, cmap='gray')

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    top_left = min_loc  #Change to max_loc for all except for TM_SQDIFF
    bottom_right = (top_left[0] + width, top_left[1] + height)
    cv2.rectangle(img_gray, top_left, bottom_right, (255, 0, 0), 2) 
    plt.imshow(img_gray)


# References:
#https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_template.html#sphx-glr-auto-examples-features-detection-plot-template-py
#https://towardsdatascience.com/object-detection-on-python-using-template-matching-ab4243a0ca62