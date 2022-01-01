import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from PIL import Image
import cv2




def main():
    st.title("FindMe")
    
    st.write('''
    *Find look-alike objects from a big picture*
    ''')

    image_src = st.file_uploader("Big picture", type='png')
    if image_src is not None:
        img_rgb = Image.open(image_src)
        st.image(img_rgb, caption='Uploaded Image.', use_column_width=True)
        template_src = st.file_uploader("What to find", type='png')
        if template_src is not None:
            template = Image.open(template_src)
            st.image(template, caption='Uploaded Image.', use_column_width=False)
    
    try:
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    except:
        img_gray = data.coins()
        template = img_gray[170:220, 75:130]

    height, width = template.shape[::]
    
    result = cv2.matchTemplate(img_gray, template, cv2.TM_SQDIFF)
    plt.imshow(result, cmap='gray')

    threshold = 0.5 #For TM_CCOEFF_NORMED, larger values = good fit.

    loc = np.where( res >= threshold)  

    for pt in zip(loc[::-1]): 
        cv2.rectangle(img_gray, pt, (pt[0] + width, pt[1] + height), (255, 0, 0), 1)
    plt.imshow(img_gray)


    
    

if __name__ == "__main__":
    main()

# References:
#https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_template.html#sphx-glr-auto-examples-features-detection-plot-template-py
#https://towardsdatascience.com/object-detection-on-python-using-template-matching-ab4243a0ca62