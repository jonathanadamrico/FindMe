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
        template_src = st.file_uploader("Object to find", type='png')
        st.image(img_rgb, caption='Big Picture', use_column_width=True)
        st.write("***")
        
        if template_src is not None:
            template = Image.open(template_src)
            
            img_gray = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_BGR2GRAY)
            template = cv2.cvtColor(np.array(template), cv2.COLOR_BGR2GRAY)

            st.image(template, caption='Object to Find', use_column_width=False)
            height, width = template.shape[::]
            st.write("***")
    
            st.write("Finding objects...")
    
            result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

            threshold = 0.8 #For TM_CCOEFF_NORMED, larger values = good fit.

            loc = np.where( result >= threshold)  

            for pt in zip(*loc[::-1]): 
                top_left = pt
                bottom_right = (pt[0] + width, pt[1] + height)
                cv2.rectangle(img_gray, top_left, bottom_right, (0, 255, 0), 1)
                
            st.write("***")
            st.write("### Results")
            st.image(img_gray, caption='Objects found', use_column_width=True)

            st.write("### Thank you!")
    
    

if __name__ == "__main__":
    main()

# References:
#https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_template.html#sphx-glr-auto-examples-features-detection-plot-template-py
#https://towardsdatascience.com/object-detection-on-python-using-template-matching-ab4243a0ca62