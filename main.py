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
    
    ***
    ''')

    image_src = st.file_uploader("Upload the big picture where we need to find the objects", type='png')
    if image_src is not None:
        img_rgb = Image.open(image_src)
        st.image(img_rgb, caption='Big Picture', use_column_width=True)
        st.write("***")
        
        template_src = st.file_uploader("Upload a template that looks similar to the objects that we need to find", type='png')
        if template_src is not None:
            template = Image.open(template_src)
            
            img_gray = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_BGR2GRAY)
            template = cv2.cvtColor(np.array(template), cv2.COLOR_BGR2GRAY)

            st.image(template, caption='Template Object', use_column_width=False)
            height, width = template.shape[::]
            st.write("***")
    
            threshold = st.slider('Select a value for the threshold', 0.0, 1.0, 0.5)
    
            st.write(f"Finding objects at {threshold} similarity threshold...")
    
            result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

            loc = np.where( result >= threshold)  

            for pt in zip(*loc[::-1]): 
                top_left = pt
                bottom_right = (pt[0] + width, pt[1] + height)
                cv2.rectangle(img_gray, top_left, bottom_right, (0, 255, 0), 1)
                
            st.write("***")
            st.write("### Result")
            st.image(img_gray, caption='Objects found', use_column_width=True)

            st.write("***")
            st.write('''As you can see, the results may not be accurate especially when the objects of interest are of different sizes.
            There is still a lot of work to be done but I hope this was somehow helpful. You may visit the github source page https://github.com/jonathanadamrico/FindMe 
            for the codes and references. 
            
            ### Thank you!
            ''')

    
    

if __name__ == "__main__":
    main()

# References:
#https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_template.html#sphx-glr-auto-examples-features-detection-plot-template-py
#https://towardsdatascience.com/object-detection-on-python-using-template-matching-ab4243a0ca62