import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import cv2


def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


def main():
    st.set_page_config(page_title='FindMe')
    
    st.title("FindMe")
    
    st.write('''
    **Hidden Object Finder**

    Finds hidden objects in a big crowded picture based on a template object.
    Note that the application uses opencv-python's matchTemplate module and currently works best on grayscale images and unrotated objects.
    ''')
    
    sample_image = Image.open('input/ShakeBreak.png')
    st.image(sample_image, caption='Sample Image', use_column_width=True)
    st.write("***")

    st.write("### Step 1")
    image_src = st.file_uploader("Upload the big picture where we need to find the hidden objects", type=['png','jpg','jpeg'])
    if image_src is not None:
        img_rgb = Image.open(image_src)
        st.image(img_rgb, caption='Big Picture', use_column_width=True)
        st.write("***")
        
        st.write("### Step 2")
        template_src = st.file_uploader("Upload an image that looks similar to the hidden objects", type=['png','jpg','jpeg'])
        if template_src is not None:
            template = Image.open(template_src)
            
            img_gray = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_BGR2GRAY)
            template = cv2.cvtColor(np.array(template), cv2.COLOR_BGR2GRAY)

            st.image(template, caption='Hidden Object', use_column_width=False)
            height, width = template.shape[::]
            st.write("***")
    
            st.write("### Step 3")
            threshold = st.slider('Select a value for the threshold', 0.0, 1.0, 0.5)
    
            st.write(f"Finding objects at **{threshold}** similarity threshold...")

            result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED) 
            loc = np.where(result >= threshold)
    
            find_count = len(loc[0])

            # We want a colored rectangle on top of the gray image 
            img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

            for pt in zip(*loc[::-1]): 
                top_left = pt
                bottom_right = (pt[0] + width, pt[1] + height)
                cv2.rectangle(img_gray, top_left, bottom_right, (0, 255, 0), 1)
                
            st.write("***")
            st.write("### Result")
            
            if find_count == 0:
                st.warning("No objects were found. Try decreasing the threshold to find objects.")
                if threshold == 0:
                    st.warning("Unfortunately, the algorithm is not good enough to find the hidden object(s).")
            elif find_count > 100:
                st.warning("You may try increasing the threshold to avoid too many false positives.")
            else:
                st.success("Object(s) found!")
                
            st.image(img_gray, use_column_width=True)

            

            st.write("***")
            st.write('''The results may not be very accurate when the hidden objects are of different sizes, colors, backgrounds,
            and rotations compared to the template image.
            You may visit the [github page](https://github.com/jonathanadamrico/FindMe) for the source codes, documentations, and references. 
        
            ''')
            st.write("### Thank you!")

    
    

if __name__ == "__main__":
    main()

# Reference:
#https://towardsdatascience.com/object-detection-on-python-using-template-matching-ab4243a0ca62