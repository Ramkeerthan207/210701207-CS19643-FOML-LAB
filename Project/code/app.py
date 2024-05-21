import streamlit as st
import base64
from PIL import Image
import numpy as np
import cv2
from model import model

st.markdown('<h1 style="color:black;">Animal Species Classification</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="color:gray;">The image classification model classifies image into following categories:</h2>', unsafe_allow_html=True)
st.markdown('<h3 style="color:gray;">Animal Names will be Entered Soon</h3>', unsafe_allow_html=True)



@st.cache_resource()
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file) 
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: scroll; # doesn't work
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('bg.webp')


l = ['dog','horse','elephant','butterfly','frog','cat','cow','sheep','spider','squirell']

upload= st.file_uploader('Insert image for classification', type=['png','jpg'])
c1, c2= st.columns(2)
if upload is not None:
  im= Image.open(upload)
  img= np.asarray(im)
  image= cv2.resize(img,(256, 256))
  image = image/255
  img= np.expand_dims(image, 0)
  c1.header('Input Image')
  c1.image(im)
  c1.write(img.shape)
  input_shape = (256, 256, 3)
  n_classes=10
  res_model = model()

  res_preds = res_model.predict(img)
  res_pred_classes = np.argmax(res_preds, axis=1)
  c2.header('Output')
  c2.subheader('Predicted class :')
  c2.write(l[res_pred_classes[0]] )
