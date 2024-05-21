import streamlit as st
@st.cache_resource()
def model():
    from keras import Sequential  # For building sequential models
    from keras.models import load_model  # For loading pre-trained models
    from keras.layers import Dense, GlobalAvgPool2D as GAP, Dropout  # For defining model layers
    from tensorflow.keras.applications import InceptionV3, Xception, ResNet152V2  # For using pre-trained mo0dels
    num_classes = 10

    name = "ResNet152V2"
    base_model = ResNet152V2(include_top=False, input_shape=(256,256,3), weights='imagenet')
    base_model.trainable = False

    resnet152V2 = Sequential([
        base_model,
        GAP(),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ], name=name)

    resnet152V2.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    resnet152V2.load_weights('ResNet152V2.h5')

    return resnet152V2