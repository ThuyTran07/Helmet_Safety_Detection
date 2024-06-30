import os
import base64
import streamlit as st
from pathlib import Path
from ultralytics import YOLOv10
from components.streamlit_footer import footer


@st.cache_data(max_entries=1000)
def my_prediction(image_path):
    TRAINED_MODEL_PATH = 'model/best.pt'    
    model = YOLOv10(TRAINED_MODEL_PATH)
    IMG_SIZE = 640
    CONF_THRESHOLD = 0.3
    results = model.predict(source=image_path,
                       imgsz=IMG_SIZE,
                       conf=CONF_THRESHOLD)
    
    annotated_img = results[0].plot()
    annotated_img = annotated_img[..., ::-1] #convert bgr to rgb

    st.markdown('**Detection result**')    
    st.image(annotated_img)

def main():
    
    st.set_page_config(
        page_title="Helmet Safety Detection",
        page_icon='static/favicon.png',
        layout="wide"
    )

    logo_img = open("static/logo.png", "rb").read()
    logo_base64 = base64.b64encode(logo_img).decode()
    st.markdown(
        f"""
        <img src="data:image/png;base64,{logo_base64}" width="150px">
        """,
        unsafe_allow_html=True,
        )
    
    st.markdown('<br></br>', unsafe_allow_html=True)
    st.title(':sparkles: :green[YOLOv10] Helmet Safety Detection Demo')
    
    st.markdown('<br><br>', unsafe_allow_html=True)
    st.markdown('**Upload your image below to detect**')
    uploaded_img = st.file_uploader('', type=['jpg', 'jpeg', 'png'])
    st.markdown('<br><br>', unsafe_allow_html=True)
    st.markdown('**Or try an example here**')
    st.markdown('*You should try with a real photo instead. The example is an animation for illustration purposes.')
    example_button = st.button('Run example')

    st.divider()

    if example_button:
        # process_and_display_image('static/example_img.jpg')
        my_prediction('static/example2.png')

    if uploaded_img is not None:
        save_dir = "images"
        saved_path = Path(save_dir, uploaded_img.name)

        with saved_path.open("wb") as f:
            f.write(uploaded_img.getvalue())
        try:
            my_prediction(saved_path)
        finally:
            os.remove(saved_path)

    footer()


if __name__ == '__main__':
    main()