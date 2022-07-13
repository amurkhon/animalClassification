from fastai.vision.all import *
import streamlit as st
import plotly.express as px
import pathlib
plt = platform.system()
if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath

#title
st.title("Fauna Classification Model")
st.markdown("This model is not for domestic animals. Furthermore, 'Mammal' class means only human.") or st.camera_input("If you want to use camera, you can take a picture!")

#Rasm joylash
file = st.file_uploader("Load image", type=['png','jpg','gif','svg'])
if file:
    st.image(file)

    #PIL Convert
    img=PILImage.create(file)

    #model
    model = load_learner('animal_model.pkl')

    #prediction
    pred, pred_id, probs = model.predict(img)
    st.success(f"Prediction: {pred}")
    st.info(f"Probability: {probs[pred_id]*100:.1f}%")

    #plotting
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)
