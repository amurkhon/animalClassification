from fastai.vision.all import *
import streamlit as st
import plotly.express as px
import pathlib
plt = platform.system()
if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath

#title
st.title("Hayvonlarni klassifikatsiya qiluvchi model")

#Rasm joylash
file = st.file_uploader("Rasm yuklash", type=['png','jpg','gif','svg'])

if file:
    st.image(file)

    #PIL Convert
    img=PILImage.create(file)

    #model
    model = load_learner('animal_model.pkl')

    #prediction
    pred, pred_id, probs = model.predict(img)
    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")

    #plotting
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)
