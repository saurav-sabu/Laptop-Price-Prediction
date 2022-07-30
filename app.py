from json import load
import pickle
import streamlit as st
import numpy as np

st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: 90%;
        padding-top: 5rem;
        padding-right: 5rem;
        padding-left: 5rem;
        padding-bottom: 5rem;
    }}
    img{{
    	max-width:40%;
    	margin-bottom:40px;
    }}
</style>
""",
        unsafe_allow_html=True,
    )

st.title("Laptop Predictor App")

df = pickle.load(open("df.pkl","rb"))
model = pickle.load(open("model.pkl","rb"))

company = st.selectbox("Brand of the Laptop",df["Company"].unique())
typeName = st.selectbox("Type of Laptop",df["TypeName"].unique())
ram = st.selectbox("RAM(in GB)",df["Ram"].unique())
weight = st.number_input("Weight of the Laptop")
touchscreen = st.selectbox("Touchscreen",["Yes","No"])
ips = st.selectbox("IPS",["Yes","No"])
screenSize = st.number_input("Screen Size")
screenResolution = st.selectbox("Screen Resolution",['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])
cpu = st.selectbox("Cpu",df["Cpu brand"].unique())
hdd = st.selectbox("HDD (in GB)",[0,128,256,512,1024,2048])
sdd = st.selectbox("SDD (in GB)",[0,8,128,256,512,1024])
gpu = st.selectbox("GPU",df["Gpu_brand"].unique())
os = st.selectbox("OS",df["Os"].unique())

if st.button("Predict Price"):
    ppi = None
    if touchscreen == "Yes":
        touchscreen = 1
    else:
        touchscreen = 0
    if ips == "Yes":
        ips = 1
    else:
        ips = 0

    X_res = int(screenResolution.split("x")[0])
    Y_res = int(screenResolution.split("x")[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screenSize
    query = np.array([company,typeName,ram,weight,touchscreen,ips,ppi,cpu,hdd,sdd,gpu,os])
    query = query.reshape(1,12)
    query = model.predict(query)
    st.title(f"Price of the Laptop:{int(np.exp(query)[0])}",)