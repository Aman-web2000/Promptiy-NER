import streamlit as st
from main import ner_zero_shot, ner_one_shot

st.title("NER : Promptify")

st.write("Named Entity Recognition using Promptify")

text=st.text_input("Enter the Text", key="Text Box")

mode = st.sradio("Mode of NER (Zero/One)", ['Zero Shot','One Shot'])

if st.button("RUN"):

    if mode=="Zero Shot":
        output= ner_zero_shot(text)
    else:
        output= ner_one_shot(text)

    st.write(output)
    