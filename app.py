import streamlit as st
from main import ner_zero_shot, _ner_zero_shot_tags,ner_one_shot, ner_one_shot_desc

st.title("NER : Promptify")

st.write("Named Entity Recognition using Promptify")

text=st.text_input("Enter the Text", key="Text Box")

mode = st.sradio("Mode of NER (Zero/One)", ['Zero Shot','Zero shot with custom Label','One Shot','One shot with domian knowledge'])

if st.button("RUN"):

    if "Zero Shot" in mode:
        if "custom" in mode:
            output= _ner_zero_shot_tags(text)
        output= ner_zero_shot(text)
    else:
        if "domain" in mode:
            output= ner_one_shot_desc(text)
        output= ner_one_shot(text)

    st.write(output)
