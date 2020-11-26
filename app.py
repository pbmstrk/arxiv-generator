import time
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


st.title("Title Generator")

@st.cache(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained('pbmstrk/t5-large-arxiv-abstract-title')
    model = AutoModelForSeq2SeqLM.from_pretrained('pbmstrk/t5-large-arxiv-abstract-title')
    return tokenizer, model

def generate_predictions(abstract, model, tokenizer, args):
    inputs = tokenizer(text=abstract, return_tensors="pt")
    outputs = model.generate(**inputs, **args)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


with st.spinner("Loading Model...\nDepending on your internet connection this may take a while."):
    tokenizer, model = load_model()


args = {}
abstract = st.text_area("Abstract", height=200)


option = st.selectbox('Method of generation?',
        ('Greedy Search', 'Beam Search', 'Sampling'))

if option == "Beam Search":
    num_beams = st.slider("Number of beams", min_value=2, max_value=20, value=2, step=1)
    args["num_beams"] = num_beams
    args["early_stopping"] = True

if option == "Sampling":
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=1.0)
    args["temperature"] = temperature
    args["do_sample"]=True
    args["top_k"]=0

num_generate = st.slider("Number of titles to generate", min_value=1, max_value=5)
args["num_return_sequences"] = num_generate

clicked_compute = st.button("Compute")
outputs = st.empty()
if clicked_compute:
    if not abstract:
        st.error("Please enter an abstract")
    else:
        with st.spinner('Generating titles...'):
            titles = generate_predictions(abstract, model, tokenizer, args)
        outputs.text_area("Outputs", titles)
        



