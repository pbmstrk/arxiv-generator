import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAMES = {
        "Abstract": "pbmstrk/t5-large-arxiv-title-abstract",
        "Title": "pbmstrk/t5-large-arxiv-abstract-title"
}

st.title("Arxiv Generator")

@st.cache(show_spinner=False)
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def generate_predictions(abstract, model, tokenizer, args):
    inputs = tokenizer(text=abstract, return_tensors="pt")
    outputs = model.generate(**inputs, **args)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

predict_type = st.radio("What do you want to generate?", ('Abstract', 'Title'))

with st.spinner("Loading model..."):
    model_name = MODEL_NAMES[predict_type]
    tokenizer, model = load_model(model_name)

generation_args = {}

if predict_type == "Title":
    inputs = st.text_area("Abstract", height=200)
if predict_type == "Abstract":
    inputs = st.text_area("Title", height=70)

option = st.selectbox('Method of generation?',
        ('Greedy Search', 'Beam Search', 'Sampling'))

if option == "Beam Search":
    num_beams = st.slider("Number of beams", min_value=2, max_value=20, value=2, step=1)
    generation_args["num_beams"] = num_beams
    generation_args["early_stopping"] = True

if option == "Sampling":
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=1.0)
    generation_args["temperature"] = temperature
    generation_args["do_sample"]=True
    generation_args["top_k"]=0

num_generate = st.slider("Number of titles to generate", min_value=1, max_value=5)
generation_args["num_return_sequences"] = num_generate

max_length = st.slider("Max length", min_value=10, max_value=300)
generation_args["max_length"] = max_length

clicked_compute = st.button("Compute")
outputs_area = st.empty()
if clicked_compute:
    if not inputs:
        st.error("Please enter an input")
    else:
        with st.spinner('Generating output...'):
            outputs = generate_predictions(inputs, model, tokenizer, generation_args)
        outputs_area.text_area("Outputs", outputs)

