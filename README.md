<div align="center">

<h1> Title Generator </h1>

*Generating titles from abstracts*

Small project fine-tuning Seq2Seq models to generate titles given the abstract of a paper.

</div>

### Quick Start

The fine-tuned model is available on the [Huggingface Models Hub](https://huggingface.co/), and can be loaded like any other Huggingface model,

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('pbmstrk/t5-large-arxiv-abstract-title')
model = AutoModelForSeq2SeqLM.from_pretrained('pbmstrk/t5-large-arxiv-abstract-title')
```

To generate a title, simply tokenize an abstract and use `model.generate()`.

```python
# from paper Digital Voicing of Silent Speech
abstract="In this paper, we consider the task of digitally voicing silent speech, where silently \
mouthed words are converted to audible speech based on electromyography (EMG) \
sensor measurements that capture muscle impulses. While prior work has focused \
on training speech synthesis models from EMG collected during vocalized speech, \
we are the first to train from EMG collected during silently articulated speech. \
We introduce a method of training on silent EMG by transferring audio targets from \
vocalized to silent signals. Our method greatly improves intelligibility of audio generated \
from silent EMG compared to a baseline that only trains with vocalized data, decreasing transcription \
word error rate from 64% to 4% in one data condition and 88% to 68% in another. To spur \
further development on this task, we share our new dataset of silent and vocalized facial EMG \
measurements."

inputs = tokenizer(text=abstract, return_tensors="pt")
outputs = model.generate(**inputs)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
# ['Speech Synthesis from Electromyography']
```

### Details

The model was fine-tuned on 80,000 abstract-title pairs extracted from the [Arxiv Dataset](https://www.kaggle.com/Cornell-University/arxiv). The `title-generator` module includes the `ArxivDataset` class to enable easier use of the dataset.

### Fine-tune your own Model

The `scripts/finetune.py` file can be used to fine-tune models on the title-predict task. To run the script, first install the module using
```
pip install ".[scripts]"
```
and then run the script
```
python scripts/finetune.py
```
The arguments are handled using Hydra, and can be modified either in the config file or overwritten in the command line.
