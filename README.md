# Vietnamese end-to-end speech recognition using wav2vec 2.0

### Model description

[The model](https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h) pre-trained and fine-tuned on 250 hours of [VLSP ASR dataset](https://vlsp.org.vn/vlsp2020/eval/asr) on 16kHz sampled speech audio. When using the model make sure that your speech input is also sampled at 16Khz.

### Usage

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pVBY46gSoWer2vDf0XmZ6uNV3d8lrMxx?usp=sharing)

```python
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import soundfile as sf
import torch

# load model and tokenizer
processor = Wav2Vec2Processor.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")
model = Wav2Vec2ForCTC.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")

# define function to read in sound file
def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch

# load dummy dataset and read soundfiles
ds = map_to_array({
    "file": 'audio-test/t1_0001-00010.wav'
})

# tokenize
input_values = processor(ds["speech"], return_tensors="pt", padding="longest").input_values  # Batch size 1

# retrieve logits
logits = model(input_values).logits

# take argmax and decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)
 ```

### Result WER (with 4-grams LM):

| [VIVOS](https://ailab.hcmus.edu.vn/vivos) | [VLSP-T1](https://vlsp.org.vn/vlsp2020/eval/asr) | [VLSP-T2](https://vlsp.org.vn/vlsp2020/eval/asr) |
|---|---|---|
| 6.1 | 9.1 | 40.8 |


### Model Parameters License

The ASR model parameters are made available for non-commercial use only, under the terms of the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license. You can find details at: https://creativecommons.org/licenses/by-nc/4.0/legalcode
