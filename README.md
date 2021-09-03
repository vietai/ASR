# Vietnamese end-to-end speech recognition using wav2vec 2.0

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vietnamese-end-to-end-speech-recognition/speech-recognition-on-common-voice-vi)](https://paperswithcode.com/sota/speech-recognition-on-common-voice-vi?p=vietnamese-end-to-end-speech-recognition)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vietnamese-end-to-end-speech-recognition/speech-recognition-on-vivos)](https://paperswithcode.com/sota/speech-recognition-on-vivos?p=vietnamese-end-to-end-speech-recognition)


### Model description

[Our models](https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h) are pre-trained on 13k hours of Vietnamese youtube audio (un-label data) and fine-tuned on 250 hours labeled of [VLSP ASR dataset](https://vlsp.org.vn/vlsp2020/eval/asr) on 16kHz sampled speech audio. 

We use [wav2vec2 architecture](https://ai.facebook.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/) for the pre-trained model. Follow wav2vec2 paper:

>For the first time that learning powerful representations from speech audio alone followed by fine-tuning on transcribed speech can outperform the best semi-supervised methods while being conceptually simpler.

For fine-tuning phase, wav2vec2 is fine-tuned using Connectionist Temporal Classification (CTC), which is an algorithm that is used to train neural networks for sequence-to-sequence problems and mainly in Automatic Speech Recognition and handwriting recognition.

| Model | #params | Pre-training data | Fine-tune data |
|---|---|---|---|
| [base]((https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h)) | 95M | 13k hours | 250 hours |

In a formal ASR system, two components are required: acoustic model and language model. Here ctc-wav2vec fine-tuned model works as an acoustic model. For the language model, we provide a [4-grams model](https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h/blob/main/vi_lm_4grams.bin.zip) trained on 2GB of spoken text. 

Detail of training and fine-tuning process, the audience can follow [fairseq github](https://github.com/pytorch/fairseq/tree/master/examples/wav2vec) and [huggingface blog](https://huggingface.co/blog/fine-tune-wav2vec2-english).

### Benchmark WER result:

| | [VIVOS](https://ailab.hcmus.edu.vn/vivos) | [COMMON VOICE VI](https://paperswithcode.com/dataset/common-voice) | [VLSP-T1](https://vlsp.org.vn/vlsp2020/eval/asr) | [VLSP-T2](https://vlsp.org.vn/vlsp2020/eval/asr) |
|---|---|---|---|---|
|without LM| 10.77 | 18.34 | 13.33 | 51.45 |
|with 4-grams LM| 6.15 | 11.52 | 9.11 | 40.81 |


### Example usage

When using the model make sure that your speech input is sampled at 16Khz. Audio length should be shorter than 10s. Following the Colab link below to use a combination of CTC-wav2vec and 4-grams LM.

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

### Model Parameters License

The ASR model parameters are made available for non-commercial use only, under the terms of the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license. You can find details at: https://creativecommons.org/licenses/by-nc/4.0/legalcode

### Citation

[![CITE](https://zenodo.org/badge/DOI/10.5281/zenodo.5356039.svg)](https://github.com/vietai/ASR)

```text
@misc{Thai_Binh_Nguyen_wav2vec2_vi_2021,
  author = {Thai Binh Nguyen},
  doi = {10.5281/zenodo.5356039},
  month = {09},
  title = {{Vietnamese end-to-end speech recognition using wav2vec 2.0}},
  url = {https://github.com/vietai/ASR},
  year = {2021}
}
```

**Please CITE** our repo when it is used to help produce published results or is incorporated into other software.

### Contact 

nguyenvulebinh@gmail.com / binh@vietai.org

[![Follow](https://img.shields.io/twitter/follow/nguyenvulebinh?style=social)](https://twitter.com/intent/follow?screen_name=nguyenvulebinh)
