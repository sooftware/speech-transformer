# Speech-Transformer
  
`Speech Transformer` is a transformer framework specialized in speech recognition tasks.  
I implemented this repo by referring to several repositories.  
I appreciate any kind of [feedback or contribution](https://github.com/sooftware/Speech-Transformer/issues)  
  
<img src="https://user-images.githubusercontent.com/42150335/90434869-17e41400-e109-11ea-9738-9a4a53f884c7.png" width=500>
  
This repository focused on implementing transformers that are specialized in speech recognition.  
While at the same time striving for a readable code. To improve readability,  
I designed the model structure to fit as much as possible to the blocks in the above Transformers figure.  
  
## Usage
```python
from transformer import SpeechTransformer

model = SpeechTransformer(num_classes, d_model=512, num_heads=8, input_dim=80, extractor='vgg')
output = model(inputs, input_lengths, targets, return_attns=False)
```
  
## Prerequisites
  
* Numpy: `pip install numpy` (Refer [here](https://github.com/numpy/numpy) for problem installing Numpy).
* Pytorch: Refer to [PyTorch website](http://pytorch.org/) to install the version w.r.t. your environment.   
  
## Reference  
  
* [Speech-Transformer](https://github.com/kaituoxu/Speech-Transformer)
* [transformer](https://github.com/JayParks/transformer)
* [nlp-tutorial](https://github.com/graykode/nlp-tutorial)
* [transformer-pytorch](https://github.com/dreamgonfly/transformer-pytorch)
* [attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch)
  
## Author
  
* Soohwan Kim [@sooftware](https://github.com/sooftware)
* Contacts: sh951011@gmail.com
