# Speech-Transformer
  
PyTorch implementation of [The SpeechTransformer for Large-scale Mandarin Chinese Speech Recognition](https://ieeexplore.ieee.org/document/8682586).
    
<img src="https://user-images.githubusercontent.com/42150335/90434869-17e41400-e109-11ea-9738-9a4a53f884c7.png" width=500>
  
Speech Transformer is a transformer framework specialized in speech recognition tasks.  
This repository contains only model code, but you can train with speech transformer with this [repository](https://github.com/sooftware/KoSpeech).  
I appreciate any kind of [feedback or contribution](https://github.com/sooftware/Speech-Transformer/issues)  
    
## Usage
- Training
```python
import torch
from speech_transformer import SpeechTransformer

BATCH_SIZE, SEQ_LENGTH, DIM, NUM_CLASSES = 3, 12345, 80, 4

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

inputs = torch.rand(BATCH_SIZE, SEQ_LENGTH, DIM).to(device)
input_lengths = torch.IntTensor([100, 50, 8])
targets = torch.LongTensor([[2, 3, 3, 3, 3, 3, 2, 2, 1, 0],
                            [2, 3, 3, 3, 3, 3, 2, 1, 2, 0],
                            [2, 3, 3, 3, 3, 3, 2, 2, 0, 1]]).to(device)  # 1 means <eos_token>
target_lengths = torch.IntTensor([10, 9, 8])

model = SpeechTransformer(num_classes=NUM_CLASSES, d_model=512, num_heads=8, input_dim=DIM)
predictions, logits = model(inputs, input_lengths, targets, target_lengths)
```
- Beam Search Decoding
```python
import torch
from speech_transformer import SpeechTransformer

BATCH_SIZE, SEQ_LENGTH, DIM, NUM_CLASSES = 3, 12345, 80, 10

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

inputs = torch.rand(BATCH_SIZE, SEQ_LENGTH, DIM).to(device)  # BxTxD
input_lengths = torch.LongTensor([SEQ_LENGTH, SEQ_LENGTH - 10, SEQ_LENGTH - 20]).to(device)

model = SpeechTransformer(num_classes=NUM_CLASSES, d_model=512, num_heads=8, input_dim=DIM)
model.set_beam_decoder(batch_size=BATCH_SIZE, beam_size=3)
predictions, _ = model(inputs, input_lengths)
```
  
## Troubleshoots and Contributing
If you have any questions, bug reports, and feature requests, please [open an issue](https://github.com/sooftware/Jasper-pytorch/issues) on github or   
contacts sh951011@gmail.com please.
  
I appreciate any kind of feedback or contribution.  Feel free to proceed with small issues like bug fixes, documentation improvement.  For major contributions and new features, please discuss with the collaborators in corresponding issues.  
  
## Code Style
I follow [PEP-8](https://www.python.org/dev/peps/pep-0008/) for code style. Especially the style of docstrings is important to generate documentation.  
  
## Reference
- [The SpeechTransformer for Large-scale Mandarin Chinese Speech Recognition (Yuanyuan Zhao et al, 2019)](https://ieeexplore.ieee.org/document/8682586)  
- [kaituoxu/Speech-Transformer](https://github.com/kaituoxu/Speech-Transformer)
  
## Author
  
* Soohwan Kim [@sooftware](https://github.com/sooftware)
* Contacts: sh951011@gmail.com
