# Introduction

This is a PyTorch implementation of [A New Method of Region Embedding for Text Classification](https://openreview.net/pdf?id=BkSDMA36Z)  (ICLR 2018).
MR dataset is used  by default.

![region embedding](figures/region_embedding.png)


**NOTE**: in this implementation, only Context-Word Region Embedding is available for now. I do not have a plan to implement Word-Context Region Embedding
in the intermediate future. If anyone has a chance to implement the former, please let me know. Thanks in advance.

I also added a self-attention layer on top of the region embedding layer to get better results.

# Requirements
* python 3
* PyTorch >= 0.3
* sklearn
* numpy

# Usage
```
python main.py [arguments]

```

# Arguments
```
-h, -help       help information
-embed_dim      dimension of word embeddings
-epochs         number of epoches

```
