# semantic-fruitfly – The Original

### What's this?
This is the code described in [Preissner & Herbelot (2019)](http://ceur-ws.org/Vol-2481/paper59.pdf "To be Fair: a Case for Cognitively-Inspired Models of Meaning"). It provides an environment for an application of the Fruitfly Algorithm (FFA) to distributional semantics (DS). 

The FFA is a hashing algorithm that uses random indexing to perform locality-sensitive hashing. It is inspired by the olfactory system of the fruit fly (hence the name). The FFA presented here is an extension of the algorithm described in [Dasgupta et al. (2017)](https://science.sciencemag.org/content/sci/358/6364/793.full.pdf). 
This implementation of the FFA is adapted to an incremental count-based approach to distributional semantics. In contrast to other approaches, it is relatively lightweight and transparent, allowing to tinterpret the vectors that it produces.
This implementation of the FFA takes any non-negative embedding-like input and produces binary hash signatures.

This repository provides 
- code that creates count models from text resources
- an implementation of the FFA
- a tiny data set as showcase resource
- a dataset for evaluation


### Installation 
After cloning the repository, make sure that the packages stated in requirements.txt are installed in your environment. You can try 
```bash
pip install -r requirements.txt
```

### The Main Parts

#### Fruitfly.py
The algorithm itself. If you want to apply the FFA to a set of word vectors (e.g., count vectors or non-negative embeddings), execute `Fruitfly.py` directly. You can also evaluate a space against a testset in this way.
FFAs (and of course, the hash seignatures that they produce) can be logged for later use.  

#### Incrementor.py
Objects of the Incrementor class take the job of counting co-occurrences in text (i.e., populating and maintaining a count table) and, optionally, of maintaining a Fruitfly object alongside. It can thus connect a count model to an FFA and keep them optimally compatible. Count models as well as FFAs can be newly set-up or loaded from a file.

#### spacebreeder.py
This script implements the main experiment described in [Preissner & Herbelot (2019)](http://ceur-ws.org/Vol-2481/paper59.pdf "To be Fair: a Case for Cognitively-Inspired Models of Meaning").
Given a text resource, it builds a frequency count model step by step (e.g., 1M words at a time) and maintains a Fruitfly object alongside. After each step, the count model is hashed by the FFA and the resulting hash signatures are evaluated against a test set. There is an option to train a Word2Vec model at the end of each step for a comparison. The results and produced spaces are stored in the `results/` directory. 

By working step by step and being independent from previously encountered data in the hashing phase, this setup is incremental.

For copyright reasons, the [Word2Vec code](https://github.com/tmikolov/word2vec "Word2Vec C implementation") is not provided.

#### hyperopt.py
This script takes a non-negative space and performs a hyperparameter grid search over the FFA. To speed up the grid search, the runs can be parallelized.

#### Data
the `data/` directory contains
- `chunks/` — 10 files with \~10k words each, taken from a [Wikipedia corpus](https://archive.org/details/enwiki-20181120 "Wikimedia dump from 2018-11-20")
- `MEN_dataset_natural_form_full` — a human-annotated list of word pair similarities, taken from [Bruni et al. (2014)](https://staff.fnwi.uva.nl/e.bruni/MEN)

Any outputs go to the `results/` directory or subdirectories created at runtime.


### Instabilities and Further Work
- There is an option to count co-occurrences line by line. However, this option is not tested.
- There is an option to count, hash, and evaluate part-of-speech tagged data. However, this is not stable.
- `Incrementor.py` and `Fruitfly.py` implement a functionality to reduce the number of dimensions and input nodes, respectively, to counteract their inflation by infrequent words. This is not described in [Preissner & Herbelot (2019)](http://ceur-ws.org/Vol-2481/paper59.pdf "To be Fair: a Case for Cognitively-Inspired Models of Meaning").
- This implementation of the FFA as well as several other parts of the code are not optimized for speed. Future implementations might benefit from such an optimization.
