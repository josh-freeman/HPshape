# HPShape: Shape of Harry Potter books with Word2Vec

## Introduction

Can we associate a style of writing to a shape? This is the kind of idea we explore here. 

## How we proceeded

- (First we learned about NLP by focusing on chapters 23 and 24 of the Modern AI book by Peter Norvig and Russell. Learning about theoretical concepts, like context free grammars and suchlike, was instrumental to get a better grasp of the context. Further, we implemented an n-gram prediction model, as seen in the examples/ngrams.ipynb file.)

- Implementation

  - After reading the first word2vec paper, we implemented Word2Vec as a basic neural model, consisting of a first one-of-V layer, followed by a projection into latent space of dimension D, followed by prediction of the context. 

  - In the following, V is the size of the vocabulary. 2c is the word window size.

    ![NN](https://user-images.githubusercontent.com/47647715/192298107-e61bd927-e448-407a-ae1c-6e9f037e5f34.png)

  - Prediction of the context was done in a (significantly?) different way than in the text

    - Let $T$ be the number of words in a training sentence that are at the center of a window.The loss was seen as an average log likelihood, $$\displaystyle l=\frac{1}{T}\log\left({\sum_{t=1}^T\sum_\underset{j\neq 0}{{-c\leq j\leq c}}p(w_{t+j}|w_j)}\right).$$ ##TODO finish this

  - To make the validation set during training, we take the final 10% of the text used for training and preprocess it separately. This is done to prevent an intersection between 

  - A slight difference here of how we predicted the context of a given word is that we modelized a context as a single (non normalized) histogram vector.

- Interpretation

  - We then projected the words into a F (:=15) dimensional space with TSVD (LSA) before going to 3 dimensional space with t-SNE.
  - After this, we plotted the data by interpreting the 3D vectors as spherical coordinates. This  was done because the data was somewhat distributed in a ball. 


## Results and limitations 

- We observed that some words were closer to similar words. For example, November was close to autumn and yellowish. However, this is not a scientific proof of anything. 
- A crucial problem is, however, the lack of precise testing tasks to test the model on. For instance, we used no objective way to objectively test the ability of the model to capture semantic relationships between **characters**.
- The number of epochs is also a problem here: we only trained from scratch for about 5 epochs. During these few epochs, we see some overfitting.
- A few other points:
  - In the papers, we did not find the values of all hyperparameters used.
  - In the few epochs, we saw heavy overfitting.
  - Next, we could try it with more epochs. We could also study the persistent homology of these sets of vectors.


## Reproduce it at home

- You are welcome to use the bash or batch scripts provided. Otherwise, you can
  - Clone the repository by typing ``git clone https://github.com/josh-freeman/HPshape.git``
  - Download the HP books on github at ``https://github.com/formcept/whiteboard/tree/master/nbviewer/notebooks/data``
  - **Training:** 
  - If you don't already have one, prepare a folder at the same level as the `src` and `util` folders, by the name of the constant `RESOURCES_DIRNAME` (defaults to `examples`). Do the same for the `CHECKPOINT_DIRNAME`.
  - change the ``constants.py`` file according to what you want. You might want to think about:
    - The number of epochs
    - The amount of RAM allocated to the spacy model (if you are on linux and the amount is too high for your computer, the program might crash). 
    - The list of text files (Harry Potter books, for example) to train on. This **needs** to match a list of files in your folder by the name of the constant `RESOURCES_DIRNAME` (`examples` by default). 

  - run the main module by typing ``python3 -m src.__main__``
  - **Loading:** Use the ``load.py`` script to load



Word of warning: we chose to allocate 2 MB of RAM in pure text for the lemmatizer. This means that a lot more is 
going to be used for your computer. Please run this on a computer with at least 20 GIgs of RAM