# HPShape: Shape of Harry Potter books with Word2Vec

## Introduction

Can we associate a style of writing to a shape? This is the kind of idea we explore here. 

## How we proceeded

- (First we learned about NLP by focusing on chapters 23 and 24 of the Modern AI book by Peter Norvig and Russell. Learning about theoretical concepts, like context free grammars and suchlike, was instrumental to get a better grasp of the context.)
- After reading the first word2vec paper, we implemented Word2Vec as a basic neural model, consisting of a first one-of-V layer, followed by a projection into latent space of dimension D, followed by prediction of the context. A slight difference here of how we predicted the context of a given word is that we modelized a context as a single (non normalized) histogram vector.
- We then projected the words into a F (:=15) dimensional space with TSVD (LSA) before going to 3 dimensional space with t-SNE.

## Results and limitations 

- We observed that some 
- A crucial problem is, however, the lack of precise testing tasks to test the model on. For instance, we used no objective way to objectively test the ability of the model to capture semantic relationships between **characters**.
- The number of epochs is also a problem here: we only trained from scratch for about 5 epochs. During these few epochs, we see some overfitting.

## Reproduce it at home

- Clone the repository by typing``git clone ...``
- Download the HP books on github at ...
- **Training:** change the ``constants.py`` file according to what you want, and run the main module by typing ``...``
- **Loading:** Use the ``load.py`` script to load

