# ontology-matching

Idea: two ontologies, pass a product from one and calculate the similarity to ALL the ones from the other -> top 10 choices?

With this module you can use pre-trained model or - if you have enought data - generate your own from scratch. A collectiond of recommended ready-to-use models can be found [here](http://ahogrammer.com/2017/01/20/the-list-of-pretrained-word-embeddings).

There's several options available to build your own model based on a term-frequency model. These are:
* Latent Semantic Indexing (LSI)
* Random Projections (RP)
* Latent Dirichlet Allocation (LDA)
* Hierarchical Dirichlet Process (HDP)

To determine similarity between two phrases the cosine similarity 

  ![eq1](https://github.com/faboo8/ontology-matching/blob/master/media/CodeCogsEqn.gif)

is used although this might not be the best for probabilistic models. 

However, pre-trained as well as models trained from texts disregard semantic similarity e.g. cat <-> feline. Thus a similarity measure using [WordNet](https://wordnet.princeton.edu/) is implemented. So far I'm using a maximum similarity approach using the Wu & Palmer similarity (more info [here](http://search.cpan.org/dist/WordNet-Similarity/lib/WordNet/Similarity/wup.pm)) since it worked well with the data I'm working on. In the future I want to use the weights from the inverse document frequency (similar to [here](https://nlpforhackers.io/tf-idf/)) but I haven't worked out the kinks yet.
