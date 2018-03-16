# ontology-matching

Idea: two ontologies, pass a product from one and calculate the similarity to ALL the ones from the other -> top 10 choices?

With this module you can use pre-trained model or - if you have enought data - generate your own from scratch. A collectiond of recommended ready-to-use models can be found [here](http://ahogrammer.com/2017/01/20/the-list-of-pretrained-word-embeddings).

There's several options available to build your own model based on a term-frequency model. These are:
* Latent Semantic Indexing (LSI)
* Random Projections (RP)
* Latent Dirichlet Allocation (LDA)
* Hierarchical Dirichlet Process (HDP)

To determine similarity between two phrases the cosine similaity is used:

![eq1](https://github.com/faboo8/ontology-matching/blob/master/media/CodeCogsEqn.gif)


