# ontology-matching


<p align="center">
  <img src="https://github.com/faboo8/ontology-matching/blob/master/media/0.jpg" alt="sign"/>
</p>
  
**Idea**: 
I have two ontologies (or more general two lists) and want to match the entries from the first to the second. IN order to do this, I pass an entry from one and calculate the similarity to all the ones from the other. The goal is to have an *n*:1 matching with minimal user input. A threshold for the similarity shall be defined under which additional user input is required (that should be dependent on how likely the next best matches are).

With this module you can use pre-trained model or - if you have enough data - generate your own from scratch. A collectiond of recommended ready-to-use models can be found [here](http://ahogrammer.com/2017/01/20/the-list-of-pretrained-word-embeddings). Per default Google's Word2Vec model is used. 

## Computing similarities

To determine similarity between two phrases the cosine similarity 

<p align="center">
  <img src="https://github.com/faboo8/ontology-matching/blob/master/media/CodeCogsEqn.gif" alt="eq1"/>
</p>


is used although this might not be the best for probabilistic models. 

However, pre-trained as well as models trained from texts disregard semantic similarity e.g. cat <-> feline. Thus a similarity measure using [WordNet](https://wordnet.princeton.edu/) is implemented. So far I'm using a maximum similarity approach using the Wu & Palmer similarity (more info [here](http://search.cpan.org/dist/WordNet-Similarity/lib/WordNet/Similarity/wup.pm)) since it worked well with the data I'm working on. In the future I want to use the weights from the inverse document frequency (similar to [here](https://nlpforhackers.io/tf-idf/)) but I haven't worked out the kinks yet.


The PhraseVector class offers 4 ways to compute the similarity between two phrases:
* **CosineSimilarity**:  can be seen as a comparison between documents on a normalized space because we’re not taking into the consideration only the magnitude of each word count (tf-idf) of each document, but the 'angle' between the documents.
* **WordNetSimilarity**: as described above
* **PhraseCompare**: not yet functional but it will allow the user to input their own model and perform a similarity calculation with the methods from the gensim module.
* **CombinedSimilarity**: computes an averaged similarity measure of CosineSimilarity and WordNetSimilarity. A list can be passed as the weight otherwise [0.8,0.2] will be used. 

## Building your own model 

There's several options available to build your own model based on a term-frequency model. These are:
* Latent Semantic Indexing (LSI)
* Random Projections (RP)
* Latent Dirichlet Allocation (LDA)
* Hierarchical Dirichlet Process (HDP)

This feature is still being implemented and isn't fully functional yet. 

## How to use

So far this module works with pandas DataFrames hardcoded in the script. Save the two lists/ontologies as as dataframe df1 and df2 and run the following command in the terminal:

`python phrase_similarity.py`

## Dependencies

Tested with:
gesim 3.4, tqdm 4.19.5, numpy 1.9.3, nltk 3.2.5, pandas 0.22, python 3.63


## Tasks
In order of importance:
- [ ] Make use of the different hierarchies of the ontologies
- [ ] Make user input easier
- [ ] Define a procedure to determine the 'best' weights for CombinedSimilarity (training set needed)
- [ ] Make building your own model functional
- [ ] Rewrite WordNetSimilarity to use weights (returns 1.0 too often)

