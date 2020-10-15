# Plainifier

*Plainifier* is a solution for multi-word lexical simplification, described in the article *[Multi-Word Lexical Simplification](https://www.aclweb.org/anthology/TODO.pdf)* presented at the [COLING 2020](https://coling2020.org/) conference in Barcelona.

Plainifier uses the [TerseBERT](https://github.com/piotrmp/tersebert) language model to recursively generate replacement candidates that fit in a given context. These candidates are ranked according to the following criteria:
* Probability, i.e. likelihood according to the language model,
* Similarity, i.e. how much the generated fragment resembles the meaning of the original one, measured by cosine distance of token embeddings,
* Familiarity, i.e. how commonly used the included words are, according to frequency in a large corpus.

If you need any more information consult [the paper](https://www.aclweb.org/anthology/TODO.pdf) or contact its authors! 

## Running Plainifier
The code for plainifier is included in the ```plainify.py``` file, which also includes a usage example. To run Plainifier, you will need to have the following resources available:
* [Hugging Face Transformers](https://github.com/huggingface/transformers), [PyTorch](https://pytorch.org/) and [NumPy](https://numpy.org/) libraries,
* [TerseBERT](https://github.com/piotrmp/tersebert) model,
* [fastText](https://fasttext.cc/docs/en/english-vectors.html) embeddings file (crawl-300d-2M-subword.vec),
* Frequency table of unigrams obtained from [Google Books Ngrams](http://storage.googleapis.com/books/ngrams/books/datasetsv2.html), available here in ```unigrams-df.tsv```.

Note that the running time of Plainifier heavily depends on the parameters specified. For the meaning of the parameters and their values used in the evaluation, refer to [the paper](https://www.aclweb.org/anthology/TODO.pdf).

## Licence
Plainifier is released under the [GNU GPL 3.0](https://www.gnu.org/licenses/gpl-3.0.html) licence.

## Citation

Przybyła, P. and Shardlow, M., 2020, December. Multi-Word Lexical Simplification. In Proceedings of the 28th International Conference on Computational Linguistics.


    @inproceedings{plainifier,
        title = "Multi-Word Lexical Simplification",
        author = {Przyby{\l}a, Piotr and Shardlow, Matthew},
        booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
        month = dec,
        year = "2020",
        address = "Barcelona, Spain",
        publisher = "Association for Computational Linguistics",
    }

