# encoding: utf-8
from __future__ import unicode_literals
import codecs
import glob
import time
import stopword_filtering
import tokenization
import gensim
import dictionary_tokenization

class EnglishSentences(object):
  def __init__(self, globPattern, tknsr, debug=False):
    self.globPattern = globPattern
    self.debug = debug
    self.tknsr = tknsr

  def __iter__(self):
    i = 0
    for f in glob.glob(self.globPattern):
      i+=1
      txt = get_text(f)
      for sentence in tokenization.segment_to_sentences(txt):
        yield self.tknsr(sentence)
      if self.debug and i % 100 == 0:
        print(i)


def train_and_save(multiword_dic_path, stopword_path, corpus_files_glob_pattern, mode_file_name = 'rock_music.w2v'):
  # Creates a tokenizer that will see multiword band names as single "tokens". Tokens are traditionally single word
  tknsr = dictionary_tokenization.DictionaryBasedMultiwordFinder(dictionary_path=multiword_dic_path)

  # Retrieves a filter to remove stopwords. In natural language processing, stopwords are words that are used very often and add very little information
  # and are so removed from the set of words. Such common words can also throw off many natural language algorithms so they occur so often 
  # and in so many contexts that they can become overimportant to the models
  filtr = stopword_filtering.StopwordFilter(stopword_path)

  # Build a function that tokenizer which will filter out stop words and create multiwords
  tokenizze = lambda text: filtr.filter(tknsr.tokenise(text))

  # EnglishSentences class simply reads in all the files and applies the tokenizer to them and then returns the large corpus of english sentences.
  # This filtered and tokenized corpus is fed to the gensim word2vec model
  wvc = gensim.models.Word2Vec(EnglishSentences(corpus_files_glob_pattern, tokenizze, True), min_count=2)
  wvc.save(mode_file_name)


'''
# Uncomment to train =>

train_and_save('data/wiki_rock_multiword_dic.txt', 'data/stop-words-english1.txt',
               '<THE_LOCATION>/wiki_rock_corpus/*.txt')

'''