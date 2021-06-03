!pip install gensim

from gensim.models.word2vec import Word2Vec
import gensim.downloader as api

print(api.info)
print(api.info())

model_w2v = api.load('word2vec-google-news-300')

print(model_w2v.most_similar("cookies",topn=10))
print(model_w2v.doesnt_match(["Bannana","Melon","Pear","House"]))
king = model_w2v['king']
print(king)
print(type(king))
woman = model_w2v['woman']
man = model_w2v['man']

sim_queen_vector = king + woman - man

print(model_w2v.similar_by_vector(sim_queen_vector))


