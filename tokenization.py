sentence = 'Looking for a PART-TIME Consultant in Public Health! Interim role from the end of June until March next year.'
splited_sentence = sentence.split()
print(splited_sentence)
print(len(splited_sentence))
print(type(splited_sentence))

!pip install stanza
import stanza
en = stanza.download('en')
en = stanza.Pipeline(lang='en')
print(en)
print(type(en))

tokenized = en(sentence)
print(tokenized.sentences[0])
print(len(tokenized.sentences))
print(sentence)

for sen in tokenized.sentences:
  for w in sen.tokens:
    print(w.text)

en2 = stanza.Pipeline(lang='en')
tokenized_sent_2 = en2('Hari went to school')
print(tokenized_sent_2.sentences[0].tokens[0])
print(type(tokenized_sent_2.sentences[0].tokens[0]))

dicts = tokenized_sent_2.to_dict()
obj1 = dicts[0][0]['xpos']
print(obj1)
print(type(obj1))

for sen in dicts:
  for words in sen:
    print(words['text'], words['lemma'], words['xpos'])

