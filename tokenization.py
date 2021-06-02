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

