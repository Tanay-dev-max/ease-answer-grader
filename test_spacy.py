import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Photosynthesis converts sunlight into chemical energy.")
for chunk in doc.noun_chunks:
    print(chunk.text)