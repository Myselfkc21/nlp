import spacy
import random
from spacy.util import minibatch
from spacy.training.example import Example

train_data = [
       ('What is the price of a 2BHK apartment in Whitefield?', 
     {"entities": [(22, 26, "CONFIGURATION"), (40, 50, "LOCATION")]}),
    ('Find me a 3BHK flat in Koramangala', 
     {"entities": [(11, 15, "CONFIGURATION"), (23, 34, "LOCATION")]}),

    # Area + Location
    ('What is the area of a 1200 sqft plot in Electronic City?', 
     {"entities": [(21, 30, "AREA"), (38, 53, "LOCATION")]}),
    ('How much does a 2500 sqft villa cost in Jayanagar?', 
     {"entities": [(15, 24, "AREA"), (40, 49, "LOCATION")]}),

    # Developer
    ('Who is the developer of Prestige Lakeside Habitat?', 
     {"entities": [(21, 48, "DEVELOPER")]}),
    ('Tell me about Sobha Dream Acres', 
     {"entities": [(13, 31, "DEVELOPER")]}),

    # Price + Configuration + Location
    ('What is the price of a 3BHK apartment in HSR Layout?', 
     {"entities": [(22, 26, "CONFIGURATION"), (40, 50, "LOCATION")]}),
    ('How much does a 4BHK villa cost in Indiranagar for ₹2 crores?', 
     {"entities": [(15, 19, "CONFIGURATION"), (39, 50, "LOCATION"), (55, 64, "PRICE")]}),
    ('Find a 2000 sqft villa in Sarjapur Road for ₹1.5 crore', 
     {"entities": [(7, 16, "AREA"), (30, 42, "LOCATION"), (47, 56, "PRICE")]}),
    ('Is there a 2BHK apartment available in Whitefield for ₹75 lakhs?', 
     {"entities": [(12, 16, "CONFIGURATION"), (39, 49, "LOCATION"), (54, 63, "PRICE")]}),
    ('What is the cost of a 3BHK penthouse in Hebbal for ₹3.2 crores?', 
     {"entities": [(21, 25, "CONFIGURATION"), (39, 45, "LOCATION"), (50, 60, "PRICE")]}),

    # Developer + Price
    ('How much does an apartment by Brigade Group cost for ₹85 lakhs?', 
     {"entities": [(23, 36, "DEVELOPER"), (46, 55, "PRICE")]}),
    ('What is the starting price of a property by Godrej Properties at ₹1.8 crores?', 
     {"entities": [(41, 58, "DEVELOPER"), (63, 73, "PRICE")]}),
]

nlp= spacy.load('en_core_web_sm')
print(nlp.pipe_names)
if 'ner' not in nlp.pipe_names:
   ner = nlp.add_pipe('ner')
else:
    ner=nlp.get_pipe('ner')

for _,annotations in train_data:
    for ent in annotations['entities']:
        if ent[2] not in ner.labels:
            ner.add_label(ent[2])
            print(ent[2])

other_pipes=[pipe for pipe in nlp.pipe_names if pipe != 'ner']
print(other_pipes)
with nlp.disable_pipes(*other_pipes):
    optimizer=nlp.begin_training()

    epochs=100
    for epoch in range(epochs):
        random.shuffle(train_data)
        losses={}
        batches = minibatch(train_data,size=2)
        for batch in batches:
            examples=[]
            for text,annotations in batch:
                doc = nlp.make_doc(text)
                example=Example.from_dict(doc,annotations)
                examples.append(examples)
            nlp.update(examples,drop=0.5,losses=losses)
        print(f'Epoch {epoch+1}, Losses:{losses}')

nlp.to_disk('custom_ner_model')
trained_nlp= spacy.load('custom_ner_model')
test_sentences = [
    "I want to buy a 2BHK apartment in Indiranagar.",
    "Find me a 3BHK villa in Koramangala for ₹1.2 crores.",
    "What is the price of a 1200 sqft plot in Whitefield?",
    "Is there a 4BHK penthouse available in Jayanagar?",
    "I am interested in a 2500 sqft villa by Prestige Group.",
    "How much does a Sobha Dream Acres flat cost?",
    "Can I get an apartment in Sarjapur Road for ₹80 lakhs?",
    "Who is the developer of Godrej Properties?",
    "I need a 2000 sqft house in HSR Layout for ₹1.5 crore.",
    "Tell me about Brigade Group projects in Electronic City."
]

for text in test_sentences:
    doc=trainednlp(text)
    print(f'Text:{text}')
    print('Entitites',[(ent.text,ent.label)for ent in doc.ents])
    print()