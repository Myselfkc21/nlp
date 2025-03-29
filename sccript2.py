import spacy
import random
from spacy.util import minibatch
from spacy.training.example import Example

train_data = [

    ("3BHK apartment in Mumbai for ₹1.5 crores by Lodha Group", {
        "entities": [(0, 4, "CONFIGURATION"), (15, 21, "LOCATION"), 
                   (26, 37, "PRICE"), (41, 52, "DEVELOPER")]
    }),
    
    # Price variations
    ("2BHK flat in Bangalore priced at ₹85 lakhs (DLF)", {
        "entities": [(0, 4, "CONFIGURATION"), (13, 22, "LOCATION"),
                   (33, 43, "PRICE"), (46, 49, "DEVELOPER")]
    }),
    ("₹3.25 crore penthouse in Hyderabad", {
        "entities": [(0, 10, "PRICE"), (22, 31, "LOCATION")]
    }),
    
    # Developer naming variations
    ("Prestige's new 4BHK project in Chennai", {
        "entities": [(0, 8, "DEVELOPER"), (17, 21, "CONFIGURATION"),
                   (33, 40, "LOCATION")]
    }),
    ("Property by Godrej Properties: 2BHK at ₹1.8 cr", {
        "entities": [(11, 28, "DEVELOPER"), (30, 34, "CONFIGURATION"),
                   (38, 46, "PRICE")]
    }),
    
    # Area specifications
    ("1500 sqft plot available in Whitefield", {
        "entities": [(0, 9, "AREA"), (27, 37, "LOCATION")]
    }),
    ("3000 sq.ft villa by Sobha in Kochi", {
        "entities": [(0, 9, "AREA"), (21, 26, "DEVELOPER"),
                   (30, 35, "LOCATION")]
    }),
    
    # Combined configurations
    ("2BHK+study apartment in Pune ₹1.25 cr", {
        "entities": [(0, 9, "CONFIGURATION"), (20, 23, "LOCATION"),
                   (24, 33, "PRICE")]
    }),
    ("3BHK with servant room in Gurgaon", {
        "entities": [(0, 4, "CONFIGURATION"), (21, 28, "LOCATION")]
    }),
    
    # Alternative price formats
    ("1BHK for 75 lacs in Noida", {
        "entities": [(0, 4, "CONFIGURATION"), (9, 17, "PRICE"),
                   (21, 26, "LOCATION")]
    }),
    ("4BHK @ ₹4.5 crore Mumbai", {
        "entities": [(0, 4, "CONFIGURATION"), (6, 16, "PRICE"),
                   (17, 23, "LOCATION")]
    }),
    
    # Developer abbreviations
    ("Brig. launches 3BHK in Kolkata", {
        "entities": [(0, 5, "DEVELOPER"), (16, 20, "CONFIGURATION"),
                   (24, 31, "LOCATION")]
    }),
    
    # Edge cases
    ("1RK studio flat Pune 25L", {
        "entities": [(0, 3, "CONFIGURATION"), (15, 19, "LOCATION"),
                   (20, 23, "PRICE")]
    }),
    ("₹1cr 2BHK ready-to-move Bangalore", {
        "entities": [(0, 4, "PRICE"), (5, 9, "CONFIGURATION"),
                   (23, 32, "LOCATION")]
    }),
    
    # Multi-word locations
    ("4BHK in Navi Mumbai by Rustomjee", {
        "entities": [(0, 4, "CONFIGURATION"), (8, 19, "LOCATION"),
                   (23, 32, "DEVELOPER")]
    }),
    
    # Complex price strings
    ("2BHK for 1 crore 20 lakhs Hyderabad", {
        "entities": [(0, 4, "CONFIGURATION"), (9, 24, "PRICE"),
                   (25, 34, "LOCATION")]
    }),
    
    # Partial matches
    ("Godrej premium residences 3BHK", {
        "entities": [(0, 6, "DEVELOPER"), (24, 28, "CONFIGURATION")]
    })
]

def train_spacy_ner(train_data, model='', iterations=100):
    """
    Train a SpaCy NER model with custom entities
    """
    nlp= spacy.load('en_core_web_sm')

    
    if 'ner' not in nlp.pipe_names:
        ner = nlp.add_pipe('ner', last=True)
    else:
        ner = nlp.get_pipe('ner')
    

    for _, annotations in train_data:
        for ent in annotations.get('entities', []):
            ner.add_label(ent[2])
    

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    

    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        
        for itn in range(iterations):
            random.shuffle(train_data)
            losses = {}
            
            # Batch training
            batches = minibatch(train_data, size=2)
            for batch in batches:
                examples = []
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    examples.append(example)
                
                # Update model
                nlp.update(examples, drop=0.5, losses=losses, sgd=optimizer)
            
            print(f'Iteration {itn+1}, Losses: {losses}')
    
    return nlp


trained_nlp = train_spacy_ner(train_data)


trained_nlp.to_disk('real_estate_ner_model')


test_sentences = [
    "3BHK apartment in Mumbai",
    "2.5 crores",
    "Prestige Group",
    "1800 sqft",
    "4BHK villa in Bangalore for ₹3.8 crores by Sobha",
    "Godrej launches 2BHK flats in Hyderabad at ₹1.2 crore",
    "2500 sqft plot in Chennai by Brigade for ₹2.5 crores",
    "1RK studio in Pune (₹25 lakhs)",
    "DLF's 3BHK + study in Gurgaon",
    "Contact us for more details."
]


print("\nEntity Predictions:")
for text in test_sentences:
    doc = trained_nlp(text)
    print(f'\nText: {text}')
    print('Entities:', [(ent.text, ent.label_) for ent in doc.ents])