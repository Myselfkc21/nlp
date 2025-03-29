import spacy
import random
from spacy.util import minibatch
from spacy.training import Example
from word2number import w2n

def fine_tune_spacy_ner(train_data, model_path='en_core_web_sm', iterations=50):
    """
    Fine-tune an existing SpaCy NER model with custom entities
    Improved to capture full price entities
    """
    nlp = spacy.load(model_path)

    # Ensure NER pipeline exists
    if 'ner' not in nlp.pipe_names:
        ner = nlp.add_pipe('ner', last=True)
    else:
        ner = nlp.get_pipe('ner')

    # Add custom labels
    for _, annotations in train_data:
        for ent in annotations.get('entities', []):
            if ent[2] not in ner.labels:
                ner.add_label(ent[2])
                print(f"Added new label: {ent[2]}")

    # Disable other pipeline components
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']

    # Fine-tuning
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.resume_training()

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

# Expanded training data with more precise price entities
train_data = [
    ("1BHK for 75 lacs in Noida", {
        "entities": [(0, 4, "CONFIGURATION"), (9, 17, "PRICE"), (21, 26, "LOCATION")]
    }),
    ("4BHK @ ₹4.5 crore Mumbai", {
        "entities": [(0, 4, "CONFIGURATION"), (6, 16, "PRICE"), (17, 23, "LOCATION")]
    }),
    ("2BHK for 1 crore 20 lakhs Hyderabad", {
        "entities": [(0, 4, "CONFIGURATION"), (9, 24, "PRICE"), (25, 34, "LOCATION")]
    }),
    ("₹1.2 crore 2BHK in Bangalore", {
        "entities": [(0, 10, "PRICE"), (11, 15, "CONFIGURATION"), (19, 28, "LOCATION")]
    }),
    ("85 lakhs 3BHK in Chennai", {
        "entities": [(0, 10, "PRICE"), (11, 15, "CONFIGURATION"), (19, 26, "LOCATION")]
    }),
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
]

# Train the model
trained_nlp = fine_tune_spacy_ner(train_data)

# Save the fine-tuned model
trained_nlp.to_disk('fine_tuned_real_estate_ner')

# Test sentences
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

# Inference
# print(word2number)
print("\nEntity Predictions:") 
for text in test_sentences:
    doc = trained_nlp(text)
    print(f'\nText: {text}')
    print('Entities:', [(ent.text, ent.label_) for ent in doc.ents])

            
           
               
    