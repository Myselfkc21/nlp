import spacy
import random
from spacy.util import minibatch
from spacy.training import Example
from word2number import w2n
import re
def normalize_price(price_text):
    price_text = price_text.lower()
    

    numeric_pattern = re.compile(r'\d{1,3}(?:,\d{3})*(?:\.\d+)?')
    numeric_match = numeric_pattern.search(price_text)
    
    if not numeric_match:
        return None
    numeric_str = numeric_match.group(0).replace(',', '')
    
    try:
 
        price_value = float(numeric_str)
        

        if re.search(r'l|lac|lakh|lakhs|hundred thousand|hundred thousands', price_text):
            price_value *= 100000
        elif re.search(r'cr|crore|crores|c', price_text):
            price_value *= 10000000
        elif re.search(r'm|million|millions|mil', price_text):
            price_value *= 1000000
        elif re.search(r'b|billion|billions', price_text):
            price_value *= 1000000000
        elif re.search(r't|trillion|trillions', price_text):
            price_value *= 1000000000000
        if price_value.is_integer():
            return int(price_value)
        return price_value
        
    except ValueError:
        return None
    


def fine_tune_spacy_ner(train_data, model_path='en_core_web_md', iterations=30):
    """
    Fine-tune an existing SpaCy NER model with custom entities
    Improved to capture full price entities
    """
    nlp = spacy.load(model_path, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])

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
                # print(f"Added new label: {ent[2]}")

    # Disable other pipeline components
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']

    # Fine-tuning
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.resume_training()

        for itn in range(iterations):
            random.shuffle(train_data)
            losses = {}

            # Batch training
            batches = minibatch(train_data, size=8)
            for batch in batches:
                examples = []
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    examples.append(example)

                # Update model
                nlp.update(examples, drop=0.25, losses=losses, sgd=optimizer)

            # print(f'Iteration {itn+1}, Losses: {losses}')

    return nlp

# def text_preprocessing(text):
    
# Expanded training data with more precise price entities
train_data = [
    # Base pattern examples with verified indices - adding DEVELOPER and AREA
    ("1BHK of 650 sq ft by Godrej Properties for 75 lacs in Noida", {
        "entities": [(0, 4, "CONFIGURATION"), (8, 17, "AREA"), (21, 37, "DEVELOPER"), (42, 49, "PRICE"), (53, 58, "LOCATION")]
    }),
    ("2BHK of 950 sq ft by DLF for 90 lacs in Delhi", {
        "entities": [(0, 4, "CONFIGURATION"), (8, 17, "AREA"), (21, 24, "DEVELOPER"), (29, 36, "PRICE"), (40, 45, "LOCATION")]
    }),
    ("3BHK with 1400 sq ft area by Lodha Group for 1.2 crore in Mumbai", {
        "entities": [(0, 4, "CONFIGURATION"), (10, 24, "AREA"), (28, 39, "DEVELOPER"), (44, 53, "PRICE"), (57, 63, "LOCATION")]
    }),
    
    # Price variations with correct indices - adding DEVELOPER and AREA
    ("1BHK flat of 600 sq ft by Prestige Group for ₹75 lacs in Noida", {
        "entities": [(0, 8, "CONFIGURATION"), (12, 21, "AREA"), (25, 39, "DEVELOPER"), (44, 52, "PRICE"), (56, 61, "LOCATION")]
    }),
    ("2BHK apartment of 1100 sq ft by Sobha for ₹1.5 crore in Gurgaon", {
        "entities": [(0, 14, "CONFIGURATION"), (18, 28, "AREA"), (32, 37, "DEVELOPER"), (42, 52, "PRICE"), (56, 63, "LOCATION")]
    }),
    ("3BHK duplex of 1650 sq ft by Oberoi Realty for ₹2.75 crores in Bangalore", {
        "entities": [(0, 11, "CONFIGURATION"), (15, 25, "AREA"), (29, 42, "DEVELOPER"), (47, 59, "PRICE"), (63, 72, "LOCATION")]
    }),
    
    # Different word order but same entity pattern - adding DEVELOPER and AREA
    ("For 85 lacs, 2BHK of 900 sq ft by Raheja in Chennai", {
        "entities": [(4, 11, "PRICE"), (13, 17, "CONFIGURATION"), (21, 30, "AREA"), (34, 40, "DEVELOPER"), (44, 51, "LOCATION")]
    }),
    ("In Hyderabad, 3BHK of 1500 sq ft by Brigade Group for 1.8 crore", {
        "entities": [(3, 12, "LOCATION"), (14, 18, "CONFIGURATION"), (22, 32, "AREA"), (36, 49, "DEVELOPER"), (54, 63, "PRICE")]
    }),
    ("Price is 60 lacs for 1BHK of 550 sq ft by Godrej in Pune", {
        "entities": [(9, 16, "PRICE"), (21, 25, "CONFIGURATION"), (29, 38, "AREA"), (42, 48, "DEVELOPER"), (52, 56, "LOCATION")]
    }),
    
    # Variations in price formatting - adding DEVELOPER and AREA
    ("4BHK villa of 2200 sq ft by L&T for 2 cr in Kolkata", {
        "entities": [(0, 9, "CONFIGURATION"), (13, 23, "AREA"), (27, 30, "DEVELOPER"), (35, 39, "PRICE"), (43, 50, "LOCATION")]
    }),
    ("2BHK flat of 950 sq ft by Tata Housing for 1cr in Jaipur", {
        "entities": [(0, 8, "CONFIGURATION"), (12, 21, "AREA"), (25, 36, "DEVELOPER"), (41, 44, "PRICE"), (48, 54, "LOCATION")]
    }),
    ("3BHK of 1300 sq ft by Prestige for 95L in Kochi", {
        "entities": [(0, 4, "CONFIGURATION"), (8, 18, "AREA"), (22, 30, "DEVELOPER"), (35, 38, "PRICE"), (42, 47, "LOCATION")]
    }),
    
    # With lakhs spelled differently - adding DEVELOPER and AREA
    ("1BHK of 600 sq ft by Puravankara for 45 lakhs in Ahmedabad", {
        "entities": [(0, 4, "CONFIGURATION"), (8, 17, "AREA"), (21, 32, "DEVELOPER"), (37, 45, "PRICE"), (49, 58, "LOCATION")]
    }),
    ("2BHK flat of 925 sq ft by K Raheja for 75 lakh in Surat", {
        "entities": [(0, 8, "CONFIGURATION"), (12, 21, "AREA"), (25, 33, "DEVELOPER"), (38, 45, "PRICE"), (49, 54, "LOCATION")]
    }),
    
    # With spacing variations but correct indices - adding DEVELOPER and AREA
    ("1BHK of 580 sq ft by DLF for75lacs in Noida", {
        "entities": [(0, 4, "CONFIGURATION"), (8, 17, "AREA"), (21, 24, "DEVELOPER"), (28, 34, "PRICE"), (38, 43, "LOCATION")]
    }),
    ("2BHK of 1000 sq ft by Brigade for 90lacs in Delhi", {
        "entities": [(0, 4, "CONFIGURATION"), (8, 18, "AREA"), (22, 29, "DEVELOPER"), (34, 40, "PRICE"), (44, 49, "LOCATION")]
    }),
    
    # Price with currency symbol together - adding DEVELOPER and AREA
    ("3BHK with 1450 sq ft area by Sobha for₹1.2crore in Mumbai", {
        "entities": [(0, 4, "CONFIGURATION"), (10, 24, "AREA"), (28, 33, "DEVELOPER"), (37, 46, "PRICE"), (50, 56, "LOCATION")]
    }),
    ("2BHK apartment of 975 sq ft by Lodha for₹85L in Chandigarh", {
        "entities": [(0, 14, "CONFIGURATION"), (18, 27, "AREA"), (31, 36, "DEVELOPER"), (40, 44, "PRICE"), (48, 58, "LOCATION")]
    }),
    
    # More location variations - adding DEVELOPER and AREA
    ("1BHK flat of 575 sq ft by Godrej for 50 lacs in South Delhi", {
        "entities": [(0, 8, "CONFIGURATION"), (12, 21, "AREA"), (25, 31, "DEVELOPER"), (36, 43, "PRICE"), (47, 58, "LOCATION")]
    }),
    ("3BHK duplex of 1700 sq ft by Prestige for 2.5 crores in East Bangalore", {
        "entities": [(0, 11, "CONFIGURATION"), (15, 25, "AREA"), (29, 37, "DEVELOPER"), (42, 52, "PRICE"), (56, 70, "LOCATION")]
    }),
    
    # More BHK variations - adding DEVELOPER and AREA
    ("1.5BHK of 700 sq ft by Shapoorji Pallonji for 60 lacs in Thane", {
        "entities": [(0, 6, "CONFIGURATION"), (10, 19, "AREA"), (23, 41, "DEVELOPER"), (46, 53, "PRICE"), (57, 62, "LOCATION")]
    }),
    ("2.5BHK apartment of 1200 sq ft by Hiranandani for 1.1 crore in Navi Mumbai", {
        "entities": [(0, 16, "CONFIGURATION"), (20, 30, "AREA"), (34, 44, "DEVELOPER"), (49, 58, "PRICE"), (62, 73, "LOCATION")]
    }),
    
    # With 'only' in price - adding DEVELOPER and AREA
    ("1BHK of 550 sq ft by Kolte Patil for only 40 lacs in Lucknow", {
        "entities": [(0, 4, "CONFIGURATION"), (8, 17, "AREA"), (21, 31, "DEVELOPER"), (41, 48, "PRICE"), (52, 59, "LOCATION")]
    }),
    ("2BHK with 1050 sq ft by Oberoi for just ₹95L in Indore", {
        "entities": [(0, 4, "CONFIGURATION"), (10, 20, "AREA"), (24, 30, "DEVELOPER"), (40, 44, "PRICE"), (48, 54, "LOCATION")]
    }),
    
    # Additional configuration formats - adding DEVELOPER and AREA
    ("1 BHK flat of 525 sq ft by Tata Housing for 50 lacs in Bhopal", {
        "entities": [(0, 9, "CONFIGURATION"), (13, 22, "AREA"), (26, 37, "DEVELOPER"), (42, 49, "PRICE"), (53, 59, "LOCATION")]
    }),
    ("2 Bedroom apartment of 1000 sq ft by Godrej for 85 lacs in Nagpur", {
        "entities": [(0, 19, "CONFIGURATION"), (23, 33, "AREA"), (37, 43, "DEVELOPER"), (48, 55, "PRICE"), (59, 65, "LOCATION")]
    }),
    
    # With "apartment" or "flat" mentioned - adding DEVELOPER and AREA
    ("1BHK flat of 600 sq ft by Mahindra Lifespaces for 55 lacs in Vadodara", {
        "entities": [(0, 8, "CONFIGURATION"), (12, 21, "AREA"), (25, 44, "DEVELOPER"), (49, 56, "PRICE"), (60, 68, "LOCATION")]
    }),
    ("2BHK apartment of 1150 sq ft by Brigade Group for 1.05 crore in Coimbatore", {
        "entities": [(0, 14, "CONFIGURATION"), (18, 28, "AREA"), (32, 45, "DEVELOPER"), (50, 60, "PRICE"), (64, 74, "LOCATION")]
    }),
    
    # With location first - adding DEVELOPER and AREA
    ("In Noida, 1BHK of 580 sq ft by DLF for 75 lacs", {
        "entities": [(3, 8, "LOCATION"), (10, 14, "CONFIGURATION"), (18, 27, "AREA"), (31, 34, "DEVELOPER"), (39, 46, "PRICE")]
    }),
    ("In Delhi, 2BHK flat of 950 sq ft by Tata Housing for 90 lacs", {
        "entities": [(3, 8, "LOCATION"), (10, 18, "CONFIGURATION"), (22, 31, "AREA"), (35, 46, "DEVELOPER"), (51, 58, "PRICE")]
    }),
    
    # Million and billion formats - adding DEVELOPER and AREA
    ("1BHK of 625 sq ft by Godrej for 7.5 million in Noida", {
        "entities": [(0, 4, "CONFIGURATION"), (8, 17, "AREA"), (21, 27, "DEVELOPER"), (32, 42, "PRICE"), (46, 51, "LOCATION")]
    }),
    ("3BHK penthouse of 2500 sq ft by Lodha for 0.2 billion in Mumbai", {
        "entities": [(0, 14, "CONFIGURATION"), (18, 28, "AREA"), (32, 37, "DEVELOPER"), (42, 52, "PRICE"), (56, 62, "LOCATION")]
    }),
    
    # Crore and lakh in same price - adding DEVELOPER and AREA
    ("2BHK of 1050 sq ft by Prestige for 1 crore 20 lakh in Pune", {
        "entities": [(0, 4, "CONFIGURATION"), (8, 18, "AREA"), (22, 30, "DEVELOPER"), (35, 50, "PRICE"), (54, 58, "LOCATION")]
    }),
    ("3BHK apartment of 1600 sq ft by Brigade for 2 crore 50 lacs in Hyderabad", {
        "entities": [(0, 14, "CONFIGURATION"), (18, 28, "AREA"), (32, 39, "DEVELOPER"), (44, 59, "PRICE"), (63, 72, "LOCATION")]
    }),
    
    # With rupees spelled out - adding DEVELOPER and AREA
    ("1BHK of 550 sq ft by Godrej Properties for Rupees 75 lacs in Noida", {
        "entities": [(0, 4, "CONFIGURATION"), (8, 17, "AREA"), (21, 37, "DEVELOPER"), (42, 56, "PRICE"), (60, 65, "LOCATION")]
    }),
    ("2BHK of 1000 sq ft by L&T Realty for Rs. 90 lacs in Delhi", {
        "entities": [(0, 4, "CONFIGURATION"), (8, 18, "AREA"), (22, 31, "DEVELOPER"), (36, 46, "PRICE"), (50, 55, "LOCATION")]
    }),
    
    # With decimal points in lacs - adding DEVELOPER and AREA
    ("1BHK flat of 600 sq ft by Sobha for 75.5 lacs in Noida", {
        "entities": [(0, 8, "CONFIGURATION"), (12, 21, "AREA"), (25, 30, "DEVELOPER"), (35, 44, "PRICE"), (48, 53, "LOCATION")]
    }),
    ("2BHK of 950 sq ft by Prestige Group for 90.25 lacs in Delhi", {
        "entities": [(0, 4, "CONFIGURATION"), (8, 17, "AREA"), (21, 35, "DEVELOPER"), (40, 50, "PRICE"), (54, 59, "LOCATION")]
    }),

    # Additional examples with different area formats
    ("1BHK of 60 sq meters by Tata Housing for 65 lacs in Thane", {
        "entities": [(0, 4, "CONFIGURATION"), (8, 19, "AREA"), (23, 34, "DEVELOPER"), (39, 46, "PRICE"), (50, 55, "LOCATION")]
    }),
    ("3BHK with 1800 square feet area by Oberoi for 2.1 crores in Mumbai", {
        "entities": [(0, 4, "CONFIGURATION"), (10, 31, "AREA"), (35, 41, "DEVELOPER"), (46, 56, "PRICE"), (60, 66, "LOCATION")]
    }),
    ("2BHK of 850-900 sq ft by Brigade for 75 lacs in Bangalore", {
        "entities": [(0, 4, "CONFIGURATION"), (8, 21, "AREA"), (25, 32, "DEVELOPER"), (37, 44, "PRICE"), (48, 57, "LOCATION")]
    }),
    ("4BHK villa with carpet area 2200 sq ft by DLF for 3.5cr in Gurgaon", {
        "entities": [(0, 9, "CONFIGURATION"), (15, 34, "AREA"), (38, 41, "DEVELOPER"), (46, 51, "PRICE"), (55, 62, "LOCATION")]
    }),

    # Examples with specific developers but no area
    ("3BHK by Lodha Group for 1.2 crore in Mumbai", {
        "entities": [(0, 4, "CONFIGURATION"), (8, 19, "DEVELOPER"), (24, 33, "PRICE"), (37, 43, "LOCATION")]
    }),
    ("1BHK flat by Godrej Properties for 48 lakhs in Thane", {
        "entities": [(0, 8, "CONFIGURATION"), (12, 28, "DEVELOPER"), (33, 41, "PRICE"), (45, 50, "LOCATION")]
    }),

    # Examples with area but no developer
    ("2BHK of 950 sq ft for 82 lacs in Andheri", {
        "entities": [(0, 4, "CONFIGURATION"), (8, 17, "AREA"), (22, 29, "PRICE"), (33, 40, "LOCATION")]
    }),
    ("3BHK apartment with 1500 sq ft area for 1.3 crore in Powai", {
        "entities": [(0, 14, "CONFIGURATION"), (20, 34, "AREA"), (39, 48, "PRICE"), (52, 57, "LOCATION")]
    }),

    # Examples with complex developer names
    ("2BHK of 900 sq ft by Shapoorji Pallonji Developers for 80 lacs in Pune", {
        "entities": [(0, 4, "CONFIGURATION"), (8, 17, "AREA"), (21, 49, "DEVELOPER"), (54, 61, "PRICE"), (65, 69, "LOCATION")]
    }),
    ("3BHK with 1600 sq ft by Mahindra Lifespaces Developers for 1.4 crore in Kandivali", {
        "entities": [(0, 4, "CONFIGURATION"), (10, 20, "AREA"), (24, 53, "DEVELOPER"), (58, 67, "PRICE"), (71, 80, "LOCATION")]
    }),

    # Examples with both super built-up and carpet areas
    ("2BHK with carpet area 750 sq ft (super built-up 950 sq ft) by DLF for 85L in Gurgaon", {
        "entities": [(0, 4, "CONFIGURATION"), (10, 52, "AREA"), (56, 59, "DEVELOPER"), (64, 67, "PRICE"), (71, 78, "LOCATION")]
    }),
    ("3BHK with built-up area 1400 sq ft (carpet 1100 sq ft) by Prestige for 1.25cr in Bangalore", {
        "entities": [(0, 4, "CONFIGURATION"), (10, 53, "AREA"), (57, 65, "DEVELOPER"), (70, 76, "PRICE"), (80, 89, "LOCATION")]
    })
]
# Train the model
trained_nlp = fine_tune_spacy_ner(train_data)

# Save the fine-tuned model
trained_nlp.to_disk('fine_tuned_real_estate_ner')

# Test sentences
test_sentences = [
   # Test Case 1: Standard format with all five entity types in typical order
    "Spacious 3BHK apartment of 1650 sq.ft by Godrej Properties for ₹1.75 crore in Bandra West",
    
    # Test Case 2: Price range with all entity types and complex area format
    "Looking for 2BHK between ₹80L - ₹1.2Cr with carpet area 950-1050 sq.ft by Lodha Group in Powai",
    
    # Test Case 3: Multiple entities of the same type
    "Compare: 1BHK (600 sq.ft) by DLF in Chembur for 60L vs 2BHK (950 sq.ft) by Raheja in Andheri for 1.4cr",
    
    # Test Case 4: Complex formats for all entity types
    "Duplex Penthouse (4BHK+servant room) with super built-up area 2800 sq.ft by Oberoi Realty for more than ₹8cr in Worli",
    
    # Test Case 5: Various qualifiers with international formats
    "Budget-friendly 1.5BHK below 45 lakhs with approximately 720 sq.ft by Tata Housing in Thane West near station"]

# Inference
# print(word2number)
print("\nEntity Predictions:")
entity_data = []

for idx, text in enumerate(test_sentences):
    doc = trained_nlp(text)
    
    print(f'\nText {idx+1}: {text}')
    
    # Create a numeric entry for this sentence
    sentence_data = {
        "id": idx + 1,
        "text": text,
        "entities": []
    }
    
    # Print all entities first
    print('All Entities:', [(ent.text, ent.label_) for ent in doc.ents])
    
    # Process each entity and add to the structured data
    for ent_idx, ent in enumerate(doc.ents):
        entity_info = {
            "id": ent_idx + 1,
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        }
        
        # Add normalized values for price entities
        if ent.label_ == 'PRICE':
            normalized_price = normalize_price(ent.text)
            if normalized_price is not None:
                print(f"Original: {ent.text}")
                print(f"Normalized: {normalized_price}")
                entity_info["normalized"] = normalized_price
        
        # Add normalized values for area entities (if you have this function)
        if ent.label_ == 'AREA' and 'normalize_area' in globals():
            normalized_area = normalize_area(ent.text)
            if normalized_area is not None:
                print(f"Original Area: {ent.text}")
                print(f"Normalized Area: {normalized_area}")
                entity_info["normalized"] = normalized_area
        
        # Add the entity to the sentence data
        sentence_data["entities"].append(entity_info)
    
    # Add the sentence data to the overall data array
    entity_data.append(sentence_data)

# Print the structured data array (you can also save it to JSON if needed)
print("\nStructured Data Array:")
import json
print(json.dumps(entity_data, indent=2))

# Alternatively, you can save this to a file
with open('entity_predictions.json', 'w') as f:
    json.dump(entity_data, f, indent=2)

print("\nData saved to entity_predictions.json")
    
   
