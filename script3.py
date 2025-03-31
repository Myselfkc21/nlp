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
    # Base pattern examples with verified indices
    ("1BHK for 75 lacs in Noida", {
        "entities": [(0, 4, "CONFIGURATION"), (9, 16, "PRICE"), (20, 25, "LOCATION")]
    }),
    ("2BHK for 90 lacs in Delhi", {
        "entities": [(0, 4, "CONFIGURATION"), (9, 16, "PRICE"), (20, 25, "LOCATION")]
    }),
    ("3BHK for 1.2 crore in Mumbai", {
        "entities": [(0, 4, "CONFIGURATION"), (9, 18, "PRICE"), (22, 28, "LOCATION")]
    }),
    
    # Price variations with correct indices
    ("1BHK for ₹75 lacs in Noida", {
        "entities": [(0, 4, "CONFIGURATION"), (9, 17, "PRICE"), (21, 26, "LOCATION")]
    }),
    ("2BHK for ₹1.5 crore in Gurgaon", {
        "entities": [(0, 4, "CONFIGURATION"), (9, 19, "PRICE"), (23, 30, "LOCATION")]
    }),
    ("3BHK for ₹2.75 crores in Bangalore", {
        "entities": [(0, 4, "CONFIGURATION"), (9, 21, "PRICE"), (25, 34, "LOCATION")]
    }),
    
    # Different word order but same entity pattern
    ("For 85 lacs, 2BHK in Chennai", {
        "entities": [(4, 11, "PRICE"), (13, 17, "CONFIGURATION"), (21, 28, "LOCATION")]
    }),
    ("In Hyderabad, 3BHK for 1.8 crore", {
        "entities": [(3, 12, "LOCATION"), (14, 18, "CONFIGURATION"), (23, 32, "PRICE")]
    }),
    ("Price is 60 lacs for 1BHK in Pune", {
        "entities": [(9, 16, "PRICE"), (21, 25, "CONFIGURATION"), (29, 33, "LOCATION")]
    }),
    
    # Variations in price formatting
    ("4BHK for 2 cr in Kolkata", {
        "entities": [(0, 4, "CONFIGURATION"), (9, 13, "PRICE"), (17, 24, "LOCATION")]
    }),
    ("2BHK for 1cr in Jaipur", {
        "entities": [(0, 4, "CONFIGURATION"), (9, 12, "PRICE"), (16, 22, "LOCATION")]
    }),
    ("3BHK for 95L in Kochi", {
        "entities": [(0, 4, "CONFIGURATION"), (9, 12, "PRICE"), (16, 21, "LOCATION")]
    }),
    
    # With lakhs spelled differently
    ("1BHK for 45 lakhs in Ahmedabad", {
        "entities": [(0, 4, "CONFIGURATION"), (9, 17, "PRICE"), (21, 30, "LOCATION")]
    }),
    ("2BHK for 75 lakh in Surat", {
        "entities": [(0, 4, "CONFIGURATION"), (9, 16, "PRICE"), (20, 25, "LOCATION")]
    }),
    
    # With spacing variations but correct indices
    ("1BHK for75lacs in Noida", {
        "entities": [(0, 4, "CONFIGURATION"), (8, 14, "PRICE"), (18, 23, "LOCATION")]
    }),
    ("2BHK for 90lacs in Delhi", {
        "entities": [(0, 4, "CONFIGURATION"), (9, 15, "PRICE"), (19, 24, "LOCATION")]
    }),
    
    # Price with currency symbol together
    ("3BHK for₹1.2crore in Mumbai", {
        "entities": [(0, 4, "CONFIGURATION"), (8, 17, "PRICE"), (21, 27, "LOCATION")]
    }),
    ("2BHK for₹85L in Chandigarh", {
        "entities": [(0, 4, "CONFIGURATION"), (8, 12, "PRICE"), (16, 26, "LOCATION")]
    }),
    
    # More location variations
    ("1BHK for 50 lacs in South Delhi", {
        "entities": [(0, 4, "CONFIGURATION"), (9, 16, "PRICE"), (20, 31, "LOCATION")]
    }),
    ("3BHK for 2.5 crores in East Bangalore", {
        "entities": [(0, 4, "CONFIGURATION"), (9, 19, "PRICE"), (23, 37, "LOCATION")]
    }),
    
    # More BHK variations
    ("1.5BHK for 60 lacs in Thane", {
        "entities": [(0, 6, "CONFIGURATION"), (11, 18, "PRICE"), (22, 27, "LOCATION")]
    }),
    ("2.5BHK for 1.1 crore in Navi Mumbai", {
        "entities": [(0, 6, "CONFIGURATION"), (11, 20, "PRICE"), (24, 35, "LOCATION")]
    }),
    
    # With 'only' in price
    ("1BHK for only 40 lacs in Lucknow", {
        "entities": [(0, 4, "CONFIGURATION"), (14, 21, "PRICE"), (25, 32, "LOCATION")]
    }),
    ("2BHK for just ₹95L in Indore", {
        "entities": [(0, 4, "CONFIGURATION"), (14, 18, "PRICE"), (22, 28, "LOCATION")]
    }),
    
    # Additional configuration formats
    ("1 BHK for 50 lacs in Bhopal", {
        "entities": [(0, 5, "CONFIGURATION"), (10, 17, "PRICE"), (21, 27, "LOCATION")]
    }),
    ("2 Bedroom for 85 lacs in Nagpur", {
        "entities": [(0, 9, "CONFIGURATION"), (14, 21, "PRICE"), (25, 31, "LOCATION")]
    }),
    
    # With "apartment" or "flat" mentioned
    ("1BHK flat for 55 lacs in Vadodara", {
        "entities": [(0, 4, "CONFIGURATION"), (14, 21, "PRICE"), (25, 33, "LOCATION")]
    }),
    ("2BHK apartment for 1.05 crore in Coimbatore", {
        "entities": [(0, 4, "CONFIGURATION"), (19, 29, "PRICE"), (33, 43, "LOCATION")]
    }),
    
    # With location first
    ("In Noida, 1BHK for 75 lacs", {
        "entities": [(3, 8, "LOCATION"), (10, 14, "CONFIGURATION"), (19, 26, "PRICE")]
    }),
    ("In Delhi, 2BHK for 90 lacs", {
        "entities": [(3, 8, "LOCATION"), (10, 14, "CONFIGURATION"), (19, 26, "PRICE")]
    }),
    
    # Million and billion formats
    ("1BHK for 7.5 million in Noida", {
        "entities": [(0, 4, "CONFIGURATION"), (9, 19, "PRICE"), (23, 28, "LOCATION")]
    }),
    ("3BHK for 0.2 billion in Mumbai", {
        "entities": [(0, 4, "CONFIGURATION"), (9, 19, "PRICE"), (23, 29, "LOCATION")]
    }),
    
    # Crore and lakh in same price
    ("2BHK for 1 crore 20 lakh in Pune", {
        "entities": [(0, 4, "CONFIGURATION"), (9, 24, "PRICE"), (28, 32, "LOCATION")]
    }),
    ("3BHK for 2 crore 50 lacs in Hyderabad", {
        "entities": [(0, 4, "CONFIGURATION"), (9, 24, "PRICE"), (28, 37, "LOCATION")]
    }),
    
    # With rupees spelled out
    ("1BHK for Rupees 75 lacs in Noida", {
        "entities": [(0, 4, "CONFIGURATION"), (9, 23, "PRICE"), (27, 32, "LOCATION")]
    }),
    ("2BHK for Rs. 90 lacs in Delhi", {
        "entities": [(0, 4, "CONFIGURATION"), (9, 19, "PRICE"), (23, 28, "LOCATION")]
    }),
    
    # With decimal points in lacs
    ("1BHK for 75.5 lacs in Noida", {
        "entities": [(0, 4, "CONFIGURATION"), (9, 18, "PRICE"), (22, 27, "LOCATION")]
    }),
    ("2BHK for 90.25 lacs in Delhi", {
        "entities": [(0, 4, "CONFIGURATION"), (9, 19, "PRICE"), (23, 28, "LOCATION")]
    }),
]
# Train the model
trained_nlp = fine_tune_spacy_ner(train_data)

# Save the fine-tuned model
trained_nlp.to_disk('fine_tuned_real_estate_ner')

# Test sentences
test_sentences = [
    # Real estate listings
    "Beautiful 2BHK for 87 lacs in Powai with modern amenities and 24/7 security.",
    "Spacious 3BHK apartment available for ₹1.35 crore in Bandra West, close to the sea.",
    "1BHK starting from ₹45L onwards in upcoming Godrej project in Thane.",
    
    # User queries
    "Is 1.2 cr for a 2BHK in Indiranagar Bangalore worth it?",
    "Can I get a decent 1BHK within 50-55 lacs budget in Andheri East?",
    "What's the average rate for 3BHK in and around Whitefield area?",
    
    # Chat conversations
    "Agent: We have a premium 3BHK available in Koramangala for 1.8 crore.\nBuyer: That's over my budget. Do you have any 2BHK options around 1.1-1.2 cr in the same area?",
    "I visited that 2BHK in HSR Layout yesterday, they're asking 95L but I think we can negotiate to 90 lacs.",
    "My friend purchased a 4BHK in Gurugram last month for 2.35 crore, but prices have gone up since then.",
    
    # Social media posts
    "Just booked my dream 3BHK in Hiranandani Gardens for ₹2.25cr! So excited to move in next year. #homeowner",
    "Housing prices are crazy these days. A friend paid 75L for a 1BHK in Kharadi. Is this normal?",
    "Anyone looking for property in Vashi? My uncle is selling his 2BHK for 90 lacs, great location near station.",
    
    # News articles
    "Property prices in South Mumbai see 15% increase, with average 2BHK now costing upwards of ₹3 crore.",
    "Affordable housing project launched in Greater Noida, offering 1BHK apartments at just 35 lacs.",
    "Luxury 4BHK penthouses in the new Trump Tower project in Worli start from ₹15 crore, targeting ultra-HNI buyers.",
    
    # Mixed with irrelevant information
    "My cousin who works at TCS just purchased a 2BHK for 72 lacs in Hinjewadi, says it's a good investment for the future.",
    "Despite the 12% interest rate, they decided to take a loan and buy the 3BHK in Electronic City for ₹1.15 crore rather than continue renting.",
    "The property tax for a 1000 sq ft 2BHK in Yelahanka comes to about ₹8000 per year, while the apartment itself costs around 65 lacs.",
    
    # Online forums/reviews
    "Pros: Great location, well-constructed 3BHK at 1.45 cr is reasonable for Malad West. Cons: Society maintenance is too high.",
    "Q: Budget 70L for 2BHK in Navi Mumbai, which areas should I look at? A: Try Kharghar or Seawoods.",
    "I purchased a 1BHK in this society last year for 52L, and the builder has now launched phase 2 starting at 58L for the same configuration.",
    
    # With multiple entities of the same type
    "Comparing real estate in different cities: 2BHK in Bangalore costs around 85-90L, while in Pune you can get the same for 65-70L, and in Mumbai it would be at least 1.2 cr.",
    "Property appreciation: My 3BHK in Gurgaon purchased at 1.4 crore in 2018 is now valued at 1.8 crore.",
    
    # With regional dialect/slang
    "Bhai, ek solid 2BHK in Andheri for 1.25 cr only, ekdum best deal hai market mein!",
    "The 1BHK in OMR for 45 lakhs is fully furnished machan, no need to spend extra.",
    
    # Phone conversations
    "Yes sir, we have multiple options - a 2BHK in Wakad for 68L, another one in Baner for 82 lacs, and if you're interested in 3BHK, we have one in Aundh for 1.25 cr.",
    "Madam, the possession date for your 1BHK in Thoraipakkam costing 52L is delayed by 3 months, we apologize for the inconvenience."

    "its a 2bhk for 75 lacs in surat, and 3bhk for 1.4 cr in kochi",
]

# Inference
# print(word2number)
print("\nEntity Predictions:") 
for text in test_sentences:
    doc = trained_nlp(text)
    print(f'\nText: {text}')
    print('Entities:', [(ent.text, ent.label_) for ent in doc.ents])
    for ent in doc.ents:
     if ent.label_ == 'PRICE':
        normalized_price = normalize_price(ent.text)
        if normalized_price is not None:
            print(f"Original: {ent.text}")
            print(f"Normalized: {normalized_price}")
    
   
