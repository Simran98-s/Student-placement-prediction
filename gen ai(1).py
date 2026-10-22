
# Install required libraries
# pip install transformers torch sentencepiece

from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load pretrained model and tokenizer
model_name = "t5-small"

tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Sample news article
article = """
Artificial Intelligence is rapidly transforming healthcare by improving
diagnosis, assisting doctors, and reducing medical errors.
Machine learning algorithms can analyze thousands of medical images
within seconds and detect diseases such as cancer more accurately.
Hospitals are also using AI-powered robots for surgery and patient care.
Experts believe that AI will continue to improve healthcare quality while
reducing costs and saving lives in the future.
"""

# Add T5 prefix
input_text = "summarize: " + article

# Tokenize input
inputs = tokenizer.encode(
    input_text,
    return_tensors="pt",
    max_length=512,
    truncation=True
)

# Generate summary
summary_ids = model.generate(
    inputs,
    max_length=50,
    min_length=20,
    length_penalty=2.0,
    num_beams=4,
    early_stopping=True
)

# Decode summary
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("Original Article:\n")
print(article)

print("\nGenerated Summary:\n")
print(summary)