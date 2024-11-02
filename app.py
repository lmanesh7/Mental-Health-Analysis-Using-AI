import streamlit as st
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import time
import sys

pipe = pipeline("text-generation", model="Tianlin668/MentalBART")
tokenizer = AutoTokenizer.from_pretrained("Tianlin668/MentalBART")
model = AutoModelForSeq2SeqLM.from_pretrained("Tianlin668/MentalBART")

def type_text(text, delay=0.05):
    t = st.empty()
    for i in range(len(text) + 1):
        t.markdown("## %s..." % text[0:i])
   
        time.sleep(delay)
    print()  # Move to the next line after finishing

# Example usage
# type_text("Here's how to print text in Python with a typing animation effect.\n")
# type_text("\nStep-by-Step Guide:")
# type_text("1. To print a single line of text, use `print('your text here')`.")
# type_text("2. To print multiple lines, you can use multiple `print()` statements.")
# type_text("3. You can also use escape characters like `\\n` for a new line.\n")

# # Typing effect with variable formatting
# name = "Manesh"
# type_text(f"\nHello, {name}! Welcome to Python printing.")

# # Multi-line text example with typing effect
# type_text("""
# Here is a multi-line message:
# - You can list items
# - Print paragraphs
# - Or even create ASCII art
# """)

# # Example of a separator for cleaner output
# type_text("\n" + "=" * 40)
# type_text("End of the demonstration.")
# type_text("=" * 40)

prompt_start = "Consider this post: "
prompt_end  = " Question: how is the person feeling?"
text =  st.text_input("Enter Prompt")
prompt = prompt_start+str(text)+prompt_end
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=2048)
type_text(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])