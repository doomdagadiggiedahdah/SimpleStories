import json
import nltk
from collections import Counter
from nltk.util import ngrams

# Download NLTK resources if you haven't done it yet
nltk.download('punkt')
nltk.download('punkt_tab')

# Function to extract n-grams from text
def extract_ngrams(text, n):
    tokens = nltk.word_tokenize(text.lower())  # Tokenize and convert to lowercase
    n_grams = ngrams(tokens, n)
    return n_grams

# Load the JSONL data
def load_jsonl(file_path):
    texts = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data = json.loads(line)
                
                # Check if data is a list
                if isinstance(data, list):
                    for item in data:
                        # Extract 'choices' section if it's a dictionary
                        if isinstance(item, dict):
                            choices = item.get('choices', [])
                            if choices and 'message' in choices[0]:
                                content = choices[0]['message'].get('content', '')
                                if content:
                                    texts.append(content)
                
                # If data is a dictionary
                elif isinstance(data, dict):
                    choices = data.get('choices', [])
                    if choices and 'message' in choices[0]:
                        content = choices[0]['message'].get('content', '')
                        if content:
                            texts.append(content)
                
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
                # Handle cases where the structure is unexpected or invalid JSON
                print(f"Error processing line: {e}")
                continue
    return texts

# Perform n-gram analysis
def ngram_analysis(texts, n):
    ngram_counter = Counter()
    for text in texts:
        ngram_counter.update(extract_ngrams(text, n))
    return ngram_counter.most_common()

# Main function
def main():
    # Path to your JSONL file
    file_path = 'story_results.jsonl'

    # Load text data from the JSONL file
    texts = load_jsonl(file_path)

    if not texts:
        print("No valid text data found in the JSONL file.")
        return

    # Get the most common 2-grams
    common_2grams = ngram_analysis(texts, 2)
    print("\nMost Common 2-grams:")
    for gram, count in common_2grams[:10]:  # Display top 10 for brevity
        print(f"{gram}: {count}")

    # Get the most common 3-grams
    common_3grams = ngram_analysis(texts, 3)
    print("\nMost Common 3-grams:")
    for gram, count in common_3grams[:10]:  # Display top 10 for brevity
        print(f"{gram}: {count}")

if __name__ == '__main__':
    main()
