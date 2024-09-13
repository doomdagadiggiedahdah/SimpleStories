import json
import re

# Function to clean and tokenize the text (lowercase, remove punctuation, and split by whitespace)
def clean_and_tokenize(text):
    text = re.sub(r'\W+', ' ', text.lower())  # Remove punctuation and make lowercase
    # text = text.lower()
    tokens = text.split()  # Split by whitespace
    return tokens

# Function to process the JSON content and extract unique words from the assistant's responses
def extract_unique_words_from_stories(json_file):
    unique_words = set()  # Set to store unique words

    # Open the file and handle the multiple JSON objects
    with open(json_file, 'r') as file:
        json_lines = file.read().splitlines()  # Split the file into lines

    for json_str in json_lines:
        if json_str.strip():  # Check if the line is not empty
            try:
                data = json.loads(json_str)  # Parse each line as a JSON object
                
                # Iterate through the messages and check for the 'assistant' role
                if isinstance(data, list):  # Handle the case where data is a list
                    for entry in data:
                        if isinstance(entry, dict):  # Ensure each entry is a dictionary
                            # Process only if the "choices" key exists in the entry
                            for choice in entry.get("choices", []):
                                if choice["message"]["role"] == "assistant":
                                    story_content = choice["message"]["content"]
                                    # Tokenize and add the words to the unique_words set
                                    tokens = clean_and_tokenize(story_content)
                                    unique_words.update(tokens)

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {e}")
    
    return unique_words

# Function to save unique words to a file
def save_unique_words_to_file(unique_words, output_file):
    with open(output_file, 'w') as file:
        for word in sorted(unique_words):  # Sort the words alphabetically for easy reading
            file.write(f"{word}\n")

# Main function to run the script
if __name__ == "__main__":
    # Path to the JSON file (replace this with the actual file path)
    json_file_path = 'story_results.jsonl'
    
    # Output file to save unique words
    output_file_path = 'unique_words.txt'

    # Extract unique words from stories
    unique_words = extract_unique_words_from_stories(json_file_path)

    # Save unique words to a text file
    save_unique_words_to_file(unique_words, output_file_path)

    # Output the total number of unique words
    print(f"Total unique words across all stories: {len(unique_words)}")
    print(f"Unique words have been saved to {output_file_path}")
