import json

def process_jsonl_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        # Iterate through each line in the input file
        for line in infile:
            try:
                # Parse the JSON object in the line
                json_objects = json.loads(line)
                
                # Extract the content of the "assistant" message
                for obj in json_objects:
                    if isinstance(obj, dict):  # Ensure it's a dictionary
                        if "choices" in obj:  # Check if it has choices
                            for choice in obj["choices"]:
                                if "message" in choice and choice["message"]["role"] == "assistant":
                                    # Extract the content of the assistant's message
                                    assistant_content = choice["message"].get("content", "")
                                    
                                    # Write the extracted content to the output file
                                    outfile.write(assistant_content.strip() + '\n\n')
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue

if __name__ == "__main__":
    input_file = 'story_results.jsonl'  # Your input file path
    output_file = 'spm_input.txt'  # Your output file path
    process_jsonl_file(input_file, output_file)
