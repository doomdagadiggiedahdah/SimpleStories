from datasets import load_dataset

def dump_stories_to_file(output_file):
    # Load the SimpleStories dataset
    ds = load_dataset("lennart-finke/SimpleStories")
    
    # Open the output file in write mode
    with open(output_file, 'w') as outfile:
        # Iterate through the dataset (we'll assume the 'story' field contains the story text)
        for example in ds['train']:  # Assuming the dataset has a 'train' split
            story_text = example.get("story", "")  # Adjust the key based on the dataset structure
            
            if story_text:
                # Write each story to the output file, followed by a newline for separation
                outfile.write(story_text.strip() + '\n\n')

if __name__ == "__main__":
    output_file = 'spm_stories.txt'  # Desired output file
    dump_stories_to_file(output_file)
