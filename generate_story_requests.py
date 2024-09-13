import json
from typing import List, Dict

# List of word counts to query
# WORD_COUNTS = [100, 200, 300, 400, 500]
WORD_COUNTS = [100] 
# Number of stories to generate for each word count
NUM_STORIES = 500
NUM_PARAGRAPHS = 5

def generate_story_prompt(count: int) -> str:
    """
    Generate a prompt for a children's story using the most common words.
    """
    prompt = f"""
    Take a deep breath, and think about this step by step. You have time.
    Think of the most common {count} number of words of the English language.
    Imagine you are a 3 year old dreaming. Tell me your dream from the 3rd person view.
    The dream should be {NUM_PARAGRAPHS} paragraphs long.
    If you use a verb, only use it in the infinitive form: "to..."
    """
    return prompt
    # Only use the following punctuation: ".!?,
    # Do not use possessives. Instead phrase as "the coat of Henry" use "Henry's coat" or instead of "Lilly's dad", use "the dad of Lilly".
    # Do not use compound words. Instead of "won't" use "will not", or instead of "shouldn't" use "should not".
    # Do not use future or past tense. Only current tense.

def create_story_request(count: int, run: int) -> Dict:
    """
    Create a request for the OpenAI API to generate a story.
    """
    return {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a children's story writer."},
            {"role": "user", "content": generate_story_prompt(count)}
        ],
        "temperature": 1,
        "metadata": {
            "word_count": count,
            "story_number": run + 1
        }
    }

def generate_requests() -> List[Dict]:
    """
    Generate a list of story requests for different word counts.
    """
    requests = []
    for count in WORD_COUNTS:
        for run in range(NUM_STORIES):
            requests.append(create_story_request(count, run))
    return requests

def save_requests(requests: List[Dict], filename: str):
    """
    Save the generated requests to a JSONL file.
    """
    with open(filename, 'w') as f:
        for request in requests:
            json.dump(request, f)
            f.write('\n')

def main():
    requests = generate_requests()
    save_requests(requests, "story_requests.jsonl")
    print(f"Generated {len(requests)} story requests and saved to story_requests.jsonl")

if __name__ == "__main__":
    main()