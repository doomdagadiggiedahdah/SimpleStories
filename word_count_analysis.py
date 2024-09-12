import os
from openai import OpenAI
import requests
import json
import time
from typing import List, Dict

# API endpoint (replace with actual endpoint)
# API_ENDPOINT = "https://api.example.com/word-frequency"

# List of word counts to query
WORD_COUNTS = [100, 500, 1000, 2000, 3000]

# Number of times to run each query
NUM_RUNS = 5

def get_common_words(count: int) -> str:
    """
    Make an API call to get the most common words.
    """
    prompt = f"Please generate the most common {count} number of words in English. Return only the words in the form 1. the 2. a 3. (etc.)."
    client = OpenAI()
    story = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return story.choices[0].message.content
    # params = {"count": count}
    # response = requests.get(API_ENDPOINT, params=params)
    # response.raise_for_status()
    # return response.json()["words"]

def run_queries() -> List[Dict]:
    """
    Run queries for each word count, multiple times.
    """
    results = []
    for count in WORD_COUNTS[:1]:
        for run in range(NUM_RUNS):
            try:
                words = get_common_words(count)
                result = {
                    "word_count": count,
                    "run_number": run + 1,
                    "words": words,
                    "timestamp": time.time()
                }
                results.append(result)
                print(f"Completed run {run + 1} for {count} words")
            except requests.RequestException as e:
                print(f"Error in run {run + 1} for {count} words: {e}")
    return results

def save_to_jsonl(data: List[Dict], filename: str):
    """
    Save results to a JSONL file.
    """
    with open(filename, 'w') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')

if __name__ == "__main__":
    results = run_queries()
    save_to_jsonl(results, "word_frequency_results.jsonl")
    print("Results saved to word_frequency_results.jsonl")