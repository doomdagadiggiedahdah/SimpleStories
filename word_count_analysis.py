import os
import json
import time
from typing import List, Dict
from collections import Counter
import matplotlib.pyplot as plt
from openai import OpenAI

# List of word counts to query
WORD_COUNTS = [100]  # You can add back 500, 1000, 2000, 3000 when ready
# Number of times to run each query
NUM_RUNS = 1

def get_common_words(count: int) -> str:
    """
    Make an API call to get the most common words.
    """
    prompt = f"""
    Please generate a children's story using only the most common {count} number of words in English. 
    Return only the words of the list.
    Use this format, or parsing will break: 
    1. the
    2. a
    3. ..."""
    client = OpenAI()
    story = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return story.choices[0].message.content

def process_words(word_list: str) -> List[str]:
    """
    Process the word list by stripping numbers and extra whitespace.
    """
    lines = word_list.split('\n')
    return [line.split('. ', 1)[-1].strip().lower() for line in lines if line.strip()]

def compare_results(results: List[Dict]) -> Dict:
    """
    Compare the words returned across different runs for each word count.
    """
    comparison = {}
    for count in WORD_COUNTS:
        count_results = [r for r in results if r['word_count'] == count]
        all_words = [set(process_words(r['words'])) for r in count_results]
        
        common_words = set.intersection(*all_words)
        unique_words = set.union(*all_words)
        
        comparison[count] = {
            'common_word_count': len(common_words),
            'common_word_percentage': len(common_words) / count * 100,
            'unique_word_count': len(unique_words),
            'unique_word_percentage': len(unique_words) / count * 100,
            'common_words': list(common_words),
        }
    
    return comparison

def create_graphs(comparison: Dict):
    """
    Create and save graphs based on the comparison results.
    """
    counts = list(comparison.keys())
    common_percentages = [data['common_word_percentage'] for data in comparison.values()]
    unique_percentages = [data['unique_word_percentage'] for data in comparison.values()]

    plt.figure(figsize=(12, 6))
    plt.plot(counts, common_percentages, marker='o', label='Common Words')
    plt.plot(counts, unique_percentages, marker='o', label='Unique Words')
    plt.xlabel('Word Count')
    plt.ylabel('Percentage')
    plt.title('Common and Unique Words Percentages')
    plt.legend()
    plt.grid(True)
    plt.savefig('word_comparison_graph.png')
    plt.close()

def generate_request_file(results: List[Dict], filename: str):
    """
    Generate a JSONL file with requests for the API processor script.
    """
    with open(filename, 'w') as f:
        for result in results:
            request = {
                "model": "text-embedding-3-small",
                "input": result['words'],
                "metadata": {
                    "word_count": result['word_count'],
                    "run_number": result['run_number']
                }
            }
            json.dump(request, f)
            f.write('\n')

def main():
    results = []
    for count in WORD_COUNTS:
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
            except Exception as e:
                print(f"Error in run {run + 1} for {count} words: {e}")

    comparison = compare_results(results)
    
    # Save results
    with open("word_frequency_results.jsonl", 'w') as f:
        for item in results:
            json.dump(item, f)
            f.write('\n')
    
    with open("word_comparison_results.jsonl", 'w') as f:
        json.dump(comparison, f)
        f.write('\n')
    
    # Print summary and create graphs
    create_graphs(comparison)
    for count, data in comparison.items():
        print(f"\nFor {count} words:")
        print(f"  Common words across all runs: {data['common_word_count']} ({data['common_word_percentage']:.2f}%)")
        print(f"  Unique words across all runs: {data['unique_word_count']} ({data['unique_word_percentage']:.2f}%)")
    
    # Generate request file
    generate_request_file(results, "word_frequency_requests.jsonl")
    
    print("\nResults saved to word_frequency_results.jsonl")
    print("Comparison results saved to word_comparison_results.jsonl")
    print("Graph saved as word_comparison_graph.png")
    print("API request file generated: word_frequency_requests.jsonl")

if __name__ == "__main__":
    main()