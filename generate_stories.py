import random
#import google.generativeai as genai
from openai import OpenAI
import hashlib
import json
#import threading
import os
import time
import anthropic



# TODO: finalize these sets
features = ["dialogue", "moral value", "plot twist", "foreshadowing", "conflict"] # "bad endings" is taken out. this list is taken directly from the paper
test_verbs = ["run", "jump", "play", "eat", "sleep", "drink", "sing", "draw", "dance", "swim"]
test_nouns = ["cat", "dog", "ball", "toy", "mom", "dad", "car", "book", "bed", "cup"]
test_adjectives = ["big", "small", "hot", "cold", "sweet", "loud", "soft", "fast", "slow", "red"]

class StoryParams:
    def __init__(self, verb, noun, adjective, story_features):
        self.verb = verb
        self.noun = noun
        self.adjective = adjective
        self.story_features = story_features


def create_tiny_story_prompt(params: StoryParams):
    noun, verb, adjective, features = params.noun, params.verb, params.adjective, params.story_features

    prompt_template = (f"""
        Write a short story (3-5 paragraphs) which only uses very simple words that a 3 year old child would likely understand. 
        The story should use the verb ”{verb}”, the noun ”{noun}” and the adjective ”{adjective}”. The story
        should have the following features: {features}
        Remember to only use simple words!""")
    prompt = prompt_template.format(**locals())
    return prompt


# TODO: add Gemma
def generate_content(gen_model, prompt):
    match gen_model:
        case "openai":
            api_key = os.getenv('OPENAI_TS_API_KEY')
            client = OpenAI(api_key=api_key)
            story = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return story.choices[0].message.content

        case "anthropic":
            client = anthropic.Anthropic()
            message = client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content[0].text


def generate_tiny_story(gen_model, params: StoryParams):
    prompt = create_tiny_story_prompt(params)
    id = hashlib.md5(prompt.encode()).hexdigest()
    
    try:
        story = generate_content(gen_model, prompt)
        return {
            'id': id,
            'story': story,
            'instruction': {
                'features': list(params.story_features),
                'prompt': prompt,
                'words': [params.verb, params.noun, params.adjective]
            },
            'model': gen_model
        }
    except ValueError as e:
        return {
            'instruction': {
                'features': list(params.story_features),
                'words': [params.verb, params.noun, params.adjective]
            },
            'error': str(e),
            'model': gen_model
        }


def generate_and_log_tiny_stories(gen_model, params: StoryParams, num_stories):#, thread_id: int = 0):    
    params = StoryParams(params.verb, params.noun, params.adjective, params.story_features)
    json_struct = generate_tiny_story(gen_model, params)

    filename = 'stories.json' if 'story' in json_struct else 'failed_stories.json'
    with open(filename, "a") as f:
        json.dump(json_struct, f)
        f.write('\n')
    return json_struct


def main(num_stories):
    for i in range(num_stories):
        # TODO: figure out the combos of all vna and features.
        rand_verb = random.choice(test_verbs)
        rand_noun = random.choice(test_nouns)
        rand_adj  = random.choice(test_adjectives)
        rand_feat = random.choices(features)
        params = StoryParams(rand_verb, rand_noun, rand_adj, rand_feat)

        print(create_tiny_story_prompt(params))
        data = generate_and_log_tiny_stories("openai", params, num_stories)
        print(f"openai: {data}")
        #data = generate_and_log_tiny_stories("anthropic", params, num_stories)
        #print(f"anthropic: {data}")

    
if __name__ == '__main__':
    main(num_stories=1)