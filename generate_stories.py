import random
#import google.generativeai as genai
from openai import OpenAI
import hashlib
import json
#import threading
import os
import time

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

    prompt_template = (f"""\
        Write a short story (3-5 paragraphs) which only uses very simple words that a 3 year old child would likely understand. 
        The story should use the verb ”{verb}”, the noun ”{noun}” and the adjective ”{adjective}”. The story
        should have the following features: {features}
        Remember to only use simple words!""")
    prompt = prompt_template.format(**locals())
    return prompt


def generate_content(gen_model, prompt):
    match gen_model:
        case "openai":
            client = OpenAI()
            story = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return story.choices[0].message.content
    
def generate_tiny_story(gen_model, params: StoryParams):
    prompt = create_tiny_story_prompt(params)    
    story = generate_content(gen_model, prompt)
    return story
    
    #try:
        #story_text = story.text
    #except ValueError as e:
        #print('Error:', e)
        #return None
    #json_struct = {'instruction': {
        #'features': list(params.story_features), 
        #'prompt': prompt,
        #'words': [params.verb, params.noun, params.adjective]
        #},
        #'story': story_text,
        #'source': 'gemini-1.5-flash'}
    #return json_struct


def generate_and_log_tiny_stories(gen_model):#, thread_id: int = 0):    
    rand_verb = random.choice(test_verbs)
    rand_noun = random.choice(test_nouns)
    rand_adj  = random.choice(test_adjectives)
    rand_feat = random.choices(features)

    params = StoryParams(rand_verb, rand_noun, rand_adj, rand_feat)
    story = generate_tiny_story(gen_model, params)
    return story

    #for i in range(0, num_stories):
        #try:
            #story = generate_tiny_story(gen_model, params)
            #if story:
                #md5 = hashlib.md5(story['story'].encode()).hexdigest()
                #with open(f'spanish_stories/generated_story_{i}_{md5}.json', 'w') as fp:
                    #json.dump(story, fp)
            #else:
                #print('Error generating story', i)
        #except Exception as e:
            #print('Error generating story', i, e)
            #print('sleeping for 10 seconds at index', i)
            #time.sleep(10)
        #if i % 1000 == 0:
            #print('At index', i)


def main():
    #client = OpenAI()
    #completion = client.chat.completions.create(
        #model="gpt-4o",
        #messages=[
            #{"role": "user", "content": generate_and_log_tiny_stories}
        #]
    #)
    rand_verb = random.choice(test_verbs)
    rand_noun = random.choice(test_nouns)
    rand_adj  = random.choice(test_adjectives)
    rand_feat = random.choices(features)
    params = StoryParams(rand_verb, rand_noun, rand_adj, rand_feat)
    print(create_tiny_story_prompt(params))
    print(generate_and_log_tiny_stories("openai"))

    
if __name__ == '__main__':
    main()
    