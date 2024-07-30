import random
import google.generativeai as genai
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

# not sure if the Params Generator will still be used, I think we want something more deterministic.

class StoryParamsGenerator:
    def __init__(self, accepted_verbs, accepted_nouns, accepted_adjectives, story_features_list, story_features_cum_weights):
        self.accepted_verbs = accepted_verbs
        self.accepted_nouns = accepted_nouns
        self.accepted_adjectives = accepted_adjectives

        self.story_features_list = story_features_list
        self.story_features_cum_weights = story_features_cum_weights

    def generate(self) -> StoryParams:
        verb = random.choice(self.accepted_verbs)
        noun = random.choice(self.accepted_nouns)
        adjective = random.choice(self.accepted_adjectives)

        story_features = random.choices(self.story_features_list, cum_weights=self.story_features_cum_weights)[0]

        return StoryParams(verb, noun, adjective, story_features)
    

def create_tiny_story_prompt(params: StoryParams):
    noun, verb, adjective = params.noun, params.verb, params.adjective

    story_features_combined = [features[f] for f in params.story_features]
    if len(story_features_combined) == 1:
        story_features_combined = story_features_combined[0]
    else:
        story_features_combined = ', '.join(story_features_combined[:-1]) + ' y ' + story_features_combined[-1]
    prompt_template = (
        prompt = f"""Write a short story (3-5 paragraphs) which only uses very simple words that a 3 year old child would likely understand. 
                    The story should use the verb ”{verb}”, the noun ”{noun}” and the adjective ”{adjective}”. The story
                    should have the following features: {story_features_combined}
                    Remember to only use simple words!""")
    prompt = prompt_template.format(**locals())
    return prompt

    
def generate_tiny_story(gen_model):
    rand_verb = random.choice(test_verbs)
    rand_noun = random.choice(test_nouns)
    rand_adj  = random.choice(test_adjectives)
    rand_feat = random.choices(features)

    #params = StoryParamsGenerator()
    params = StoryParams(rand_verb, rand_noun, rand_adj, rand_feat)
    prompt = create_tiny_story_prompt(params)    
    story = gen_model.generate_content(prompt)
    
    try:
        story_text = story.text
    except ValueError as e:
        print('Error:', e)
        return None
    json_struct = {'instruction': {
        'features': list(params.story_features), 
        'prompt': prompt,
        'words': [params.verb, params.noun, params.adjective]
        },
        'story': story_text,
        'source': 'gemini-1.5-flash'}
    return json_struct


def generate_and_log_tiny_stories(gen_model, params_generator: StoryParamsGenerator, num_stories: int):#, thread_id: int = 0):    
    for i in range(0, num_stories):
        try:
            story = generate_tiny_story(gen_model, params_generator)
            if story:
                md5 = hashlib.md5(story['story'].encode()).hexdigest()
                with open(f'spanish_stories/generated_story_{thread_id}_{i}_{md5}.json', 'w') as fp:
                    json.dump(story, fp)
            else:
                print('Error generating story', i)
        except Exception as e:
            print('Error generating story', i, e)
            print('sleeping for 10 seconds at index', i)
            time.sleep(10)
        if i % 1000 == 0:
            print('At index', i)



#def main():
    #GEMINI_API_KEY = open(os.path.expanduser('~/.gemini_api_key'), 'r').read().strip()
    #genai.configure(api_key=GEMINI_API_KEY)
    #safety_settings = [{'category': c, 'threshold': 'BLOCK_NONE'} for c in
                    #['HARM_CATEGORY_DANGEROUS', 'HARM_CATEGORY_HARASSMENT', 'HARM_CATEGORY_HATE_SPEECH',
                     #'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'HARM_CATEGORY_DANGEROUS_CONTENT']]
    #flash = genai.GenerativeModel('gemini-1.5-flash', safety_settings=safety_settings)

    #story_params_generator = pickle.load(open('story_params_generator.pkl', 'rb'))
    #threads = []
    #NUM_DESIRED_STORIES = 2500000
    #NUM_THREADS = 40
    #for i in range(NUM_THREADS):
        #thread = threading.Thread(target=generate_and_log_tiny_stories, args=(flash, story_params_generator, NUM_DESIRED_STORIES // NUM_THREADS, i))
        #threads.append(thread)
        #thread.start()

    #for thread in threads:
        #thread.join()

def main():
    #client = OpenAI()
    #completion = client.chat.completions.create(
        #model="gpt-4o",
        #messages=[
            #{"role": "user", "content": generate_and_log_tiny_stories}
        #]
    #)



    
if __name__ == '__main__':
    main()
    