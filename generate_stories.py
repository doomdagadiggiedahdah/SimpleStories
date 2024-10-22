import random
import textwrap
import itertools
import json
import os
from openai import OpenAI
import hashlib
import time
import textstat
import anthropic
import concurrent.futures
from tqdm import tqdm
from datetime import datetime
from datasets import load_dataset
from perturbations import perturbations

MAX_STORIES_PER_COMPLETION = 40
END_STRING = "THE END."

class RateLimitException(Exception):
    pass

themes = {"en": ["Friendship", "Courage", "Coming of age", "Kindness", "Adventure", "Imagination", "Family", "Perseverance", "Curiosity", "Honesty", "Romance", "Teamwork", "Responsibility", "Strategy", "Magic", "Discovery", "Betrayal", "Deception", "Generosity", "Creativity", "Self-Acceptance", "Helping Others", "Hardship", "Agency", "Power", "Revenge", "Independence", "Problem-Solving", "Resourcefulness", "Long-Term Thinking", "Optimism", "Humor", "Love", "The Five Senses", "Tradition", "Innovation", "Hope", "Dreams", "Belonging", "Travel", "Overcoming", "Trust", "Morality", "Happiness", "Consciousness", "Failure", "Conflict", "Cooperation", "Growth", "Loss", "Celebration", "Transformation", "Scheming", "Challenge", "Planning", "Wonder", "Surprises", "Conscience", "Intelligence", "Logic", "Resilience"]}["en"]
topics = {"en": ["talking animals", "fantasy worlds", "time travel", "space exploration", "mystical creatures", "underwater adventures", "dinosaurs", "pirates", "superheroes", "fairy tales", "outer space", "hidden treasures", "magical lands", "enchanted forests", "secret societies", "robots and technology", "sports", "school life", "holidays", "cultural traditions", "magical objects", "lost civilizations", "subterranean worlds", "bygone eras", "invisibility", "giant creatures", "miniature worlds", "alien encounters", "haunted places", "shape-shifting", "island adventures", "unusual vehicles", "undercover missions", "dream worlds", "virtual worlds", "riddles", "sibling rivalry", "treasure hunts", "snowy adventures", "seasonal changes", "mysterious maps", "royal kingdoms", "living objects", "gardens", "lost cities", "the arts", "the sky"]}["en"]
styles = {"en": ["whimsical", "playful", "epic", "fairy tale-like", "modern", "classic", "lyric", "mythological", "lighthearted", "adventurous", "heartwarming", "humorous", "mystical", "action-packed", "fable-like", "surreal", "philosophical", "melancholic", "noir", "romantic", "tragic", "minimalist", "suspenseful"]}["en"]
features = {"en": ["dialogue", "a moral lesson", "a twist ending", "foreshadowing", "irony", "inner monologue", "symbolism", "a MacGuffin", "a non-linear timeline", "a flashback", "a nested structure", "a story within a story", "multiple perspectives", "Checkhov's gun", "the fourth wall", "a cliffhanger", "an anti-hero", "juxtaposition", "climactic structure"]}["en"]
grammars = {"en": ["present tense", "past tense", "future tense", "progressive aspect", "perfect aspect", "passive voice", "conditional mood", "imperative mood", "indicative mood", "relative clauses", "prepositional phrases", "indirect speech", "exclamative sentences", "comparative forms", "superlative forms", "subordinate clauses", "ellipsis", "anaphora", "cataphora", "wh-questions", "yes-no questions", "gerunds", "participle phrases", "inverted sentences", "non-finite clauses", "determiners", "quantifiers", "adjective order", "parallel structure", "discourse markers", "appositive phrases"]}["en"]

def load_personas():
    try:
        dataset = load_dataset("proj-persona/PersonaHub", "persona")
        personas = dataset['train']
        max_personas = 1
        selected_personas = random.sample(list(personas), max_personas)[0]['persona']
        return selected_personas
    except Exception as e:
        print(f"Error loading personas: {e}")
        return []

def get_random_params():
    grammar = random.choice(grammars)
    perturbation = random.choice(perturbations)
    if random.random() < 0.5:
        grammar = ""
    if random.random() < 0.5:
        perturbation = ""
    return {
        "theme": random.choice(themes),
        "topic": random.choice(topics),
        "style": random.choice(styles),
        "feature": random.choice(features),
        "grammar": grammar,
        "personas": load_personas(),
        "perturbation": perturbation,
        "num_paragraphs": random.randint(1, 9),
    }

system_prompt = textwrap.dedent(f"""\
    You are a children's storyteller specializing in creating engaging stories for young children aged 3-4 years old.
    You will be given elements to build your story around. Use these to the best of your ability to generate a story using experiences, details, and vocabulary suitable for 3-4 year olds. 
    But, if they are not suitable for children, creatively reinterpret them to be suitable to the story.
    If you need to use proper names, make them from space-separated common words. 
    Either don't give characters a name, or select from Mia, Alex, Jean, Samuel, Lily, Leo, Jose, Kim, Alice, Lena, Rita, Emmanuel, Anne, Peter, Maria or Luis. 
    Complex story structure is great, but please remember to only use simple words.
    Similarly, try not to start the story with "Once upon a time,". Other beginnings are preferred.
    Please do your best, even if you have to remove some details (if you have to remove details, don't mention it).
    Only return the story. 
""")


def create_simple_story_prompt(params):
    num_stories_per_completion = MAX_STORIES_PER_COMPLETION // max(3, params['num_paragraphs'])
    #num_stories_per_completion = 5

    singular = params['num_paragraphs'] == 1
    template_singular = f"Write a short story ({params['num_paragraphs']} paragraphs, with ~85 words per paragraph) using very basic words that a preschool child could understand. \nThe story "
    template_plural = textwrap.dedent(f"""\
        Write {num_stories_per_completion} short stories ({params['num_paragraphs']} paragraph{'' if singular else 's'} each) using very basic words that a young child could understand.
        Do not number each story or write a headline. Make the stories diverse by fully exploring the theme, but each story should be self-contained. Separate the stories by putting {END_STRING} in between.
        Each story 
        """)

    template = textwrap.dedent("""\
        should be about {theme}, include {topic}, be {style} in its writing style and ideally feature {feature}, 
        and include characters with the following personalities: {personas}. 
        Make sure to use character details to steer the story; again, if they are not suitable for children, creatively reinterpret them to be suitable to the story.
        {perturbation} {grammar} 
        """)

    if singular:
        template = template_singular + template
    else:
        template = template_plural + template

    params = params.copy()
    if params['grammar']:
        params['grammar'] = f" The most important thing is to write an engaging easy story, but where it makes sense, demonstrate the use of {params['grammar']}."

    prompt = template.format(**params)
    return prompt, num_stories_per_completion

def generate_content(gen_model, prompt):
    assert "gpt" in gen_model or "claude" in gen_model, "Invalid model name"
    if "gpt" in gen_model:  # OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY_SIMPLESTORIES"])
        completion = client.chat.completions.create(
            model=gen_model,
            top_p=0.7,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        completion = completion.choices[0].message.content
    elif "claude" in gen_model:  # Anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY_SIMPLESTORIES"])
        completion = client.messages.create(
            model=gen_model,
            max_tokens=min(1024*MAX_STORIES_PER_COMPLETION, 8192),
            top_p=0.7,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        completion = completion.content[0].text

    return completion

def process_completion(gen_model, completion, params, expected_num_stories=None, 
                       spache_score=None, flesch_kincaid_score=None, flesch_reading_ease=None):
    id = hashlib.md5(completion.encode()).hexdigest()
    stories = [x.strip() for x in completion.split(END_STRING) if len(x.strip()) > 1]
    table = str.maketrans({
                "\u201d": "\"",
                "\u201c": "\"",
                "\u2019": "'",
                "\u2018": "'",
                "\u2014": "-",
                "\u2026": "..."
    })
    stories = [x.translate(table) for x in stories]
    if (len(stories) != expected_num_stories and expected_num_stories):
        print(f"Completion did not include expected number of stories, actual={len(stories)} != expected={expected_num_stories}\nend of completion: {completion[-100:]}")
    return [{
        'generation_id': id + "-" + str(k),
        'story': story,
        'model': gen_model,
        'num_stories_in_completion': len(stories),
        "expected_num_stories_in_completion": expected_num_stories,
        'spache_score': spache_score,  # Add Spache score
        'flesch_kincaid_score': flesch_kincaid_score,  # Add Flesch-Kincaid score
        'flesch_reading_ease': flesch_reading_ease,  # Add Flesch Reading Ease score
        **params
    } for k, story in enumerate(stories)] 

def evaluate_story(story):
    spache_score = textstat.spache_readability(story)
    flesch_kincaid_score = textstat.flesch_kincaid_grade(story)
    flesch_reading_ease = textstat.flesch_reading_ease(story)
    return spache_score, flesch_kincaid_score, flesch_reading_ease  

def generate_simple_story(gen_model, params: dict):
    prompt, expected_num_stories = create_simple_story_prompt(params.copy())
    
    try:
        completion = generate_content(gen_model, prompt)
        spache_score, flesch_kincaid_score, flesch_reading_ease = evaluate_story(completion)

        return process_completion(gen_model, completion, params, expected_num_stories,
                                          spache_score, flesch_kincaid_score, flesch_reading_ease)
    except Exception as e:
        # TODO Implement Rate Limit Logic for different APIs
        raise RateLimitException(e)

def generate_and_log_simple_stories(gen_model: str, params: dict, formatted_time: str):
    json_struct = generate_simple_story(gen_model, params)
    lines = [json.dumps(item) for item in json_struct if 'story' in item]
    
    filename = f'data/stories-{gen_model}-{formatted_time}.jsonl'
    with open(filename, "a") as f:
        f.write("\n".join(lines) + "\n")
    for line in json_struct:
        print("##############################")
        print(line['story'])
        print(line['personas'])
        print(line['perturbation'])
        spache_score, fk_score, fl_reading = evaluate_story(line['story'])
        print(f"{spache_score , fk_score, fl_reading = }")
        print()

def worker_thread(gen_model: str, params: dict, formatted_time: str):
    while True:
        try:
            return generate_and_log_simple_stories(gen_model, params, formatted_time)
        except RateLimitException as e:
            print(f"Rate limit hit: {e}, backing off for 5 seconds...")
            time.sleep(5)
            continue

def main(num_completions: int, num_threads: int = 20, model = "gpt-4o-mini"):
    print(f"Number of Parameter Combinations: {len(themes)*len(topics)*len(styles)*len(features)}")

    if not os.path.exists("data"):
        os.makedirs("data")
    now = datetime.now()
    formatted_time = now.strftime('%Y-%m-%d-%H-%M-%S')

    # Generate regular simple stories
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_story = {
            executor.submit(worker_thread, model, get_random_params(), formatted_time): i for i in range(num_completions)
        }

        for future in tqdm(concurrent.futures.as_completed(future_to_story), total=num_completions, desc="Generating stories"):
            try:
                data = future.result()
            except Exception as e:
                print(f"Story generation failed with exception: {e}")

# Reference models: ["gpt-4o", "gpt-4o-mini", "claude-sonnet-3.5-20240620"]
if __name__ == '__main__':
    NUM_COMPLETIONS = 200
    main(NUM_COMPLETIONS, num_threads=10, model="gpt-4o-mini")
    main(NUM_COMPLETIONS, num_threads=10, model="claude-3-5-sonnet-20240620")
