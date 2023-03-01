import time
import openai
import pandas as pd
from tqdm import tqdm
from openai.error import RateLimitError

MY_API_KEY = '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'
#currently using lab payment
#LAB_KEY = 'org-t6ygn1B2vF9oML9g9HCjCT5f'
import collections
import re
import string

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

# Define a function that adds a delay to a Completion API call
def delayed_completion(delay_in_seconds: float = 1, **kwargs):
    """Delay a completion by a specified amount of time."""

    # Sleep for the delay
    time.sleep(delay_in_seconds)

    # Call the Completion API and return the result
    return openai.Completion.create(**kwargs)

ans_list = {}
openai.api_key = MY_API_KEY
#openai.api_key = os.getenv("OPENAI_API_KEY")


#prompt = 'Read the following the context, and answer the question:'




#start = time.time()

MODEL = 'text-davinci-003'
prompt = 'Read the following context carefully and choose one best option out of four to answer the question :'
#for i in tqdm(range(len(data['context']))):
name_list = ['train_100.jsonl','train_300.jsonl','train_500.jsonl',]
for name in name_list:
    count = 0
    count_total = 0
    data = pd.read_json(name, lines=True)
    for i in tqdm(range(1000,1200)):
        #print(i)
        text = ''
        ctx = 'context: ' + data['context'].iloc[i]
        #option = 'options:\n'
        option = 'options: \n'
        for j in range(4):
            option = option + f'option{j}: ' + data[f'option_{j}'].iloc[i] + '\n'
        question = 'question: ' + data['query'].iloc[i]
        text = prompt + '\n'  + question + '\n' + ctx + '\n' + option  + 'answer:'

        delay = True

        try:
            if delay:
                # Calculate the delay based on your rate limit
                rate_limit_per_minute = 40
                delay = 60.0 / rate_limit_per_minute
                #send every sample in a second

                response = delayed_completion(
                    delay_in_seconds=delay,
                    model=MODEL,
                    prompt=text,
                    temperature=0.8,
                    max_tokens=100,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
            else:
                response = openai.Completion.create(
                    model=MODEL,
                    prompt=text,
                    temperature=0.8,
                    max_tokens=100,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
        except openai.error.RateLimitError:
            rate_limit_per_minute = 20
            delay = 60.0 / rate_limit_per_minute
            # send every sample in a second

            response = delayed_completion(
                delay_in_seconds=delay,
                model="text-davinci-003",
                prompt=text,
                temperature=0.8,
                max_tokens=100,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
        ans = response["choices"][0]['text']
        #print(name, '\n' + text + '\n' + 'answer : ' + ans + '\n\n\n')
        #print(response["choices"][0]['text'])
        scores = []
        for k in range(4):
            option =  data[f'option_{k}'].iloc[i]
            scores.append(compute_f1(ans, option))

        if scores.index(max(scores)) == data['label'].iloc[i]:
            count += 1
        count_total += 1

    acc = float(count) / float(count_total) * 100.0
    print(f'For file {name}, there are {count_total} questions and the GPT-3 got {count} quesitons right. The accuracy is {acc}%')
    #end = time.time()

    #300-340: 45,47.5,42.5
    #400-440: 45, 52.5,52.5
    #550-600: 48, 48,56
    #700-800: 45, 50, 57.99
    #1000-1200：42.5 49 50.5