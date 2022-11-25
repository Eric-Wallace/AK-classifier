from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from transformers import DataCollatorWithPadding
import numpy as np
import urllib.request as libreq
import feedparser
import pytz
from datetime import datetime
import scipy

# get latest tweets from today's arxiv feed
#### TODO, only does NLP papers right now.
page_id = 0
all_inputs = []
all_possible_tweets = []
todays_date = '2022-11-23'#datetime.now(pytz.timezone('US/Pacific')).strftime('%Y-%m-%d')
print(todays_date)
submittedDataIsToday = True
while submittedDataIsToday:
    page = '&start=' + str(page_id * 10) + '&max_results=10'
    with libreq.urlopen('http://export.arxiv.org/api/query?search_query=cat:cs.CL&sortBy=submittedDate&sortOrder=descending' + page) as url:
        r = url.read()
        feed = feedparser.parse(r)
        for entry in feed['entries']:
            if todays_date not in entry['published']:
                submittedDataIsToday = False
                break
            input = 'Title: ' + entry['title'].replace('\n','').replace('  ',' ') + '\n'
            author_list = 'Authors: ' + ', '.join([x['name'] for x in feed['entries'][0]['authors']])
            input += author_list[0:250] + '\n' # trim those long author lists like BLOOM
            input += 'Abstract: ' + entry['summary'].replace('\n',' ')
            all_possible_tweets.append(entry['title'].replace('\n','').replace('  ',' ') + '\n' + 'abs: ' + entry['id'].replace('v1',''))
            all_inputs.append({'text': input})
        page_id += 1

tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-125m")
tokenizer.pad_token_id = 1

model = AutoModelForSequenceClassification.from_pretrained("./results")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

dataset = Dataset.from_list(all_inputs)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=1024) #1300

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# fake training args
training_args = TrainingArguments(
    do_train = False,
    do_predict = True,
    per_device_eval_batch_size=1, # 1
    fp16=False,#################################
    output_dir='None'
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# get eval predictions
predictions = trainer.predict(tokenized_datasets)
positive_preds = scipy.special.softmax(predictions.predictions, axis=1)[:,1]

# sort by positive pred
all_inputs = [x for _, x in sorted(zip(positive_preds, all_inputs))]
positive_preds = sorted(positive_preds)

for example_id, example in enumerate(all_inputs):
    print(example)
    print(positive_preds[example_id])
    # if positive_preds[example_id] > 0.5:
        
    #     print(example)
    #     print(all_possible_tweets[example_id])