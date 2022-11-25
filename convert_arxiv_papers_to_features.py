import urllib.request as libreq
import pickle
import feedparser
import re
from tqdm import tqdm
import json
import random

# first get "positive examples", i.e., arxiv papers posted by one of the AK's
tweets = pickle.load(open('data/tweetdata.pkl','rb'))
# filter to the text fo tweets that have at least a double digit number of likes
tweets = [tweets['Embedded_text'][i] for i in range(len(tweets)) if (len(tweets['Likes'][i]) > 1)]

regex = r'https:\/\/arxiv.org\/abs\/([0-9]+.[0-9]+)' # identifies arxiv links
positive_examples = set()
for tweet in tqdm(tweets):
    matches = list(re.finditer(regex, tweet, re.MULTILINE))
    if len(matches) > 0: # found an arxiv link
        if len(matches) == 1: # ignore tweets with two arxiv links
            arxiv_id = matches[0].group(1)
            # extract title, abstract, and authors using the arXiv API.
            with libreq.urlopen('http://export.arxiv.org/api/query?search_query=id:' + arxiv_id) as url:
                r = url.read()
                feed = feedparser.parse(r)
                input = 'Title: ' + feed['entries'][0]['title'].replace('\n','').replace('  ',' ')  + '\n'
                author_list = 'Authors: ' + ', '.join([x['name'] for x in feed['entries'][0]['authors']])
                input += author_list[0:250] + '\n' # trim those long author lists like BLOOM
                input += 'Abstract: ' + feed['entries'][0]['summary'].replace('\n',' ')
                positive_examples.add(input)
print('Number of Positive Examples', len(positive_examples))

# next get "negative examples", i.e., random papers from arxiv machine learning and 
# related communities, which were not posted by one of the AK's
negative_example_count = 5 * len(positive_examples) # we artifically make the class distribution more negative to simulate having lots of negatives, as it is in practice
negative_examples = set()
for page_id in tqdm(range(int(negative_example_count / 10))): # pages consist of 10 entries
    page_offset = 25 # start scraping from pages in the past, so the newest data can be used for testing
    #### TODO, only does NLP papers right now as negatives (from cs.CL)
    page = '&start=' + str(page_id * 10) + '&max_results=10'
    with libreq.urlopen('http://export.arxiv.org/api/query?search_query=cat:cs.CL&sortBy=submittedDate&sortOrder=descending' + page) as url:
        r = url.read()
        feed = feedparser.parse(r)
        for entry in feed['entries']:
            input = 'Title: ' + entry['title'].replace('\n','').replace('  ',' ') + '\n'
            author_list = 'Authors: ' + ', '.join([x['name'] for x in entry['authors']])
            input += author_list[0:250] + '\n' # trim those long author lists like BLOOM
            input += 'Abstract: ' + entry['summary'].replace('\n',' ')
            if input not in positive_examples: # ignore examples that have been tweeted by the AK's
                negative_examples.add(input)
            
positive_examples = list(positive_examples)
negative_examples = list(negative_examples)
random.shuffle(positive_examples)
random.shuffle(negative_examples)

# training data is random 90% of examples
with open('data/tweets_dataset_train.jsonl','w') as outf:
    for example in positive_examples[0:int(len(positive_examples)*0.9)]:
        json.dump({'text': example, 'label': 1}, outf)
        outf.write('\n')
    for example in negative_examples[0:int(len(negative_examples)*0.9)]:
        json.dump({'text': example, 'label': 0}, outf)
        outf.write('\n')

# validation data is the rest
with open('data/tweets_dataset_test.jsonl','w') as outf:
    for example in positive_examples[int(len(positive_examples)*0.9):]:
        json.dump({'text': example, 'label': 1}, outf)
        outf.write('\n')
    for example in negative_examples[int(len(negative_examples)*0.9):]:
        json.dump({'text': example, 'label': 0}, outf)
        outf.write('\n')