# Arxiv Twitter Bot

A model that is trained to imitate the behavior of twitter users @_akhaliq and @arankomatsuzaki, to ease their burden in curating top arxiv papers.

## Training the Model

### 1. Collect Tweets

We collect all tweets from @_akhaliq and @arankomatsuzaki and filter to the tweet that have an arxiv paper link in them. We will use a (probably illegal) twitter scraping API Scweet. It will take control of your Chrome browser and grab the tweets.
```
pip install Scweet
python scrape_arxiv_tweets.py
```
It stores the tweets in `data/arxiv_tweets.pkl`.

### 2. Find Associated Arxiv Papers 

For each tweet, we find the associated arxiv paper and extract its title, author list, and abstract. We also collect a set of "negative" examples, which are random NLP papers not tweeted by _akhaliq or arankomatsuzaki. TODO, do more than just NLP papers as the negatives.

`python convert_arxiv_papers_to_features.py`

### 3. Train a Binary Classifier.

We train a Huggingface model to classify tweets. We use the Galactica language model from Facebook as our pre-trained model, as it is trained on large amounts of scientific text.

`python train_classifier.py`

Here is an ROC curve for the classifier, using a random holdout set of 10% of the tweets.

## 4. Run The Classifier and Post Tweets

Scrapes the daily dump of arxiv papers and classifies positive examples. Then it tweets them out using the official Twitter API.

`python run_classifier_on_latest_papers.py`