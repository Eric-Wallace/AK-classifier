# Arxiv Twitter Bot

This repo trains and runs a model to imitate the behavior of twitter users @_akhaliq and @arankomatsuzaki.

## Training the Model

### 1. Collect Tweets

We collect all tweets from @_akhaliq and @arankomatsuzaki and filter to the tweets that contain an arxiv paper link. We will use a twitter scraping API Scweet. It will take control of your Chrome browser and grab the tweets.
```
python scrape_arxiv_tweets.py
```
It stores the tweets in `data/arxiv_tweets.pkl`.

### 2. Find Associated Arxiv Papers 

For each tweet, we find the associated arxiv paper and extract its title, author list, and abstract. We also collect a set of "negative" examples, which are random NLP papers not tweeted by _akhaliq or arankomatsuzaki. TODO, only using NLP papers as the negatives.

`python convert_arxiv_papers_to_features.py`

### 3. Train a Binary Classifier.

We train a Huggingface model to classify tweets. We use the Galactica language model as our pre-trained model, as it is trained on large amounts of scientific text.

`python train_classifier.py`

Here is an ROC curve for the classifier, using a random holdout set of 10% of the tweets. 
![ROC Curve](./data/roc_plot.png =250x)

## Running the Model

## 4. Run The Classifier on New Papers and Tweet Them

Scrapes the daily dump of arxiv papers, scores them with the classifier, and then tweets out the papers with the highest scores.

`python run_classifier_on_latest_papers.py`
