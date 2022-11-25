from Scweet.scweet import scrape
import pickle
import pandas as pd

ak_data = scrape(from_account = '_akhaliq', headless=False, display_type="Top", save_images=False, resume=False, filter_replies=True, since='2019-01-01')
aran_data = scrape(from_account = 'arankomatsuzaki', headless=False, display_type="Top", save_images=False, resume=False, filter_replies=True, since='2019-04-01') #04-01

pickle.dump(pd.concat([ak_data + aran_data]), open('data/tweetdata.pkl','wb'))

# pickle.dump(text, open('data/arxiv_tweets.pkl','wb'))

# text = [aran_data['Embedded_text'][i] for i in range(len(aran_data))]
# pickle.dump(text, open('data/arxiv_tweets2.pkl','wb'))
# pickle.dump(aran_data, open('rawdata2.pkl','wb'))