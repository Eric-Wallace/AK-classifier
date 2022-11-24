from Scweet.scweet import scrape

ak_data = scrape(from_account = '_akhaliq', headless=False, display_type="Top", save_images=False, resume=False, filter_replies=True, since='2019-01-01')
text = [ak_data['Embedded_text'][i] for i in range(len(ak_data))]

pickle.dump(text, open('data/arxiv_tweets.pkl','wb'))

aran_data = scrape(from_account = 'arankomatsuzaki', headless=False, display_type="Top", save_images=False, resume=False, filter_replies=True, since='2019-01-01')
text.extend([aran_data['Embedded_text'][i] for i in range(len(aran_data))])


pickle.dump(text, open('data/arxiv_tweets.pkl','wb'))