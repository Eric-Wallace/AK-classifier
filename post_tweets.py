import pandas as pd
import tweepy 

# place the twitter_secrets file under <User>/anaconda3/Lib
from twitter_secrets import twitter_secrets as ts

consumer_key = ts.CONSUMER_KEY
consumer_secret = ts.CONSUMER_SECRET
access_token = ts.ACCESS_TOKEN
access_secret = ts.ACCESS_SECRET

auth=tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_secret)
api=tweepy.API(auth)

tweet='this is an automated test tweet using Python'
# image_path ='Test Images/ETH_price.png'

# Generate text tweet
api.update_status(tweet)
# api.update_with_media(image_path, tweet_text)