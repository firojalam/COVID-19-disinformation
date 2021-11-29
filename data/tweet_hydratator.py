import configparser
import argparse
import pandas as pd
import tweepy
import os


'''
Simple Tweet Hydratator CLI tool

The script expects the following arguments:

--keys_file (-k) of format:

[KEYS]
API_Key=[YOUR API KEY]
API_Secret_Key=[YOUR API SECRET]
Access_Token=[YOUR ACCESS TOKEN]
Access_Token_Secret=[YOUR TOKEN SECRET]

--ids_file (-i) of tsv format

--ids_column (-c) string speficiying the column of the ids_file containing
  the Tweet Ids
'''


def hydratator(args):

    config = configparser.ConfigParser()
    config.read(args.keys_file)

    access_token = config['KEYS']['access_token']
    access_token_secret = config['KEYS']['access_token_secret']
    consumer_key = config['KEYS']['api_key']
    consumer_secret = config['KEYS']['api_secret_key']

    my_ids = pd.read_csv(args.ids_file, usecols=[args.ids_column],
                         sep='\t', dtype=str)
    id_list = my_ids.iloc[:, 0].values

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    tweet_txt = []
    for id in id_list:
        try:
            tweet = api.get_status(int(id))
            tweet_txt.append(tweet.text)
        except:
            tweet_txt.append('')

    out_file = os.path.splitext(args.ids_file)[0]
    df_final = pd.DataFrame({'id': id_list, 'tweet': tweet_txt})
    df_final.to_csv(f'{out_file}_hydratated_ids.csv')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--keys_file', type=str)
    parser.add_argument('-i', '--ids_file', type=str)
    parser.add_argument('-c', '--ids_column', type=str)
    args = parser.parse_args()
    hydratator(args)


if __name__ == '__main__':
    main()
