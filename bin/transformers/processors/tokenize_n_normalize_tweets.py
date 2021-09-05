#!/usr/bin/env python
# -*- coding: utf-8 -*-
# this script
#
# Author :  Ahmed A
# Las Update: 
#

from __future__ import print_function
from nltk.tokenize.casual import TweetTokenizer
import os
import sys
import io
import re


input_stream = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')


html_escape_table = {
    "&gt;": ">",
    "&lt;": "<",
    "&apos;": "'",
    "&quot;": '"',
    "&amp;": "&"
    }

def html_escape(text):
    """Produce entities within text."""
    return "".join(html_escape_table.get(c,c) for c in text)

def normalizeArabic(text):
	text = re.sub(r'[أإآ]',r'ا',text)
	#text = re.sub(r'[ى]',r'ي',text)
	#text = re.sub(r'ة',r'ه',text)
	return text

def tokenize_text(line):
	tweet = line.strip()
	# html escape
	tweet = ' '.join(TweetTokenizer().tokenize(tweet))
	# html escape
	tweet = html_escape(tweet)
	# Remove Arabic diacs
	tweet = re.sub(r'[ًٌٍَُِّْـ]+', r'', tweet)
	# Remove tags
	tweet = re.sub(r'(<[^>]+>)', r' ', tweet)
	# Convert URLS
	tweet = re.sub('http[s]*://t.co/[^ |\t]+', ' URL ', tweet)
	# Convert @users
	tweet = re.sub(r'(@[\w]+)', r' @USER ', tweet)
	# Clear latin and spaces
	tweet = re.sub(r'([@#\w]+)', r' \1 ', tweet)

	tweet = re.sub(r'([#_]+)', r' ', tweet)

	tweet = re.sub(' +', ' ', tweet)

	# tweet = ' '.join(TweetTokenizer().tokenize(tweet))

	tweet = normalizeArabic(tweet)

	tweet = tweet.strip()

	return tweet

def main():
	for line in input_stream:
		#print(line)
		tweet = line.strip()
		#html escape
		tweet = ' '.join(TweetTokenizer().tokenize(tweet))
		#html escape
		tweet = html_escape(tweet)
		# Remove Arabic diacs
		tweet = re.sub(r'[ًٌٍَُِّْـ]+',r'',tweet)
		#Remove tags
		tweet = re.sub(r'(<[^>]+>)',r' ',tweet)
		# Convert URLS
		tweet = re.sub('http[s]*://t.co/[^ |\t]+',' URL ',tweet)
		# Convert @users
		tweet = re.sub(r'(@[\w]+)',r' @USER ',tweet)
		# Clear latin and spaces
		tweet = re.sub(r'([@#\w]+)',r' \1 ',tweet)

		tweet = re.sub(r'([#_]+)',r' ',tweet)

		tweet = re.sub(' +',' ',tweet)

		#tweet = ' '.join(TweetTokenizer().tokenize(tweet))

		tweet = normalizeArabic(tweet)

		tweet = tweet.strip()
		# if(label==''):
		# 	label = 'NOF'
		print(tweet)


if __name__ == "__main__":
	main()

	






