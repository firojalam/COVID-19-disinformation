# COVID-19 Disinformation Twitter Dataset (COVID-19 Disinfo dataset)

This repository contains a dataset and experimental scripts associated the work **["Fighting the COVID-19 Infodemic: Modeling the Perspective of Journalists, Fact-Checkers, Social Media Platforms, Policy Makers, and the Society"](https://arxiv.org/abs/2005.00033)**.  The **COVID-19 Disinfo dataset** consisting of tweets annotated with fine-grained labels related to disinformation about COVID-19. The labels answer seven different questions that are of interests to journalists, fact-checkers, social media platforms, policymakers, and society as a whole. There are annotations for Arabic, Bulgarian, Dutch and English.



__Table of contents:__
- [Contents of the Distribution](#contents-of-the-distribution)
  - [Directory Structure](#directory-structure)
  - [Examples](#examples)
  - [Statistics](#statistics)
- [Questions with Labels](#questions-with-labels)
- [List of Versions](#list-of-versions)
- [Download](#download)
- [Experiments](bin)
- [Publication](#publication)
- [Credits](#credits)
- [Licensing](#licensing)
- [Contact](#credits)
- [Acknowledgment](#acknowledgment)

## Contents of the Distribution
=============================================== <br/>

### Directory Structure
=======================<br/>
The directory contains the following files and sub-directories:


1. The following directories contains different data splits (train/dev/test) for both binary and multiclass. Each file is tab-separated, consists of tweet_id, and labels for Q1-7. For privacy concern we are not able to release tweet text and associated json objects.
  * **data/arabic/**
  * **data/bulgarian/**
  * **data/dutch/**
  * **data/english/**  
  * **data/multilang/:** This directory contains multilingual data (tweets from all languages are combined in different splits for both binary and multiclass settings).
2. data/LICENSE_CC_BY_NC_SA_4.0.txt: license information
3. **[bin/](bin)** Please see readme for details
4. Readme.md this file


### Examples
============<br/>

***Please don't take hydroxychloroquine (Plaquenil) plus Azithromycin for #COVID19 UNLESS your doctor prescribes it. Both drugs affect the QT interval of your heart and can lead to arrhythmias and sudden death, especially if you are taking other meds or have a heart condition.*** <br/>
Labels:
<ol type="A">
	<li>Q1: Yes;</li>
	<li>Q2: NO: probably contains no false info</li>
	<li>Q3: YES: definitely of interest</li>
	<li>Q4: NO: probably not harmful</li>
	<li>Q5: YES:very-urgent</li>
	<li>Q6: NO:not-harmful</li>
	<li>Q7:	NO: YES:discusses_cure</li>
</ol>


***BREAKING: @MBuhari’s Chief Of Staff, Abba Kyari, Reportedly Sick, Suspected Of Contracting #Coronavirus | Sahara Reporters A top government source told SR on Monday that Kyari has been seriously “down” since returning from a trip abroad. READ MORE: https://t.co/Acy5NcbMzQ https://t.co/kStp4cmFlr.***  <br/>
*Labels:*
<ol type="A">
	<li>Q1: Yes; </li>
	<li>Q2: NO: probably contains no false info</li>
	<li>Q3: YES: definitely of interest</li>
	<li>Q4: NO: definitely not harmful</li>
	<li>Q5: YES:not-urgent</li>
	<li>Q6: YES:rumor</li>
	<li>NO: YES:classified_as_in_question_6</li>
</ol>

### Statistics
Initial distribution of the annotated dataset
  * Arabic data: 4542 tweets
  * Bulgarian data: 4966 tweets
  * Dutch data: 3697 tweets
  * English data: 2665 tweets

More detail is available in the paper[1] [download](https://arxiv.org/abs/2005.00033).

## Questions with Labels
Below is the list of the questions and the possible labels (answers).
See the paper below or the above micromappers links for detailed definition of the annotation guidelines.


**1. Does the tweet contain a verifiable factual claim?** <br/>
*Labels:*
* *YES:* if it contains a verifiable factual claim;
* *NO:* if it does not contain a verifiable factual claim;
* *Don’t know or can’t judge:* the content of the tweet does not have enough information to make a judgment. It is recommended to categorize the tweet using this label when the content of the tweet is not understandable at all. For example, it uses a language (i.e., non-English) or references that are difficult to understand;

**2. To what extent does the tweet appear to contain false information?** <br/>
*Labels:*
1. NO, definitely contains no false information
2. NO, probably contains no false information
3. Not sure
4. YES, probably contains false information
5. YES, definitely contains false information

**3. Will the tweet’s claim have an effect on or be of interest to the general public?** <br/>
*Labels:*
1. NO, definitely not of interest
2. NO, probably not of interest
3. Not sure
4. YES, probably of interest
5. YES, definitely of interest

**4. To what extent does the tweet appear to be harmful to society, person(s), company(s) or product(s)?** <br/>
*Labels:*
1. NO, definitely not harmful
2. NO, probably not harmful
3. Not sure
4. YES, probably harmful
5. YES, definitely harmful

**5. Do you think that a professional fact-checker should verify the claim in the tweet?** <br/>
*Labels:*
1. NO, no need to check
2. NO, too trivial to check
3. YES, not urgent
4. YES, very urgent
5. Not sure

**6. Is the tweet harmful for society and why?** <br/>
*Labels:*
<ol type="A">
<li>NO, not harmful</li>
<li>NO, joke or sarcasm</li>
<li>Not sure</li>
<li>YES, panic</li>
<li>YES, xenophobic, racist, prejudices, or hate-speech</li>
<li>YES, bad cure</li>
<li>YES, rumor or conspiracy</li>
<li>YES, other</li>
</ol>

**7. Do you think that this tweet should get the attention of a government entity?**  <br/>
*Labels:*
<ol type="A">
<li>NO, not interesting</li>
<li>Not sure</li>
<li>YES, categorized as in question 6</li>
<li>YES, other</li>
<li>YES, blame authorities</li>
<li>YES, contains advice</li>
<li>YES, calls for action</li>
<li>YES, discusses action taken</li>
<li>YES, discusses cure</li>
<li>YES, asks question</li>
</ol>

## List of Versions

v1.0 [2021/11/05]: initial distribution of the annotated dataset
* Arabic data: 4966 tweets
* English data: 4542 tweets
* Bulgarian data: 3697 tweets
* Dutch data: 2665 tweets


## Download

Please see the dataset directory for get the tweet ids and labels. To crawl tweets please use tweets hydrators tools:
* [Tool in java ](https://crisisnlp.qcri.org/data/tools/TweetsRetrievalTool-v2.0.zip)
* [Twarc (Python)](https://github.com/DocNow/twarc#dehydrate)
* [Docnow (Desktop application)](https://github.com/docnow/hydrator)

In case if you do not have twitter account or access credentials please create a [Twitter Account](https://twitter.com/). Then follow this guide to retrieve access credentials for the [Twitter API](https://developer.twitter.com/en/docs/authentication/oauth-1-0a/obtaining-user-access-tokens).



## Publication:
Please cite the following paper if you are using the data or annotation guidelines

1. *Firoj Alam and Shaden Shaar and Fahim Dalvi and Hassan Sajjad and Alex Nikolov and Hamdy Mubarak and Giovanni Da San Martino and Ahmed Abdelali and Nadir Durrani and Kareem Darwish and Abdulaziz Al-Homaid and Wajdi Zaghouani and Tommaso Caselli and Gijs Danoe and Friso Stolk and Britt Bruntink and Preslav Nakov, "Fighting the COVID-19 Infodemic: Modeling the Perspective of Journalists, Fact-Checkers, Social Media Platforms, Policy Makers, and the Society", Findings of EMNLP 2021,  [download](https://arxiv.org/abs/2005.00033).*

2. *Firoj Alam, Fahim Dalvi, Shaden Shaar, Nadir Durrani, Hamdy Mubarak, Alex Nikolov, Giovanni Da San Martino,3Ahmed Abdelali,1Hassan Sajjad,1Kareem Darwish,1Preslav Nakov, "Fighting the COVID-19 Infodemic in Social Media: A Holistic Perspective and a Call to Arms", Proceedings of the International AAAI Conference on Web and Social Media. (Vol. 15, pp. 913-922). 2021. [download](https://ojs.aaai.org/index.php/ICWSM/article/view/18114/17917).*

```bib
@inproceedings{alam2020fighting,
    title={Fighting the {COVID}-19 Infodemic: Modeling the Perspective of Journalists, Fact-Checkers, Social Media Platforms, Policy Makers, and the Society},
    author={Firoj Alam and Shaden Shaar and Fahim Dalvi and Hassan Sajjad and Alex Nikolov and Hamdy Mubarak and Giovanni Da San Martino and Ahmed Abdelali and Nadir Durrani and Kareem Darwish and Abdulaziz Al-Homaid and Wajdi Zaghouani and Tommaso Caselli and Gijs Danoe and Friso Stolk and Britt Bruntink and Preslav Nakov},
    booktitle = {Findings of EMNLP 2021},
    year={2021},
}

@InProceedings{alam2020call2arms,
  title		= {Fighting the {COVID}-19 Infodemic in Social Media: A
		  Holistic Perspective and a Call to Arms},
  author	= {Alam, Firoj and Dalvi, Fahim and Shaar, Shaden and
		  Durrani, Nadir and Mubarak, Hamdy and Nikolov, Alex and {Da
		  San Martino}, Giovanni and Abdelali, Ahmed and Sajjad,
		  Hassan and Darwish, Kareem and Nakov, Preslav},
  year		= {2021},
  pages		= {913-922},
  month	= {May},
  volume	= {15},
  booktitle	= {Proceedings of the International {AAAI} Conference on Web
		  and Social Media},
  series	= {ICWSM~'21},
  url		= {https://ojs.aaai.org/index.php/ICWSM/article/view/18114}
}
```


## Credits
* Firoj Alam, Qatar Computing Research Institute, HBKU, Qatar
* Shaden Shaar, Qatar Computing Research Institute, HBKU, Qatar
* Alex Nikolov, Sofia University, Bulgaria
* Hamdy Mubarak, Qatar Computing Research Institute, HBKU, Qatar
* Giovanni Da San Martino, University of Padova, Italy
* Ahmed Abdelali, Qatar Computing Research Institute, HBKU, Qatar
* Fahim Dalvi, Qatar Computing Research Institute, HBKU, Qatar
* Nadir Durrani, Qatar Computing Research Institute, HBKU, Qatar
* Hassan Sajjad, Qatar Computing Research Institute, HBKU, Qatar
* Kareem Darwish, Qatar Computing Research Institute, HBKU, Qatar
* Preslav Nakov, Qatar Computing Research Institute, HBKU, Qatar
* Abdulaziz Al-Homaid, Qatar Computing Research Institute, HBKU, Qatar
* Wajdi Zaghouani, Hamad Bin Khalifa University, Qatar
* Tommaso Caselli, University of Groningen, The Netherlands
* Gijs Danoe, University of Groningen, The Netherlands
* Friso Stolk, University of Groningen, The Netherlands
* Britt Bruntink, University of Groningen, The Netherlands


## Licensing

This dataset is published under CC BY-NC-SA 4.0 license, which means everyone can use this dataset for non-commercial research purpose: https://creativecommons.org/licenses/by-nc/4.0/.

## Contact
Please contact tanbih@qcri.org

## Acknowledgment
Thanks to the QCRI's [Crisis Computing](https://crisiscomputing.qcri.org/) team for facilitating us with [Micromappers](https://micromappers.qcri.org/).
