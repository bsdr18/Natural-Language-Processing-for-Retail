# Natural-Language-Processing-for-Retail

## Project Title: An NLP Exploration in Video Game Analytics for Decoding Retail Sentiments

### PROJECT OVERVIEW

#### Project Task
- In this project scenario, I am assuming the role of a data scientist specializing in NLP, employed by a newly established startup launching a video game.
- The primary goal set by the startup's growth team is to achieve rapid growth in the early stages of the venture.
- The objective is to comprehensively analyze the landscape of the video game market, specifically focusing on understanding how customers assess competitors' products. This involves identifying both the positive and negative aspects that gamers appreciate or dislike in a video game.
- The ultimate aim is to equip the marketing team with insights that enable them to craft a more compelling and effective message for the newly introduced video game.

#### My Take
- As the NLP Specialist, my task involves analyzing customer reviews of video games using various NLP methods to gain a deeper understanding of their feedback and opinions.
- The key steps include downloading a dataset of Amazon reviews and creating a customized dataset from this source.
- Labeling each review with a sentiment score ranging from -1 to 1 to determine whether customers express positive or negative sentiments about the video game.
- Assessing the performance of the sentiment analyzer by comparing sentiment scores with review ratings.
- Evaluating the overall accuracy of the sentiment analyzer and exploring alternative methods of sentiment analysis.
- Classifying reviews into positive, negative, and neutral categories to understand how customers evaluate the video game.
- Summarizing findings for presentation to the Head of the Growth Team, highlighting both liked and disliked aspects of video games based on the analysis results.

### STEPS EMPLOYED
1. Creating the dataset.
2. Sentiment Scoring Using SentiWordNet
3. Sentiment Scoring Model Using NLTK Opinion Lexicon
4. Creating a Dictionary-Based Sentiment Analyzer
5. Reporting the results.

### DATASET DESCRIPTION

The dataset of Amazon reviews is accessible for download [here](https://nijianmo.github.io/amazon/index.html). Please acquire the compressed JSON file corresponding to the video game category "5 core" from the section labeled "Small subsets for experimentation." Following the completion of the download, proceed to unzip the file.

#### Data Dictionary

1. overall: Rating of the product
2. verified: A binary indicator (True/False) denoting whether the review is from a verified purchase, indicating if the reviewer has purchased the product through Amazon.
3. reviewTime: Time of the review (raw)
4. reviewerID: ID of the reviewer, e.g. A2SUAM1J3GNN3B
5. asin: ID of the product, e.g. 0000013714
6. reviewerName: The name or username of the reviewer who posted the review.
7. reviewText: The main body of the review where the reviewer provides detailed feedback or comments about the product.
8. summary: A concise summary of reviewers' main thoughts or opinions.
9. unixReviewTime: The review time in Unix timestamp format, representing the number of seconds since January 1, 1970.
10. vote: The number of helpful votes received by the review from other users. An indicator of the review's perceived usefulness.
11. style: A dictionary of the product's metadata, e.g., "Format" is "Hardcover"
12. image: Images that users post after they have received the product.

### 1. CREATING THE DATASET

#### Undersampling of Reviews - Small Corpus
Taking a random sample of the reviews by selecting 1500 reviews with rating 1, 500-500-500 reviews with ratings 2, 3, 4, and 1500 reviews with rating 5. This way we get a smaller balanced corpus.

```
one_1500 = reviews_df[reviews_df['overall']==1.0].sample(n=1500)
two_500 = reviews_df[reviews_df['overall']==2.0].sample(n=500)
three_500 = reviews_df[reviews_df['overall']==3.0].sample(n=500)
four_500 = reviews_df[reviews_df['overall']==4.0].sample(n=500)
five_1500 = reviews_df[reviews_df['overall']==5.0].sample(n=1500)
undersampled_reviews = pd.concat([one_1500, two_500, three_500, four_500, five_1500], axis=0)
```

![image](https://github.com/bsdr18/Natural-Language-Processing-for-Retail/assets/76464269/41faff59-135a-4bb8-abcf-27f60090bec0)

#### Random Sampling of 100K Reviews - Big Corpus
```sample_100K_revs = reviews_df.sample(n=100000, random_state=42)```

### 2. SENTIMENT SCORING USING SENTIWORDNET
SentiWordNet is a ```lexical resource``` for opinion mining and sentiment analysis. It is an extension of WordNet, which is a lexical database of the English language that groups words into ```sets of synonyms (synsets)``` and provides brief definitions. SentiWordNet, on the other hand, ```assigns sentiment scores to each synset in WordNet```.

For each synset in SentiWordNet, three sentiment scores are provided:
1. **Positive Score:** A numerical score representing the positivity of the term.
2. **Negative Score:** A numerical score representing the negativity of the term.
3. **Objective Score:** A numerical score representing the neutrality or objectivity of the term.

These scores are typically in the range of 0 to 1, where 0 indicates the absence of the corresponding sentiment, and 1 indicates a high degree of that sentiment.

##### ```penn_to_wn``` function:
- This function converts PennTreebank part-of-speech tags to WordNet tags.
- It is used to map different part-of-speech tags to their corresponding WordNet tags (ADJ for adjective, NOUN for noun, ADV for adverb, VERB for verb).

```
def penn_to_wn(tag):
    """
        Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None
```

##### ```get_sentiment_score``` function:
- This function calculates the sentiment score of a given text using SentiWordNet sentiment scores.
- It tokenizes the text, assigns part-of-speech tags, lemmatizes the words, retrieves WordNet synsets, and calculates sentiment scores using SentiWordNet.
- The sentiment scores are then aggregated to compute a final sentiment score for the entire text.

```
def get_sentiment_score(text):
    
    """
        This method returns the sentiment score of a given text using SentiWordNet sentiment scores.
        input: text
        output: numeric (double) score, >0 means positive sentiment and <0 means negative sentiment.
    """    
    total_score = 0
    #print(text)
    raw_sentences = sent_tokenize(text)
    #print(raw_sentences)
    
    for sentence in raw_sentences:

        sent_score = 0     
        sentence = str(sentence)
        #print(sentence)
        sentence = sentence.replace("<br />"," ").translate(str.maketrans('','',punctuation)).lower()
        tokens = TreebankWordTokenizer().tokenize(text)
        tags = pos_tag(tokens)
        for word, tag in tags:
            wn_tag = penn_to_wn(tag)
            if not wn_tag:
                continue
            lemma = WordNetLemmatizer().lemmatize(word, pos=wn_tag)
            if not lemma:
                continue
            synsets = wn.synsets(lemma, pos=wn_tag)
            if not synsets:
                continue
            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())
            sent_score += swn_synset.pos_score() - swn_synset.neg_score()

        total_score = total_score + (sent_score / len(tokens))

    
    return (total_score / len(raw_sentences)) * 100
```

##### Applying Sentiment Analysis to the Dataset (Small Corpus):
```reviews['swn_score'] = reviews['reviewText'].apply(lambda text : get_sentiment_score(text))```

##### Visualization:
- **Histogram:** A histogram of sentiment scores is plotted, excluding extreme values.

![image](https://github.com/bsdr18/Natural-Language-Processing-for-Retail/assets/76464269/1d9abf56-c327-4d91-a34f-2a6beee1b03b)

- **Categorizing Sentiments:** Sentiment labels ('positive', 'negative', 'neutral') are assigned based on predefined score thresholds.
```
reviews['swn_sentiment'] = reviews['swn_score'].apply(lambda x: "positive" if x>1 else ("negative" if x<0.5 else "neutral"))
```

- **Count Plot:** A count plot is created to show the distribution of predicted sentiments against the overall ratings given by reviewers.

![image](https://github.com/bsdr18/Natural-Language-Processing-for-Retail/assets/76464269/0afda880-d421-42f2-9066-bafe4a0061c9)

- **Boxen Plots:** Boxen plots visualize the relationship between sentiment scores and overall ratings.

![image](https://github.com/bsdr18/Natural-Language-Processing-for-Retail/assets/76464269/79bb5ce0-826a-4b8d-91b3-8ef3697522aa)

#### Confusion Matrix
To evaluate the performance of the sentiment analysis against the true sentiments derived from overall ratings.

![image](https://github.com/bsdr18/Natural-Language-Processing-for-Retail/assets/76464269/967a343e-0352-48f6-b87e-452c9ffd34e0)

### 3. SENTIMENT SCORING MODEL USING NLTK OPINION LEXICON
#### Creating Lists of Positive and Negative Words
- The positive and negative words from the opinion lexicon are extracted and stored in ```pos_words``` and ```neg_words``` lists.
```
pos_words = list(opinion_lexicon.positive())
neg_words = list(opinion_lexicon.negative())
```
#### Defining a Sentiment Analysis Function using Opinion Lexicon
- The function get_sentiment_score_oplex calculates the sentiment score of a given text using the NLTK Opinion Lexicon.
- It tokenizes the text, assigns sentiment scores to words based on their presence in positive and negative lists, and calculates a sentiment score for the entire text.
```
def get_sentiment_score_oplex(text):
    
    """
        This method returns the sentiment score of a given text using nltk opinion lexicon.
        input: text
        output: numeric (double) score, >0 means positive sentiment and <0 means negative sentiment.
    """    
    total_score = 0

    raw_sentences = sent_tokenize(text)
    
    for sentence in raw_sentences:

        sent_score = 0     
        sentence = str(sentence)
        sentence = sentence.replace("<br />"," ").translate(str.maketrans('','',punctuation)).lower()
        tokens = TreebankWordTokenizer().tokenize(text)
        for token in tokens:
            sent_score = sent_score + 1 if token in pos_words else (sent_score - 1 if token in neg_words else sent_score)
        total_score = total_score + (sent_score / len(tokens))

    
    return total_score
```

#### Applying Sentiment Analysis to the Dataset (Small Corpus)
```
reviews['oplex_sentiment_score'] = reviews['reviewText'].apply(lambda x: get_sentiment_score_oplex(x))
```

#### Visualization
- Subplots
![image](https://github.com/bsdr18/Natural-Language-Processing-for-Retail/assets/76464269/f3680743-d515-4720-ab7b-244dae7e0206)

- Countplot
![image](https://github.com/bsdr18/Natural-Language-Processing-for-Retail/assets/76464269/8eb20b39-0250-4ece-86c2-07e3d7d95b8c)

- Boxen Plots
![image](https://github.com/bsdr18/Natural-Language-Processing-for-Retail/assets/76464269/f0288aee-5c2f-4f5c-acb6-ed3f4608ff1c)

#### Confusion Matrix 
To evaluate the performance of the sentiment analysis against the true sentiments derived from overall ratings.
![image](https://github.com/bsdr18/Natural-Language-Processing-for-Retail/assets/76464269/5827d4d9-98ac-422a-a297-db3d3f9257e5)

### 4. DICTIONARY-BASED SENTIMENT ANALYSER
#### Tokenizing the Sentences and Words of the Reviews
I'm going to test different versions of word tokenizer on reviews. I will then decide which tokenizer might be better to use.

#### 1. Treebank Word Tokenizer
```
reviews["rev_text_lower"] = reviews['reviewText'].apply(lambda rev: str(rev)\
                                                        .translate(str.maketrans('', '', punctuation))\
                                                        .replace("<br />", " ")\
                                                        .lower())
reviews["tb_tokens"] = reviews['rev_text_lower'].apply(lambda rev: tb_tokenizer.tokenize(str(rev)))
```

#### 2. Casual Tokenizer
```
reviews['casual_tokens'] = reviews['rev_text_lower'].apply(lambda rev: casual_tokenize(str(rev)))
```

#### Stemming
```
stemmer = PorterStemmer()
reviews['tokens_stemmed'] = reviews['tb_tokens'].apply(lambda words: [stemmer.stem(w) for w in words])
```

#### Lemmatisation
```
def penn_to_wn(tag):
    """
        Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None
```
```
lemmatizer = WordNetLemmatizer()
def get_lemas(tokens):
    lemmas = []
    for token in tokens:
        pos = penn_to_wn(pos_tag([token])[0][1])
        if pos:
            lemma = lemmatizer.lemmatize(token, pos)
            if lemma:
                lemmas.append(lemma)
    return lemmas
```
```reviews['lemmas'] = reviews['tb_tokens'].apply(lambda tokens: get_lemas(tokens))```

#### Sentiment Predictor Baseline Model
```
def get_sentiment_score(tokens):
    score = 0
    tags = pos_tag(tokens)
    for word, tag in tags:
        wn_tag = penn_to_wn(tag)
        if not wn_tag:
            continue
        synsets = wn.synsets(word, pos=wn_tag)
        if not synsets:
            continue
        
        #most common set:
        synset = synsets[0]
        swn_synset = swn.senti_synset(synset.name())
        
        score += (swn_synset.pos_score() - swn_synset.neg_score())
        
    return score
```
```
reviews['sentiment_score'] = reviews['lemmas'].apply(lambda tokens: get_sentiment_score(tokens))
```

### 5. REPORTING THE RESULTS
- Looking at the visualizations obtained, it can be observed that the ```distribution of the calculated sentiment scores of the reviews``` skewed largely towards the positive. However, a note-worthy number of them have more of a neutral score, followed by the count of negative sentiment scored reviews.
- Among the groups of 1-5 star rated reviews, the 1,2,4,5 star rated reviews' sentiment scores did not display any ```deviations``` and had largely negative and postive words in them, respecticely.
- Suprisingly, it is seen that neural-rated (Overall: 3) reviews contained more postive sentiment than negative or neutral.
- Apart from this, the ```confusion matrix``` used to evaluate the built sentiment analyzers reveals insights into specific areas where the model excelled or faced challenges.
