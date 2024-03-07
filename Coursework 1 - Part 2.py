from tkinter.constants import W
import sklearn
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, chi2
import random
import os
import operator
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

def load_dataset(path_to_dataset_files):
  dataset = []

  for file in os.listdir(path_to_dataset_files):
    filename = os.fsdecode(file)
    full_file_path = os.path.join(path_to_dataset_files, filename)
    with open(full_file_path, encoding='windows-1252') as infile:
      file_text = ""
      for line in infile:
        if not line.strip():
          continue  # skip the empty line
        file_text += line

      dataset.append(file_text)
  return dataset

def create_category_list(category1, category2, category3, category4, category5):
  category_list = []
  category_list += category1
  category_list += category2
  category_list += category3
  category_list += category4
  category_list += category5

  return category_list

def pre_process_data():
    current_directory = os.path.dirname(os.path.realpath(__file__))

    # Load BBC News Business Dataset
    bbc_news_business_dataset = load_dataset(current_directory + '/BBC Dataset/business')
    bbc_news_business_dataset = shuffle(bbc_news_business_dataset, random_state=20)

    # Load BBC News Entertainment Dataset
    bbc_news_entertainment_dataset = load_dataset(current_directory + '/BBC Dataset/entertainment')
    bbc_news_entertainment_dataset = shuffle(bbc_news_entertainment_dataset, random_state=20)

    # Load BBC News Politics Dataset
    bbc_news_politics_dataset = load_dataset(current_directory + '/BBC Dataset/politics')
    bbc_news_politics_dataset = shuffle(bbc_news_politics_dataset, random_state=20)

    # Load BBC News Sport Dataset
    bbc_news_sport_dataset = load_dataset(current_directory + '/BBC Dataset/sport')
    bbc_news_sport_dataset = shuffle(bbc_news_sport_dataset, random_state=20)

    # Load BBC News Tech Dataset
    bbc_news_tech_dataset = load_dataset(current_directory + '/BBC Dataset/tech')
    bbc_news_tech_dataset = shuffle(bbc_news_tech_dataset, random_state=20)

    # The ratio of train to test size (70:30 train:test)
    train_test_split_size = 0.3

    # Create BBC News Business Train and Test Splits
    bbc_news_business_X_train, bbc_news_business_X_test, bbc_news_business_X_dev, bbc_news_business_Y_train, bbc_news_business_Y_test_gold, bbc_news_business_Y_dev_gold = create_category_train_test_dev_splits(
        bbc_news_business_dataset, 'Business', train_test_split_size)

    # Create BBC News Entertainment Train and Test Splits
    bbc_news_entertainment_X_train, bbc_news_entertainment_X_test, bbc_news_entertainment_X_dev, bbc_news_entertainment_Y_train, bbc_news_entertainment_Y_test_gold, bbc_news_entertainment_Y_dev_gold = create_category_train_test_dev_splits(
        bbc_news_entertainment_dataset, 'Entertainment', train_test_split_size)

    # Create BBC News Politics Train and Test Splits
    bbc_news_politics_X_train, bbc_news_politics_X_test, bbc_news_politics_X_dev, bbc_news_politics_Y_train, bbc_news_politics_Y_test_gold, bbc_news_politics_Y_dev_gold = create_category_train_test_dev_splits(
        bbc_news_politics_dataset, 'Politics', train_test_split_size)

    # Create BBC News Sport Train and Test Splits
    bbc_news_sport_X_train, bbc_news_sport_X_test, bbc_news_sport_X_dev, bbc_news_sport_Y_train, bbc_news_sport_Y_test_gold, bbc_news_sport_Y_dev_gold = create_category_train_test_dev_splits(
        bbc_news_sport_dataset, 'Sport', train_test_split_size)

    # Create BBC News Tech Train and Test Splits
    bbc_news_tech_X_train, bbc_news_tech_X_test, bbc_news_tech_X_dev, bbc_news_tech_Y_train, bbc_news_tech_Y_test_gold, bbc_news_tech_Y_dev_gold = create_category_train_test_dev_splits(
        bbc_news_tech_dataset, 'Tech', train_test_split_size)

    X_train = create_category_list(bbc_news_business_X_train, bbc_news_entertainment_X_train, bbc_news_politics_X_train,
                                   bbc_news_sport_X_train, bbc_news_tech_X_train)
    X_test = create_category_list(bbc_news_business_X_test, bbc_news_entertainment_X_test, bbc_news_politics_X_test,
                                  bbc_news_sport_X_test, bbc_news_tech_X_test)
    X_dev = create_category_list(bbc_news_business_X_dev, bbc_news_entertainment_X_dev, bbc_news_politics_X_dev,
                                 bbc_news_sport_X_dev, bbc_news_tech_X_dev)
    Y_train = create_category_list(bbc_news_business_Y_train, bbc_news_entertainment_Y_train, bbc_news_politics_Y_train,
                                   bbc_news_sport_Y_train, bbc_news_tech_Y_train)
    Y_test_gold = create_category_list(bbc_news_business_Y_test_gold, bbc_news_entertainment_Y_test_gold,
                                       bbc_news_politics_Y_test_gold, bbc_news_sport_Y_test_gold,
                                       bbc_news_tech_Y_test_gold)
    Y_dev_gold = create_category_list(bbc_news_business_Y_dev_gold, bbc_news_entertainment_Y_dev_gold,
                                      bbc_news_politics_Y_dev_gold, bbc_news_sport_Y_dev_gold, bbc_news_tech_Y_dev_gold)

    # shuffle the lists with same order
    zipped = list(zip(X_train, Y_train))
    zipped = shuffle(zipped, random_state = 20)
    X_train, Y_train = zip(*zipped)
    zipped = list(zip(X_test, Y_test_gold))
    zipped = shuffle(zipped, random_state = 20)
    X_test, Y_test_gold = zip(*zipped)
    zipped = list(zip(X_dev, Y_dev_gold))
    zipped = shuffle(zipped, random_state = 20)
    X_dev, Y_dev_gold = zip(*zipped)

    return X_train, X_test, X_dev, Y_train, Y_test_gold, Y_dev_gold

def create_input_features(dataset, vocabulary, common_part_of_speech_tags, lemmatizer):
  input_features = []

  # Populate the training input and output
  for index in range(len(dataset)):
    tf_idf_vector_input = np.asarray(get_vector_tf_idf(vocabulary, dataset[index], lemmatizer))
    part_of_speech_vector_input = np.asarray(get_vector_part_of_speech_tag(dataset[index], common_part_of_speech_tags))
    sentiment_scores  = get_sentiment(dataset[index])
    sentiment_input = np.array([sentiment_scores['neg'], sentiment_scores['neu'], sentiment_scores['pos'], sentiment_scores['compound']])

    combined_features =  np.concatenate((tf_idf_vector_input, part_of_speech_vector_input, sentiment_input))
    input_features.append(combined_features)

  return np.array(input_features)

def create_category_list(category1, category2, category3, category4, category5):
  category_list = []
  category_list += category1
  category_list += category2
  category_list += category3
  category_list += category4
  category_list += category5

  return category_list

def create_category_train_test_dev_splits(dataset, category, test_size):
  # Create output category list
  output_list = []

  for index in range(len(dataset)):
    output_list.append(category)

  # Create split of 80:30 training:dev splits, use a random_state of 20 for the seed to ensure the same random numbers are reproduced on each run, and thus the model will always achieve the same accuracy.
  dataset_X_train_set, dataset_X_test_set, dataset_Y_train_set, dataset_Y_test_set = train_test_split(dataset, output_list, test_size = test_size, random_state=20)

  # Create split of 50:50 train and test splits
  dataset_X_dev_set, dataset_X_test_set, dataset_Y_dev_set, dataset_Y_test_set = train_test_split(dataset_X_test_set, dataset_Y_test_set, test_size = 0.5,  random_state=20)

  return dataset_X_train_set, dataset_X_test_set, dataset_X_dev_set, dataset_Y_train_set, dataset_Y_test_set, dataset_Y_dev_set

def determine_number_of_part_of_speech_features(X_train, X_dev, Y_train, Y_dev_gold, stopwords, list_part_of_speech_size, lemmatizer):
  print("Determining the size of part of speech to use from list of:", list_part_of_speech_size)

  # Now we can train our model with the different number of features, and test each of them in the dev set
  best_accuracy_dev= 0.0

  for num_features in list_part_of_speech_size:
    print("Testing part of speech size of:", num_features, '...')

    # First, we get the common part of speech tag from the training set and train our svm classifier
    common_part_of_speech_tags = get_most_comon_part_of_speech_tags(X_train, num_features, stopwords, lemmatizer)

    # Then we create out X_train input data, using our common_part_of_speech_tags
    X_train_input = create_input_features(X_train, '', common_part_of_speech_tags, lemmatizer)

    X_train_input = np.asarray(X_train_input)
    Y_train = np.asarray(Y_train)

    # Now train the SCM classifier against our training data set
    svm_clf = train_classifier(X_train_input, Y_train)

    # Then we create out X_dev input data, using out common_part_of_speech_tags
    X_dev_input = create_input_features(X_dev, '', common_part_of_speech_tags, lemmatizer)

    X_dev_input = np.asarray(X_dev_input)
    Y_dev_gold = np.asarray(Y_dev_gold)

    # Now make our Y_dev predictions
    Y_dev_predictions=svm_clf.predict(X_dev_input)

    # Finally, we get the accuracy results of the classifier
    accuracy_dev=accuracy_score(Y_dev_gold, Y_dev_predictions)
    print ("Accuracy with "+str(num_features)+": "+str(round(accuracy_dev,3)))
    if accuracy_dev > best_accuracy_dev:
      best_accuracy_dev = accuracy_dev
      best_num_features = num_features

  print ("\nBest accuracy for part of speech size overall in the dev set is "+str(round(best_accuracy_dev,3))+" with "+str(best_num_features)+" features.")

  return best_num_features

def determine_number_of_vocabulary_features(X_train, X_dev, Y_train, Y_dev_gold, stopwords, list_vocab_size, lemmatizer):
  print("\nDetermining the size of vocabulary to use from list of:", list_vocab_size)

  # Now we can train our model with the different number of features, and test each of them in the dev set
  best_accuracy_dev = 0.0
  best_num_features = 0
  for num_features in list_vocab_size:
    print("Testing vocabulary size of:", num_features, '...')

    # First, we get the vocabulary from the training set and train our svm classifier
    vocabulary = get_most_common_vocabulary(X_train, num_features, stopwords, lemmatizer)

    # Then we create out X_train input data, using our vocabulary
    X_train_input = create_input_features(X_train, vocabulary, '', lemmatizer)

    X_train_input = np.asarray(X_train_input)
    Y_train = np.asarray(Y_train)

    # Now train the SCM classifier against our training data set
    svm_clf = train_classifier(X_train_input, Y_train)

    # Then we create out X_dev input data, using our vocabulary
    X_dev_input = create_input_features(X_dev, vocabulary, '', lemmatizer)

    X_dev_input = np.asarray(X_dev_input)
    Y_dev_gold = np.asarray(Y_dev_gold)

    # Now make our Y_dev predictions
    Y_dev_predictions=svm_clf.predict(X_dev_input)

    # Finally, we get the accuracy results of the classifier
    accuracy_dev=accuracy_score(Y_dev_gold, Y_dev_predictions)
    print ("Accuracy with "+str(num_features)+": "+str(round(accuracy_dev,3)))
    if accuracy_dev > best_accuracy_dev:
      best_accuracy_dev = accuracy_dev
      best_num_features = num_features

  print ("\nBest accuracy for vocabulary count overall in the dev set is "+str(round(best_accuracy_dev,3))+" with "+str(best_num_features)+" features.")

  return best_num_features

def get_list_tokens(string, lemmatizer):
  sentence_split=nltk.tokenize.sent_tokenize(string)
  list_tokens=[]
  for sentence in sentence_split:
    list_tokens_sentence=nltk.tokenize.word_tokenize(sentence)
    for token in list_tokens_sentence:
      list_tokens.append(lemmatizer.lemmatize(token).lower())
  return list_tokens

def get_part_of_speech_tags(text, lemmatizer):
  # Tokenize and lemmatize the text, then filter out stopwords
  tokens = get_list_tokens(text, lemmatizer)
  cleaned_tokens = [token for token in tokens if token not in stopwords]

  # Tag each token with its part of speech
  tagged_tokens = nltk.pos_tag(cleaned_tokens)

  return tagged_tokens

def get_sentiment(text):
  sia = SentimentIntensityAnalyzer()
  return sia.polarity_scores(text)

def get_vector_tf_idf(list_vocab, text, lemmatizer):
  vector_text=np.zeros(len(list_vocab))
  list_tokens_string = get_list_tokens(text, lemmatizer)
  for i, word in enumerate(list_vocab):
    if word in list_tokens_string:
      vector_text[i]=list_tokens_string.count(word)
  return vector_text

def get_vector_part_of_speech_tag(text, common_part_of_speech_tags):
  part_of_speech_tag_vector = np.zeros(len(common_part_of_speech_tags))
  part_of_speech_tags = get_part_of_speech_tags(text, lemmatizer)
  index = 0

  for word, part_of_speech_tag in part_of_speech_tags:
    if (part_of_speech_tag in common_part_of_speech_tags):
      index = common_part_of_speech_tags.index(part_of_speech_tag)
      part_of_speech_tag_vector[index] += 1

  return part_of_speech_tag_vector

def get_most_common_vocabulary(categorical_dataset, number_of_vocab_features, stopwords, lemmatizer):
  print("\nGetting most common vocabulary...")
  dict_word_frequency={}
  vocabulary= []

  # Iterate through each category dataset
  for categorical_data in categorical_dataset:
    sentence_tokens = get_list_tokens(categorical_data, lemmatizer)
    for word in sentence_tokens:
      if word in stopwords: continue
      if word not in dict_word_frequency: dict_word_frequency[word]=1
      else: dict_word_frequency[word]+=1

  # Now we create a sorted frequency list with the top number_of_vocab_features words, using the function "sorted". Let's see the number_of_vocab_features most frequent words
  word_frequency_sorted_list = sorted(dict_word_frequency.items(), key=operator.itemgetter(1), reverse=True)[:number_of_vocab_features]

  for word,frequency in word_frequency_sorted_list:
    vocabulary.append(word)

  return vocabulary

def get_most_comon_part_of_speech_tags(categorical_dataset, number_of_part_of_speech_features, stopwords, lemmatizer):
  print("\nGetting most common part of speech tags...")
  dict_part_of_speech_tags_frequency = {}
  part_of_speech_tags = []

  # Iterate through each category dataset
  for categorical_data in categorical_dataset:
    tagged_tokens = get_part_of_speech_tags(categorical_data, lemmatizer)

    # Count the frequencies of each POS tag
    for _, tag in tagged_tokens:
      if tag not in dict_part_of_speech_tags_frequency:
        dict_part_of_speech_tags_frequency[tag] = 1
      else:
        dict_part_of_speech_tags_frequency[tag] += 1

  # Now we create a sorted frequency list with the top number_of_part_of_speech_features tags, using the function "sorted". Let's see the number_of_part_of_speech_features most frequent tags
  part_of_speech_tag_frequency_sorted_list = sorted(dict_part_of_speech_tags_frequency.items(), key=operator.itemgetter(1), reverse=True)[:number_of_part_of_speech_features]

  for word,frequency in part_of_speech_tag_frequency_sorted_list:
    vocabulary.append(word)

  for tag,frequency in part_of_speech_tag_frequency_sorted_list:
    part_of_speech_tags.append(tag)

  return part_of_speech_tags

def train_classifier(X_train, Y_train):
  # uses a random_state of 20 for the seed to ensure the same random numbers are reproduced on each run, and thus the model will always achieve the same accuracy.
  svm_clf=sklearn.svm.SVC(kernel="linear",gamma='auto', random_state=20)
  svm_clf.fit(X_train,Y_train)

  return svm_clf


X_train, X_test, X_dev, Y_train, Y_test_gold, Y_dev_gold = pre_process_data()

# Initialize Lemmatizer to use WordNetLemmatizer
lemmatizer = nltk.stem.WordNetLemmatizer()

# First, we get the stopwords list from nltk
stopwords = set(nltk.corpus.stopwords.words('english'))

# We can add more words to the stopword list, like punctuation marks
stopwords.update([".", ",", "--", "``", "'s", "n't", "'", "``", "''", "(", ")", "-"])

# The number of words we want to use for building up our vocabulary across all datasets
# We will determine this by evaluating the best performance for this value using the dev sets
list_vocab_size = [730, 735, 740]
number_of_vocabulary_features_per_category = determine_number_of_vocabulary_features(X_train, X_dev, Y_train, Y_dev_gold, stopwords, list_vocab_size, lemmatizer)

# Populate vocabulary using the number_of_vocabulary_features_per_category determined previously
vocabulary = get_most_common_vocabulary(X_train, number_of_vocabulary_features_per_category, stopwords, lemmatizer)

# The number of part of speech tags we want to include when building up our part of speech across the datasets
# We will determine this by evaluating the best performance for this value using the dev sets
list_part_of_speech_size = [15, 30, 35]
number_of_part_of_speech_features = determine_number_of_part_of_speech_features(X_train, X_dev, Y_train, Y_dev_gold, stopwords, list_part_of_speech_size, lemmatizer)

# Populate common_part_of_speech_tags using the common_part_of_speech_tags determined previously
common_part_of_speech_tags = get_most_comon_part_of_speech_tags(X_train, number_of_part_of_speech_features, stopwords, lemmatizer)

# Create X_Train inputs, using Term Frequency Inverse Document Frequency as feature 1
print("\nCreating input features for X_train...")
X_train = create_input_features(X_train, vocabulary, common_part_of_speech_tags, lemmatizer)

# Create X_Test inputs, using Term Frequency Inverse Document Frequency as feature 1
print("\nCreating input features for X_test")
X_test = create_input_features(X_test, vocabulary, common_part_of_speech_tags, lemmatizer)

# Create X_Test inputs, using Term Frequency Inverse Document Frequency as feature 1
print("\nCreating input features for X_dev")
X_dev = create_input_features(X_dev, vocabulary, common_part_of_speech_tags, lemmatizer)

# Let's scale the data to be between 0 and 1
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
X_dev_scaled = scaler.fit_transform(X_dev)

# Feature selection - Selecting the best k features. There are 809 features total in X_train, let's try reduce this
# Let's try find the best accuracy using our dev train set
k_features_list = [150, 200, 250]
k_feature_size = 0
best_accuracy_dev = 0

print("Determining best k_feature size to use for SelectKBest algorithm")
for k_features in k_features_list:
  print("Testing k_feature size of:", k_features, '...')
  k_best = SelectKBest(chi2, k=k_features)
  X_train_best = k_best.fit_transform(X_train_scaled, Y_train)
  X_dev_best = k_best.transform(X_dev_scaled)

  X_train_best=np.asarray(X_train_best)
  X_dev_best=np.asarray(X_dev_best)
  Y_train=np.asarray(Y_train)
  Y_dev_gold=np.asarray(Y_dev_gold)

  svm_clf = train_classifier(X_train_best, Y_train)
  Y_dev_predictions=svm_clf.predict(X_dev_best)

  accuracy_dev=accuracy_score(Y_dev_gold, Y_dev_predictions)
  print ("Accuracy with "+str(k_features)+": "+str(round(accuracy_dev,3)))

  if accuracy_dev > best_accuracy_dev:
      print("Changing k_feature_size to:", k_features)
      best_accuracy_dev = accuracy_dev
      k_feature_size = k_features

print("Analysis of best k-feature size complete, will use", k_feature_size)
k_best = SelectKBest(chi2, k=k_features)
X_train_best = k_best.fit_transform(X_train_scaled, Y_train)
X_test_best = k_best.transform(X_test_scaled)
X_train_best=np.asarray(X_train_best)
X_test_best=np.asarray(X_test_best)
Y_train=np.asarray(Y_train)
Y_test_gold=np.asarray(Y_test_gold)

svm_clf = train_classifier(X_train_best, Y_train)
Y_test_predictions=svm_clf.predict(X_test_best)

print(classification_report(Y_test_gold, Y_test_predictions))
#print(confusion_matrix(Y_test_gold, Y_test_predictions))