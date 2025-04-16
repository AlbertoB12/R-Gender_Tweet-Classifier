# As a very first step, all packages must be installed and loaded
# Install
install.packages("quanteda")
install.packages("textstem")
install.packages("quanteda.textmodels")
install.packages("caret")
install.packages("ggplot2")
install.packages("quanteda.textstats")
install.packages("stm")
install.packages("wordcloud")
install.packages("igraph")
install.packages("wordnet")
install.packages("compareDF")
install.packages("testthat")
# Load
library(quanteda)
library(textstem)
library(quanteda.textmodels)
library(caret)
library(ggplot2)
library(quanteda.textstats)
library(stm)
library(wordcloud)
library(igraph)
library(wordnet)
library(compareDF)
library(testthat)

# For the classification of tweets with a Naive Bayes classification system, the data frame with the annotated tweets must first be imported
setwd("/content") # Path
gender_data <- read.csv("gender_classifier.csv", header = TRUE, sep = ",", encoding = "UTF-8", stringsAsFactors = FALSE) # Read gender dataset (csv file)
class(gender_data) # Check class

# Preprocessing
# From the entire dataset, we only keep the metadata that might be most important for our analysis
# For this purpose, a corpus is created from the dataset
gender_corpus <- corpus(gender_data$text, docvars = data.frame(ID = gender_data$X_unit_id, Text = gender_data$text, Golden = gender_data$X_golden, Gender = gender_data$gender, Favorites = gender_data$fav_number, Color = gender_data$link_color, Confidence = gender_data$gender.confidence, Description = gender_data$description, Gold = gender_data$gender_gold, Link_color = gender_data$link_color, Retweets = gender_data$retweet_count, Sidebar_color = gender_data$sidebar_color, Tweet_count = gender_data$tweet_count))
class(gender_corpus) # Check class
# For the later creation of training and test datasets, it is easier if all documents from the corpus receive an identification
docvars(gender_corpus, "Document_ID") <- 1:ndoc(gender_corpus)

# Rough check of the created corpus
head(gender_corpus)
names(docvars(gender_corpus))
ndoc_gender_corpus <- ndoc(gender_corpus)
ndoc_gender_corpus

# Standard Naive Bayes classifiers work with two variables
# For this reason, we will keep tweets with gender classification "male" and "female"
# Tweets that were classified as "brand" will be excluded from the analysis
two_gender_corpus <- corpus_subset(gender_corpus, Gender %in% c("male", "female")) # Create subcorpus
class(two_gender_corpus)
ndoc_two_gender_corpus <- ndoc(two_gender_corpus)
ndoc_two_gender_corpus

# How many posts were classified as "brand"?
ndoc_gender_corpus - ndoc_two_gender_corpus

# Before we continue preparing the corpus for the Naive Bayes classifier, we should check something
# Are all numeric "docvars" of type "numeric" (or similar)?
# This will avoid future problems
names(docvars(two_gender_corpus))
class(two_gender_corpus$Favorites)
class(two_gender_corpus$Confidence)
class(two_gender_corpus$Retweets)
class(two_gender_corpus$Tweet_count)
# Everything looks good

# The next step is to decide from which value of the variable "Confidence" we keep the categorization "male" and "female"
# What are the lowest and highest values?
range_two_gender_corpus <- range(two_gender_corpus$Confidence)
range_two_gender_corpus
range_table <- table(two_gender_corpus$Confidence) # All values in a table
range_table

# It is noticeable that the majority of posts were annotated with Confidence = 1
# This gives the dataset high credibility
# What is the mean value between the lowest and the highest value?
mid_value <- (range_two_gender_corpus[1] + range_two_gender_corpus[2]) / 2
mid_value

# We keep all posts with Confidence ≥ mid_value (0.6603) for our corpus
corpus <- corpus_subset(two_gender_corpus, Confidence >= 0.6603)
ndoc_corpus <- ndoc(corpus)
ndoc(corpus)
# How many documents were removed?
ndoc_two_gender_corpus - ndoc_corpus

# The last step before model training is to define the training and test data
# Usually, an 80:20 proportion is used
# We also use this proportion in this case
set.seed(1)
proportion <- sample(c(TRUE,FALSE), ndoc(corpus), replace = TRUE, prob = c(0.8, 0.2))
corpus_training <- corpus[proportion, ]
corpus_test <- corpus[!proportion, ]
# Check
ndoc(corpus_training)
ndoc(corpus_test)
# Is corpus_training + corpus_test = corpus (in terms of ndoc)?
if (ndoc(corpus_training) + ndoc(corpus_test) == ndoc_corpus){
  print("Yes")
} else {
  print("No")
}
# Everything looks good

# Creation of the DTM matrix for the training data
tokens_training <- quanteda::tokens(as.character(corpus_training), remove_punct = TRUE, remove_numbers = TRUE, remove_symbols = TRUE, remove_url = TRUE)
tokens_training <- tokens_remove(tokens_tolower(tokens_training), stopwords("english"))
class(tokens_training)
tokens_training

docvars(tokens_training, "Gender") <- corpus_training$Gender
head(docvars(tokens_training))
tokens_replace(tokens_training, pattern = lexicon::hash_lemmas$token, replacement = lexicon::hash_lemmas$lemma)
head(tokens_training)

dtm_training <- dfm(tokens_training)

# Creation of the DTM matrix for the test data
tokens_test <- quanteda::tokens(as.character(corpus_test), remove_punct = TRUE, remove_numbers = TRUE, remove_symbols = TRUE, remove_url = TRUE)
tokens_test <- tokens_remove(tokens_tolower(tokens_test), stopwords("english"))
class(tokens_test)
tokens_test

docvars(tokens_test, "Gender") <- corpus_test$Gender
head(docvars(tokens_test))
tokens_replace(tokens_test, pattern = lexicon::hash_lemmas$token, replacement = lexicon::hash_lemmas$lemma)
head(tokens_test)

dtm_test <- dfm(tokens_test)

# Now everything is prepared for training the Naive Bayes model
nb_training <- textmodel_nb(dtm_training, dtm_training$Gender, prior = "docfreq")

# The model is now trained
# The next step is the automatic classification of the test data
dtm_test_match <- dfm_match(dtm_test, features = featnames(dtm_training))
prediction <- predict(nb_training, newdata = dtm_test_match)
actual_class <- dtm_test_match$Gender
class_table <- table(actual_class, prediction)
class_table

# Statistical data
confusionMatrix(class_table, mode = "everything")

# Words most strongly categorized as female or male by the model
# Female
words <- data.frame(Words = colnames(dtm_training), Coef = coef(nb_training))
words$Prediction_female <- words$Coef.female - words$Coef.male
head(words[order(words$Coef.female, decreasing = TRUE), c("Words")], 60)

# Male
words$Prediction_male <- words$Coef.male - words$Coef.female
head(words[order(words$Coef.male, decreasing = TRUE), c("Words")], 60)

# Classification of spam messages
spam_data <- read.csv("spam.csv", header = TRUE, sep = ",", encoding = "UTF-8", stringsAsFactors = FALSE) # Read spam dataset (csv file)
class(spam_data)

head(spam_data)
# The columns do not have meaningful names
colnames(spam_data)
# We change the names of the first two columns
names(spam_data)[names(spam_data) == "v1"] <- "Data_type" # 1st column
names(spam_data)[names(spam_data) == "v2"] <- "Text" # 2nd column
colnames(spam_data)
head(spam_data)
# Now it looks better

# Preprocessing
# A corpus is created from the dataset
class(spam_data)
spam_corpus <- corpus(spam_data$Text, docvars = data.frame(Type = spam_data$Data_type))
head(spam_corpus)
class(spam_corpus)
head(docvars(spam_corpus))
docvars(spam_corpus, "Document_ID") <- 1:ndoc(spam_corpus)
head(docvars(spam_corpus))

# We are only interested in spam texts
# All ham texts are therefore deleted
ndoc_spam_corpus <- ndoc(spam_corpus)
ndoc_spam_corpus
only_spam_corpus <- corpus_subset(spam_corpus, Type == "spam")
ndoc_only_spam_corpus <- ndoc(only_spam_corpus)
ndoc_only_spam_corpus

# How many ham texts were deleted?
ndoc_spam_corpus - ndoc_only_spam_corpus

# Creation of the DTM matrix for the test data
tokens_test_spam <- quanteda::tokens(as.character(only_spam_corpus), remove_punct = TRUE, remove_symbols = TRUE, remove_url = TRUE)
tokens_test_spam <- tokens_remove(tokens_tolower(tokens_test_spam), stopwords("english"))
class(tokens_test_spam)
tokens_test_spam

docvars(tokens_test_spam, "Type") <- only_spam_corpus$Type
head(docvars(tokens_test_spam))
tokens_replace(tokens_test_spam, pattern = lexicon::hash_lemmas$token, replacement = lexicon::hash_lemmas$lemma)
head(tokens_test_spam)

dtm_test_spam <- dfm(tokens_test_spam)

# Automatic classification of the test data
dtm_test_spam_match <- dfm_match(dtm_test_spam, features = featnames(dtm_training))
prediction_spam <- predict(nb_training, newdata = dtm_test_spam_match)
actual_class_spam <- dtm_test_spam_match$Type
class_table_spam <- table(actual_class_spam, prediction_spam)
class_table_spam

# Classification of Fake News
fake_data <- read.csv("fake.csv", header = TRUE, sep = ",", encoding = "UTF-8", stringsAsFactors = FALSE) # Read fake news dataset (csv file)
class(fake_data)

head(fake_data)
names(fake_data)

# Preprocessing
# A corpus is created from the dataset
fake_corpus <- corpus(fake_data$content, docvars = data.frame(ID = fake_data$external_author_id, Language = fake_data$language, Following = fake_data$following, Followers = fake_data$followers))
head(fake_corpus)
class(fake_corpus)
head(docvars(fake_corpus))
docvars(fake_corpus, "Document_ID") <- 1:ndoc(fake_corpus)
head(docvars(fake_corpus))

# We are only interested in fake news texts in English
# All texts in other languages are therefore deleted
ndoc_fake_corpus <- ndoc(fake_corpus)
ndoc_fake_corpus
eng_fake_corpus <- corpus_subset(fake_corpus, Language == "English")
ndoc_eng_fake_corpus <- ndoc(eng_fake_corpus)
ndoc_eng_fake_corpus

# How many texts were deleted?
ndoc_fake_corpus - ndoc_eng_fake_corpus

# Creation of the DTM matrix for the test data
tokens_test_fake <- quanteda::tokens(as.character(eng_fake_corpus), remove_punct = TRUE, remove_symbols = TRUE, remove_url = TRUE)
tokens_test_fake <- tokens_remove(tokens_tolower(tokens_test_fake), stopwords("english"))
class(tokens_test_fake)
tokens_test_fake

docvars(tokens_test_fake, "Language") <- eng_fake_corpus$Language
head(docvars(tokens_test_fake))
tokens_replace(tokens_test_fake, pattern = lexicon::hash_lemmas$token, replacement = lexicon::hash_lemmas$lemma)
head(tokens_test_fake)

dtm_test_fake <- dfm(tokens_test_fake)

# Automatic classification of the test data
dtm_test_fake_match <- dfm_match(dtm_test_fake, features = featnames(dtm_training))
prediction_fake <- predict(nb_training, newdata = dtm_test_fake_match)
actual_class_fake <- dtm_test_fake_match$Language
class_table_fake <- table(actual_class_fake, prediction_fake)
class_table_fake

# The Naive Bayes model has been trained and all datasets classified
# Now we want to get more information about the possible differences between male and female language
# For this we use the first analyzed dataset again
# The first step is the relative frequencies
# We don't need to redo all the preprocessing steps
# To be able to analyze as many posts as possible, we merge the training and test DTM
dtm_together <- rbind(dtm_training, dtm_test)
dtm_together
names(docvars(dtm_together))

# Which words do we want to use for the analysis?
# Due to Zipf's Law, it would not make sense to include many words in the analysis
# We only take the first 15 words
freq_words <- textstat_frequency(dtm_together, n = 15)
freq_words
# All 15 words are in both groups (male and female)

terms <- c("just", "like", "get", "one", "love", "can", "day", "go", "now", "time", "people", "new", "know", "good", "shit", "fuck") # Top 15 + "shit" and "fuck"
# The character number 12 in the table is not interesting to analyze
dtm_small <- as.matrix(dtm_together[, terms])
head(dtm_small)

# For ggplot2
# Now we can separate the texts according to gender
gender <- dtm_together$Gender # We need information about gender in the matrix
dtm_small_gender <- cbind(dtm_small, gender)
freq_plot <- reshape2::melt(dtm_small_gender, id.var = "gender")
head(freq_plot)
freq_plot_gender <- cbind(freq_plot, gender)
freq_plot_gender
colnames(freq_plot_gender) <- c("Text", "Word", "Value", "Gender")
freq_plot_gender

ggplot(freq_plot_gender, aes(x = Gender, y = Value, colour = Word, group = Word)) +
  geom_line() +
  labs(x = "Gender", y = "Relative word frequency", title = "Relative Word Frequencies top 14 Words + 'shit' and 'fuck'") +
  theme_gray() +
  theme(legend.position = "none", axis.text.x = element_text(angle = 50)) +
  facet_wrap(.~Word, scales = "free")

# Now we calculate the distance between these selected terms
dist_matrix <- dist(t(dtm_small), method = "euclidean")
head(as.matrix(dist_matrix))

clust_terms <- hclust(dist_matrix, method = "complete")
cluster <- cutree(clust_terms, k = 5)
df_cluster <- data.frame(cluster = cluster, variable = names(cluster))
print(df_cluster)
# The cluster distribution with different values for k does not seem to discard clear and interesting results

# Another important point for the analysis is the creation of Topic Models
# We consider words that appear in at least 10 documents
dtm_topic <- dfm_trim(dtm_together, min_docfreq = 10)
dtm_topic

topic_stm <- stm(
  documents = dtm_topic,
  data = docvars(dtm_topic),
  K = 10,
  max.em.its = 50,
  seed = 4023330,
  init.type = "Spectral",
  verbose = TRUE)

names(topic_stm)
labelTopics(topic_stm, c(1, 3, 4, 7, 9))

# Wordcloud
cloud(topic_stm, topic = 4, scale = c(9, 0.5))
cloud(topic_stm, topic = 9, scale = c(8, 0.5))

plot(topic_stm, type = "summary", xlim = c(0, 0.3), n = 5)

# These are the most frequent topics of the entire dataset
# Are there differences when topics are calculated separately for the male and female datasets?
# gender_data # We use this data frame
# We only keep gender and Confidence ≥ 0.6603
names(gender_data)
gender_df <- gender_data[, c(6, 7, 20)]
head(gender_df)

# Now we create data frames for male and female
df_male <- subset(gender_df, gender == "male") # Male
head(df_male)

df_female <- subset(gender_df, gender == "female") # Female
head(df_female)

# Preprocessing
# Male
tokens_male <- quanteda::tokens(as.character(df_male$text), remove_punct = TRUE, remove_numbers = TRUE, remove_symbols = TRUE, remove_url = TRUE)
tokens_male <- tokens_remove(tokens_tolower(tokens_male), stopwords("english"))
class(tokens_male)
tokens_male

tokens_replace(tokens_male, pattern = lexicon::hash_lemmas$token, replacement = lexicon::hash_lemmas$lemma)
head(tokens_male)

dtm_male <- dfm(tokens_male)

# Female
tokens_female <- quanteda::tokens(as.character(df_female$text), remove_punct = TRUE, remove_numbers = TRUE, remove_symbols = TRUE, remove_url = TRUE)
tokens_female <- tokens_remove(tokens_tolower(tokens_female), stopwords("english"))
class(tokens_female)
tokens_female

tokens_replace(tokens_female, pattern = lexicon::hash_lemmas$token, replacement = lexicon::hash_lemmas$lemma)
head(tokens_female)

dtm_male <- dfm_trim(dtm_male, min_docfreq = 10)
dtm_male

dtm_female <- dfm_trim(dtm_female, min_docfreq = 10)
dtm_female

# stm male
stm_male <- stm(
  documents = dtm_male,
  data = docvars(dtm_male),
  K = 10,
  max.em.its = 50,
  seed = 4023330,
  init.type = "Spectral",
  verbose = TRUE)

# stm female
stm_female <- stm(
  documents = dtm_female,
  data = docvars(dtm_female),
  K = 10,
  max.em.its = 50,
  seed = 4023330,
  init.type = "Spectral",
  verbose = TRUE)

# Male
labelTopics(stm_male, c(1, 2, 4, 5, 6, 7, 8, 9, 10))

# Female
labelTopics(stm_female, c(1, 2, 4, 5, 7, 8, 9, 10))

# Wordclouds
# Male
cloud(stm_male, topic = 3, scale = c(10, 0.5))
# Female
cloud(stm_female, topic = 8, scale = c(4, 0.5))

# Topic order by frequency
# Male
plot(stm_male, type = "summary", xlim = c(0, 0.3), n = 4)
# Female
plot(stm_female, type = "summary", xlim = c(0, 0.3), n = 4)

# Semantic Networks
# dtm_together # We use this dtm
dtm_network <- dfm_trim(dtm_together, min_docfreq = 10)
head(dtm_network)

# Co-Occurrences
dtm_network_sparse <- dfm_weight(dtm_network, "boolean")
cooc <- t(dtm_network_sparse) %*% dtm_network_sparse
as.matrix(cooc[1:15, 1:15])

graph_network <- graph_from_adjacency_matrix(cooc, mode = "undirected", weighted = TRUE, diag = FALSE)
graph_network

# We set a minimum number of relations so that the network is clearer
min <- mean(E(graph_network)$weight)
graph_network_small <- subgraph.edges(graph_network, E(graph_network)[E(graph_network)$weight > min])
graph_network_small

# Only the top 30 words regarding frequency in the dataset are considered
freq_words <- textstat_frequency(dtm_together, n = 30)
freq_words

words_network <- c(freq_words$feature)
words_network

# Are all words in the graph?
words_network %in% V(graph_network_small)$name
# Yes

graph_words <- induced.subgraph(graph = graph_network_small, vids = words_network)

# Network Plot
plot.igraph(graph_words,
            layout = layout_nicely(graph_words),
            vertex.size = 20,
            vertex.frame.color = 'white',
            vertex.color = 'light blue',
            vertex.label.cex = 1)

# The next step is the differentiation of gender within the network
# How does the context of words change between female and male posts?
# Male
# dtm_together # We use this dtm
names(docvars(dtm_together))
dtm_male_network <- dfm_subset(dtm_together, dtm_together$Gender == "male")
dtm_male_network

dtm_male_network_trim <- dfm_trim(dtm_male_network, min_termfreq = 10)
dtm_male_network_sparse <- dfm_weight(dtm_male_network_trim, "boolean")
cooc_male <- t(dtm_male_network_sparse) %*% dtm_male_network_sparse

graph_male <- graph_from_adjacency_matrix(cooc_male, mode = "undirected", weighted = TRUE, diag = FALSE)
male_small <- mean(E(graph_male)$weight) + 2 * sd(E(graph_male)$weight)
graph_male_small <- subgraph.edges(graph_male, E(graph_male)[E(graph_male)$weight > male_small])

# Female
dtm_female_network <- dfm_subset(dtm_together, dtm_together$Gender == "female")
dtm_female_network

dtm_female_network_trim <- dfm_trim(dtm_female_network, min_termfreq = 10)
dtm_female_network_sparse <- dfm_weight(dtm_female_network_trim, "boolean")
cooc_female <- t(dtm_female_network_sparse) %*% dtm_female_network_sparse

graph_female <- graph_from_adjacency_matrix(cooc_female, mode = "undirected", weighted = TRUE, diag = FALSE)
female_small <- mean(E(graph_female)$weight) + 2 * sd(E(graph_female)$weight)
graph_female_small <- subgraph.edges(graph_female, E(graph_female)[E(graph_female)$weight > female_small])

# The two igraph variables are prepared
# Now we can graphically represent networks
# We only need to decide from which words we want to display the graphs
# Top 50 most frequent words
freq_words_network <- textstat_frequency(dtm_together, n = 50)
freq_words_network

# "like"
# Male
male_like <- neighbors(graph_male_small, "like")
male_like <- names(V(graph_male_small)[male_like])
graph_male_like <- induced.subgraph(graph = graph_male_small, vids = c("like", male_like))
# Female
female_like <- neighbors(graph_female_small, "like")
female_like <- names(V(graph_female_small)[female_like])
graph_female_like <- induced.subgraph(graph = graph_female_small, vids = c("like", female_like))
# Graphical representation
# Male
plot.igraph(graph_male_like, vertex.size = degree(graph_male_like)/8, vertex.frame.color = 'gray', vertex.color = ifelse(V(graph_male_like)$name == "like", "lightblue", "orange"), vertex.label.cex = 0.8, edge.label.cex = 5)
# Female
plot.igraph(graph_female_like, vertex.size = degree(graph_female_like)/6, vertex.frame.color = 'gray', vertex.color = ifelse(V(graph_female_like)$name == "like", "lightblue", "orange"), vertex.label.cex = 0.8, edge.label.cex = 5)
# Significance of the term in comparison between networks
words_like <- setdiff(V(graph_female_like)$name, V(graph_male_like)$name)
words_like_degrees <- degree(induced.subgraph(graph = graph_female_like, vids = words_like))
df_like <- data.frame(Words = words_like, Degree = words_like_degrees)
df_like[order(-df_like$Degree),]

# "love"
# Male
male_love <- neighbors(graph_male_small, "love")
male_love <- names(V(graph_male_small)[male_love])
graph_male_love <- induced.subgraph(graph = graph_male_small, vids = c("love", male_love))
# Female
female_love <- neighbors(graph_female_small, "love")
female_love <- names(V(graph_female_small)[female_love])
graph_female_love <- induced.subgraph(graph = graph_female_small, vids = c("love", female_love))
# Graphical representation
# Male
plot.igraph(graph_male_love, vertex.size = degree(graph_male_love)/8, vertex.frame.color = 'gray', vertex.color = ifelse(V(graph_male_love)$name == "love", "lightblue", "orange"), vertex.label.cex = 0.8, edge.label.cex = 5)
# Female
plot.igraph(graph_female_love, vertex.size = degree(graph_female_love)/6, vertex.frame.color = 'gray', vertex.color = ifelse(V(graph_female_love)$name == "love", "lightblue", "orange"), vertex.label.cex = 0.8, edge.label.cex = 5)
# Significance of the term in comparison between networks
# Female
words_love <- setdiff(V(graph_female_love)$name, V(graph_male_love)$name)
words_love_degrees <- degree(induced.subgraph(graph = graph_female_love, vids = words_love))
df_love <- data.frame(Words = words_love, Degree = words_love_degrees)
df_love[order(-df_love$Degree),]
# Male
words_love2 <- setdiff(V(graph_male_love)$name, V(graph_female_love)$name)
words_love2_degrees <- degree(induced.subgraph(graph = graph_male_love, vids = words_love2))
df_love2 <- data.frame(Words = words_love2, Degree = words_love2_degrees)
df_love2[order(-df_love2$Degree),]

# "hate"
# Male
male_hate <- neighbors(graph_male, "hate")
male_hate <- names(V(graph_male)[male_hate])
graph_male_hate <- induced.subgraph(graph = graph_male, vids = c("hate", male_hate))
# Female
female_hate <- neighbors(graph_female, "hate")
female_hate <- names(V(graph_female)[female_hate])
graph_female_hate <- induced.subgraph(graph = graph_female, vids = c("hate", female_hate))
# Graphical representation
# Male
plot.igraph(graph_male_hate, vertex.size = degree(graph_male_hate)/8, vertex.frame.color = 'gray', vertex.color = ifelse(V(graph_male_hate)$name == "hate", "lightblue", "orange"), vertex.label.cex = 0.8, edge.label.cex = 5)
# Female
plot.igraph(graph_female_hate, vertex.size = degree(graph_female_hate)/6, vertex.frame.color = 'gray', vertex.color = ifelse(V(graph_female_hate)$name == "hate", "lightblue", "orange"), vertex.label.cex = 0.8, edge.label.cex = 5)
# Significance of the term in comparison between networks
words_hate <- setdiff(V(graph_female_hate)$name, V(graph_male_hate)$name)
words_hate_degrees <- degree(induced.subgraph(graph = graph_female_hate, vids = words_hate))
df_hate <- data.frame(Words = words_hate, Degree = words_hate_degrees)
df_hate[order(-df_hate$Degree),]

# "people"
# Male
male_people <- neighbors(graph_male_small, "people")
male_people <- names(V(graph_male_small)[male_people])
graph_male_people <- induced.subgraph(graph = graph_male_small, vids = c("people", male_people))
# Female
female_people <- neighbors(graph_female_small, "people")
female_people <- names(V(graph_female_small)[female_people])
graph_female_people <- induced.subgraph(graph = graph_female_small, vids = c("people", female_people))
# Graphical representation
# Male
plot.igraph(graph_male_people, vertex.size = degree(graph_male_people)/8, vertex.frame.color = 'gray', vertex.color = ifelse(V(graph_male_people)$name == "people", "lightblue", "orange"), vertex.label.cex = 0.8, edge.label.cex = 5)
# Female
plot.igraph(graph_female_people, vertex.size = degree(graph_female_people)/6, vertex.frame.color = 'gray', vertex.color = ifelse(V(graph_female_people)$name == "people", "lightblue", "orange"), vertex.label.cex = 0.8, edge.label.cex = 5)
# Significance of the term in comparison between networks
words_people <- setdiff(V(graph_female_people)$name, V(graph_male_people)$name)
words_people_degrees <- degree(induced.subgraph(graph = graph_female_people, vids = words_people))
df_people <- data.frame(Words = words_people, Degree = words_people_degrees)
df_people[order(-df_people$Degree),]

# "life"
# Male
male_life <- neighbors(graph_male_small, "life")
male_life <- names(V(graph_male_small)[male_life])
graph_male_life <- induced.subgraph(graph = graph_male_small, vids = c("life", male_life))
# Female
female_life <- neighbors(graph_female_small, "life")
female_life <- names(V(graph_female_small)[female_life])
graph_female_life <- induced.subgraph(graph = graph_female_small, vids = c("life", female_life))
# Graphical representation
# Male
plot.igraph(graph_male_life, vertex.size = degree(graph_male_life)/8, vertex.frame.color = 'gray', vertex.color = ifelse(V(graph_male_life)$name == "life", "lightblue", "orange"), vertex.label.cex = 0.8, edge.label.cex = 5)
# Female
plot.igraph(graph_female_life, vertex.size = degree(graph_female_life)/6, vertex.frame.color = 'gray', vertex.color = ifelse(V(graph_female_life)$name == "life", "lightblue", "orange"), vertex.label.cex = 0.8, edge.label.cex = 5)
# Significance of the term in comparison between networks
words_life <- setdiff(V(graph_female_life)$name, V(graph_male_life)$name)
words_life_degrees <- degree(induced.subgraph(graph = graph_female_life, vids = words_life))
df_life <- data.frame(Words = words_life, Degree = words_life_degrees)
df_life[order(-df_life$Degree),]

# "world"
# Male
male_world <- neighbors(graph_male_small, "world")
male_world <- names(V(graph_male_small)[male_world])
graph_male_world <- induced.subgraph(graph = graph_male_small, vids = c("world", male_world))
# Female
female_world <- neighbors(graph_female_small, "world")
female_world <- names(V(graph_female_small)[female_world])
graph_female_world <- induced.subgraph(graph = graph_female_small, vids = c("world", female_world))
# Graphical representation
# Male
plot.igraph(graph_male_world, vertex.size = degree(graph_male_world)/8, vertex.frame.color = 'gray', vertex.color = ifelse(V(graph_male_world)$name == "world", "lightblue", "orange"), vertex.label.cex = 0.8, edge.label.cex = 5)
# Female
plot.igraph(graph_female_world, vertex.size = degree(graph_female_world)/6, vertex.frame.color = 'gray', vertex.color = ifelse(V(graph_female_world)$name == "world", "lightblue", "orange"), vertex.label.cex = 0.8, edge.label.cex = 5)
# Significance of the term in comparison between networks
words_world <- setdiff(V(graph_female_world)$name, V(graph_male_world)$name)
words_world_degrees <- degree(induced.subgraph(graph = graph_female_world, vids = words_world))
df_world <- data.frame(Words = words_world, Degree = words_world_degrees)
df_world[order(-df_world$Degree),]

# "shit"
# Male
male_shit <- neighbors(graph_male_small, "shit")
male_shit <- names(V(graph_male_small)[male_shit])
graph_male_shit <- induced.subgraph(graph = graph_male_small, vids = c("shit", male_shit))
# Female
female_shit <- neighbors(graph_female_small, "shit")
female_shit <- names(V(graph_female_small)[female_shit])
graph_female_shit <- induced.subgraph(graph = graph_female_small, vids = c("shit", female_shit))
# Graphical representation
# Male
plot.igraph(graph_male_shit, vertex.size = degree(graph_male_shit)/8, vertex.frame.color = 'gray', vertex.color = ifelse(V(graph_male_shit)$name == "shit", "lightblue", "orange"), vertex.label.cex = 0.8, edge.label.cex = 5)
# Female
plot.igraph(graph_female_shit, vertex.size = degree(graph_female_shit)/6, vertex.frame.color = 'gray', vertex.color = ifelse(V(graph_female_shit)$name == "shit", "lightblue", "orange"), vertex.label.cex = 0.8, edge.label.cex = 5)
# Significance of the term in comparison between networks
words_shit <- setdiff(V(graph_female_shit)$name, V(graph_male_shit)$name)
words_shit_degrees <- degree(induced.subgraph(graph = graph_female_shit, vids = words_shit))
df_shit <- data.frame(Words = words_shit, Degree = words_shit_degrees)
df_shit[order(-df_shit$Degree),]

# "fuck"
# Male
male_fuck <- neighbors(graph_male_small, "fuck")
male_fuck <- names(V(graph_male_small)[male_fuck])
graph_male_fuck <- induced.subgraph(graph = graph_male_small, vids = c("fuck", male_fuck))
# Female
female_fuck <- neighbors(graph_female_small, "fuck")
female_fuck <- names(V(graph_female_small)[female_fuck])
graph_female_fuck <- induced.subgraph(graph = graph_female_small, vids = c("fuck", female_fuck))
# Graphical representation
# Male
plot.igraph(graph_male_fuck, vertex.size = degree(graph_male_fuck)/8, vertex.frame.color = 'gray', vertex.color = ifelse(V(graph_male_fuck)$name == "fuck", "lightblue", "orange"), vertex.label.cex = 0.8, edge.label.cex = 5)
# Female
plot.igraph(graph_female_fuck, vertex.size = degree(graph_female_fuck)/6, vertex.frame.color = 'gray', vertex.color = ifelse(V(graph_female_fuck)$name == "fuck", "lightblue", "orange"), vertex.label.cex = 0.8, edge.label.cex = 5)
# Significance of the term in comparison between networks
words_fuck <- setdiff(V(graph_female_fuck)$name, V(graph_male_fuck)$name)
words_fuck_degrees <- degree(induced.subgraph(graph = graph_female_fuck, vids = words_fuck))
df_fuck <- data.frame(Words = words_fuck, Degree = words_fuck_degrees)
df_fuck[order(-df_fuck$Degree),]

# Communities-Detection
# Male
louv_male <- cluster_louvain(graph_male_small)
coord_male <- layout.fruchterman.reingold(graph_male_small)
plot.igraph(graph_male_small, layout = coord_male, vertex.color = louv_male$membership)

# Female
louv_female <- cluster_louvain(graph_female_small)
coord_female <- layout.fruchterman.reingold(graph_female_small)
plot.igraph(graph_female_small, layout = coord_female, size = 5, vertex.color = louv_female$membership)

# The analysis with the graphs is not very precise
# We keep Membership and Names
# This gives us a tabular documentation of the different groups
# Male
head(names(louv_male))
# Female
head(names(louv_female))
# Data-Frames
# Male
df_network_male <- data.frame(Name = louv_male$names, Membership = louv_male$membership)
head(df_network_male)
# Female
df_network_female <- data.frame(Name = louv_female$names, Membership = louv_female$membership)
head(df_network_female)

# A little order...
# Male
df_network_male_order <- df_network_male[order(df_network_male$Membership),]
df_network_male_order
write.csv(df_network_male_order,"Communities_male.csv", row.names = FALSE)
# Female
df_network_female_order <- df_network_female[order(df_network_female$Membership),]
df_network_female_order
write.csv(df_network_female_order,"Communities_female.csv", row.names = FALSE)