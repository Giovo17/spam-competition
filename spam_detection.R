# spam-script

# Import libraries
library(tm)
library(superml)
library(caret)
library(stringr)
library(pROC)
library(ggplot2)
library(plyr)
library(qdap)



set.seed(17)


# Import dataset

# Online from github repo
#df_train = read.csv('')
#df_test = read.csv('')


# Local from disk
setwd("~/Documents/University/Data\ Science/1°\ Year\ (2022-2023)/Statistical\ Learning\ (1)/Exercises\ -\ Statistical\ Learning/spam-competition")
df_train = read.csv("data/spam_train.csv", sep=",", encoding="latin1")
df_test = read.csv("data/spam_test.csv", sep=";", encoding="latin1")


head(df_train)
print(xtable::xtable(head(df_train), type="latex", digits=2))



### EDA ###

jpeg(file="LateX_project/img/class_barplot.jpeg", width=6, height=6, units='in', res=200)

ggplot(df_train, aes(x=as.factor(class) )) +
  geom_bar(color="#6b9bc3", fill=rgb(0.1,0.4,0.5,0.7) ) + 
  geom_text(stat='count', aes(x=class, label=paste(after_stat(count), " (", round(after_stat(count)/nrow(df_train)*100, digits=2), "%)", sep="")), vjust=-1) +
  xlab("class") +
  theme_minimal() +
  theme(text = element_text(size = 20))

dev.off()

# Unbalanced dataset



# extract message lengths

columns = c("X","strlen","wordlen","numlen","class") 
len_df = data.frame(matrix(nrow = 0, ncol = length(columns))) 
colnames(len_df) = columns

for (i in 1:nrow(df_train)) {
  
  #remove punctuations
  df_train[i, "email"] = gsub('[[:punct:] ]+',' ', df_train[i, "email"])
  
  new_row = c(X = df_train[i, "X"], strlen=nchar(df_train[i, "email"]), wordlen=stringr::str_count(df_train[i, "email"], "\\w+"), numlen=stringr::str_count(df_train[i, "email"], "[0-9]"), class=df_train[i, "class"])
  len_df = rbind(len_df, new_row)
}

colnames(len_df) = columns


len_df$strlen = as.numeric(len_df$strlen)
len_df$wordlen = as.numeric(len_df$wordlen)
len_df$numlen = as.numeric(len_df$numlen)
len_df$class = as.factor(len_df$class)


# EDA of message lengths

jpeg(file="LateX_project/img/strlen_histogram.jpeg", width=6, height=6, units='in', res=200)

ggplot(len_df, aes(x=strlen, fill=class)) +
  geom_histogram(aes(y=after_stat(density)) , binwidth=20, color="#6b9bc3", alpha=0.7, position = 'identity') +
  geom_rug(aes(x=strlen, y = NULL)) +
  scale_fill_manual(values=c("green", "orange")) +
  ylab("density") + 
  labs(fill="") +
  theme_minimal() +
  theme(text = element_text(size = 20), legend.position="top")

dev.off()


jpeg(file="LateX_project/img/wordlen_histogram.jpeg", width=6, height=6, units='in', res=200)

ggplot(len_df, aes(x=wordlen, fill=class)) +
  geom_histogram(aes(y=after_stat(density)), binwidth=4, color="#6b9bc3", alpha=0.7, position = 'identity') +
  geom_rug(aes(x=wordlen, y = NULL)) +
  scale_fill_manual(values=c("green", "orange")) +
  ylab("density") + 
  labs(fill="") +
  theme_minimal() +
  theme(text = element_text(size = 20), legend.position="top")

dev.off()


jpeg(file="LateX_project/img/numlen_histogram.jpeg", width=6, height=6, units='in', res=200)

ggplot(len_df, aes(x=numlen, fill=class)) +
  geom_histogram(aes(y=after_stat(density)), binwidth=3, color="#6b9bc3", alpha=0.7, position = 'identity') +
  geom_rug(aes(x=numlen, y = NULL)) +
  scale_fill_manual(values=c("green", "orange")) +
  ylab("density") + 
  labs(fill="") +
  theme_minimal() +
  theme(text = element_text(size = 20), legend.position="top")

dev.off()


# remove stopwords
df_train$email = qdap::rm_stopwords(df_train$email, tm::stopwords("en"))






# Transform class variable to integer: 1 = spam, 0 = ham
df_train$class = plyr::mapvalues(df_train$class, c("spam", "ham"), c("1", "0"))
df_train$class = as.numeric(df_train$class)



# --------------------------------------------------------------------------------------------------
### FEATURE EXTRACTION ###


# TF-IDF

# initialise the class
tfv = superml::TfIdfVectorizer$new(min_df=0.3, max_features=40, remove_stopwords=FALSE, 
                                   norm=TRUE, lowercase=TRUE) 
# min_df: removes the words with frequency lower than a threshold
# remove_stopwords: remove the words that end a phrase
# norm: normalize the columns
# lowercase: trasform all words to lowercase


# fit with training data
tfv$fit(df_train$email)

# create new features
train_tf_features <- tfv$transform(df_train$email)
data_train = data.frame(cbind(X = df_train$X, train_tf_features, class = df_train$class))


head(data_train)



# Include message length
len_df = subset(len_df, select = -c(class, wordlen))
data_train = merge(data_train, len_df, by="X")
data_train = subset(data_train, select = -X)


# --------------------------------------------------------------------------------------------------
### MODEL TRAINING ###


# Data partitioning
trainIndex = caret::createDataPartition(data_train$class, p = 0.8, list = FALSE, times = 1)

data_validation = data_train[-trainIndex,]
data_training = data_train[trainIndex,]


# Model fit
glm_fit = glm(class~., family=binomial, data=data_training) 
summary(glm_fit)




# --------------------------------------------------------------------------------------------------
### MODEL VALIDATION ###


glm_prediction_validation = predict.glm(glm_fit, newdata=data_validation, type="response")


glm_prediction_validation[glm_prediction_validation >= 0.5] = 1
glm_prediction_validation[glm_prediction_validation < 0.5] = 0


data_validation = data.frame(cbind(data_validation, predicted_class = glm_prediction_validation))


cm = as.matrix(table(Actual=data_validation$class, Predicted=data_validation$predicted_class))
print(cm)

accuracy = (cm[2,2]+cm[1,1]) / (cm[2,2]+cm[1,1]+cm[1,2]+cm[2,1])   # (tp+tn)/(tp+tn+fp+fn)
error_rate = 1 - accuracy
specificity = cm[1,1] / (cm[1,2]+cm[1,1])   # tn/(fp+tn)
sensitivity = cm[2,2] / (cm[1,2]+cm[2,2])   # tp/(tp+fp)
auc = pROC::auc(data_validation$class, data_validation$predicted_class)


print(accuracy)
print(error_rate)
print(specificity)
print(sensitivity)
print(auc)


roc_curve = pROC::roc(data_validation$class, data_validation$predicted_class)


jpeg(file="LateX_project/img/roc.jpeg", width=6, height=6, units='in', res=200)

ggroc(roc_curve, colour = '#6b9bc3', size = 2) +
  theme_minimal() +
  theme(text = element_text(size = 20))

dev.off()

coords(roc_curve, "best", ret = "threshold")



# Model retraining on the whole training data available (to let it see more data)

glm_fit = glm(class~., family=binomial, data=data_train)




# --------------------------------------------------------------------------------------------------
### TEST SET PREDICTION ###


columns_test = c("id_number","strlen", "numlen") 
len_df_test = data.frame(matrix(nrow = 0, ncol = length(columns_test))) 
colnames(len_df_test) = columns_test

for (i in 1:nrow(df_test)) {
  
  #remove punctuations
  df_test[i, "email"] = gsub('[[:punct:] ]+',' ', df_test[i, "email"])
  
  new_row = c(X = df_test[i, "id_number"], strlen=nchar(df_test[i, "email"]), numlen=stringr::str_count(df_test[i, "email"], "[0-9]"))
  len_df_test = rbind(len_df_test, new_row)
}

colnames(len_df_test) = columns_test


len_df_test$strlen = as.numeric(len_df_test$strlen)


# remove stopwords
df_test$email = qdap::rm_stopwords(df_test$email, tm::stopwords("en"))


# feature extraction from the test set
test_tf_features <- tfv$transform(df_test$email)
data_test = data.frame(cbind(id_number = df_test$id_number, test_tf_features))


# Include message length
data_test = merge(data_test, len_df_test, by="id_number")



glm_prediction_test = predict.glm(glm_fit, newdata=data_test, type="response")

glm_prediction_test[glm_prediction_test >= 0.5] = 1
glm_prediction_test[glm_prediction_test < 0.5] = 0


data_test = data.frame(cbind(data_test, predicted_class = glm_prediction_test))
head(data_test)

df_test = read.csv("data/spam_test.csv", sep=";", encoding="latin1")
test_results = merge(data_test, df_test, by="id_number")
test_results = subset(test_results, select = c(id_number, email, predicted_class))

head(test_results)


write.csv(test_results, file="data/spam_test_results.csv")





