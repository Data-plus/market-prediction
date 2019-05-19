# install.packages("readr")
# install.packages("tidyverse")
# install.packages("tidyquant")
# install.packages("tm")
# install.packages("syuzhet")
# install.packages("keras")

library(syuzhet) #for sentiments analysis
library(readr)
library(dplyr)
library(tidyverse)
library(ggplot2)
library(tm)
library(tidyquant)


library(keras)
# install_keras()

#########################################################################################################################################

# Reading Data
data <- read_csv("news/abcnews-date-text.csv")

head(data)
dim(data)
data_sample <- data
data_sample <- data[1:1000, 1:2]

# Text cleansing
corpus <- Corpus(VectorSource(data_sample$headline_text))
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeWords, stopwords('english'))
corpus <- tm_map(corpus, content_transformer(stripWhitespace))
data_sample_text <- data.frame(text=sapply(corpus,identity), stringsAsFactors = F)
data_sample <- cbind("publish_date"=data_sample$publish_date, "headline_text"=data_sample_text)

# Semtiment Analysis
sentiment <- get_nrc_sentiment(data_sample$text)
df.sentiment <- data.frame(t(sentiment))


df.cleaned <- data.frame(rowSums(df.sentiment[1:nrow(sentiment)]))     


#Transformation and  cleaning
names(df.cleaned)[1] <- "count"
df.cleaned <- cbind("sentiment" = rownames(df.cleaned), df.cleaned)
rownames(df.cleaned) <- NULL
df.result<-df.cleaned[1:10,]

qplot(sentiment, data=df.result, weight=count, geom="bar",fill=sentiment)+ggtitle("Entire News headlines sentiment analysis")

df.result


# Yearly sentiment analysis




#########################################################################################################################################
# Market Prediction  1 Feature

# Stock market
## S&P 500 Graph from 2010
snp.data <- read_csv("./Documents/Assignment 2/s&p500.csv")
head(snp.data)
snp.close <- snp.data[2517:3599, c(1,5)]  # Date & Closing price
head(snp.close) # 2010-01-04
tail(snp.close) # 2014-04-23

snp.data[2517:3599,1:7] %>% # Only from 2010
  ggplot(aes(x = Date, y= Close, open=Open, high = High, low = Low, close = Close)) + 
  geom_candlestick() +
  labs(title = "S&P500 Candle Stick",
       subtitle = "BBands with SMA, GLM 7 Smoothing",
       y = "Closing Price", x = "") + stat_smooth(formula=y~poly(x,7), method="glm") +
  theme_light()


# Find difference between t+1 and t
snp.diff = diff(snp.close$Close, differences = 1)
snp.diff


## Data preperation
# Change it to supervised (k step lags)
lag_transform <- function(x, k= 1){
  
  lagged =  c(rep(NA, k), x[1:(length(x)-k)])
  DF = as.data.frame(cbind(lagged, x))
  colnames(DF) <- c( paste0('x-', k), 'x')
  DF[is.na(DF)] <- 0
  return(DF)

}
snp.supervised = lag_transform(snp.diff, 1)
head(snp.supervised)


# Data Split

N = nrow(snp.supervised)
n = round(N *0.8, digits = 0)
train = snp.supervised[1:n, ]
test  = snp.supervised[(n+1):N,  ]



## scale data
scale_data = function(train, test, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_train = ((x - min(x) ) / (max(x) - min(x)  ))
  std_test  = ((test - min(x) ) / (max(x) - min(x)  ))
  
  scaled_train = std_train *(fr_max -fr_min) + fr_min
  scaled_test = std_test *(fr_max -fr_min) + fr_min
  
  return( list(scaled_train = as.vector(scaled_train), scaled_test = as.vector(scaled_test) ,scaler= c(min =min(x), max = max(x))) )
  
}

## inverse-transform
invert_scaling = function(scaled, scaler, feature_range = c(0, 1)){
  min = scaler[1]
  max = scaler[2]
  t = length(scaled)
  mins = feature_range[1]
  maxs = feature_range[2]
  inverted_dfs = numeric(t)
  
  for( i in 1:t){
    X = (scaled[i]- mins)/(maxs - mins)
    rawValues = X *(max - min) + min
    inverted_dfs[i] <- rawValues
  }
  return(inverted_dfs)
}


Scaled = scale_data(train, test, c(-1, 1))

y_train = Scaled$scaled_train[, 2]
x_train = Scaled$scaled_train[, 1]

plot(x_train, col="dark blue", main="Min Max scaling") # Try plotting

# Reshape the input to 3-dim
dim(x_train) <- c(length(x_train), 1, 1)
X_shape2 = dim(x_train)[2]
X_shape3 = dim(x_train)[3]


#########################################################################################################################################
# Modelling

units = 1
batch_size = 1

es_callback <- callback_early_stopping(monitor='val_mean_absolute_error', min_delta=0, patience=2, verbose=0)


model <- keras_model_sequential() %>%
  layer_lstm(units, batch_input_shape=c(batch_size, X_shape2, X_shape3), stateful = TRUE, return_sequences = TRUE)%>%
  layer_dropout(0.25) %>%
  layer_lstm(units, batch_input_shape=c(batch_size, X_shape2, X_shape3), stateful = TRUE, return_sequences = TRUE)%>%
  layer_dropout(0.25) %>%
  layer_lstm(units, batch_input_shape=c(batch_size, X_shape2, X_shape3), stateful = TRUE, return_sequences = FALSE)%>%
  layer_dropout(0.25) %>%
  layer_dense(units = 1)

model %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam',
  metrics = c('mae')
)
summary(model)

history <- model %>% fit(
  x_train, y_train,
  epochs = 15,
  batch_size = 1,
  callback = list(callback_tensorboard("logs/run_b")),
  shiffle = FALSE,
  validation_split = 0.2
)

history

tensorboard("logs/run_b")

#########################################################################################################################################
## Validation
val <- read_csv("./Documents/Assignment 2/full.csv")

val <- snp.data[2517:nrow(snp.data), c(1,5)] 

val.diff = diff(val$Close, differences = 1)
val.supervised = lag_transform(val.diff, 1)
head(val.supervised)

scale_data_val = function(data, feature_range = c(0, 1)) {
  x = data
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_data = ((x - min(x) ) / (max(x) - min(x)  ))

  scaled_data = std_data *(fr_max -fr_min) + fr_min

  return( list(scaled_data = as.vector(scaled_data) ,scaler= c(min =min(x), max = max(x))) )
  
}


Scaled = scale_data_val(val.supervised, c(-1, 1))
x_val = Scaled$scaled_data[, 1]
dim(x_val) <- c(length(x_val), 1, 1)

L = length(x_val)
scaler = Scaled$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_val[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + val$Close[(i)]
  # store
  predictions[i] <- yhat
}

val_df <- cbind(val[2:nrow(val),1], data.frame("Close"=predictions))

val_df

actual <- read_csv("./Documents/Assignment 2/full.csv")[,c(1,5)]

ggplot()+
  geom_line(data= val_df[1:1084,], aes(y=Close, x=Date, color = 'Prediction')) + 
  geom_line(data= val[1:1084,], aes(y=Close, x=Date, color = 'Actual')) + 
  theme_classic() + 
  labs(title = "S&P500 prediction", subtitle = "Closing Price")

val_df[1084,]

# Calculate RMSE
RMSE <- function(m,o){
  sqrt(mean((m-o)**2))
}

RMSE(val$Close[1:1084], val_df$Close[1:1084])


#########################################################################################################################################
# Market Prediction  Lookback

## scale data
scale_data = function(train, test, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_train = ((x - min(x) ) / (max(x) - min(x)  ))
  std_test  = ((test - min(x) ) / (max(x) - min(x)  ))
  
  scaled_train = std_train *(fr_max -fr_min) + fr_min
  scaled_test = std_test *(fr_max -fr_min) + fr_min
  
  return( list(scaled_train = as.vector(scaled_train), scaled_test = as.vector(scaled_test) ,scaler= c(min =min(x), max = max(x))) )
  
}

## inverse-transform
invert_scaling = function(scaled, scaler, feature_range = c(0, 1)){
  min = scaler[1]
  max = scaler[2]
  t = length(scaled)
  mins = feature_range[1]
  maxs = feature_range[2]
  inverted_dfs = numeric(t)
  
  for( i in 1:t){
    X = (scaled[i]- mins)/(maxs - mins)
    rawValues = X *(max - min) + min
    inverted_dfs[i] <- rawValues
  }
  return(inverted_dfs)
}


# Stock market
## S&P 500 Graph from 2010
snp.data <- read_csv("./Documents/Assignment 2/s&p500.csv")
head(snp.data)
snp.close <- snp.data[2517:3599, c(1,5)]  # Date & Closing price & Volume
head(snp.close) # 2010-01-04
tail(snp.close) # 2014-04-23

# Data Split
N = nrow(snp.close)
n = round(N *0.8, digits = 0)
train = snp.close[1:n, 2]
test  = snp.close[(n+1):N, 2]

Scaled = scale_data(train, test, c(-1, 1))

training_scaled <- Scaled$scaled_train$Close
testing_scaled <- Scaled$scaled_test$Close


lookback <- 30

X_train <- t(sapply(1:(length(training_scaled)-lookback), function(x) training_scaled[x:(x+lookback -1)]))
y_train <- sapply((lookback +1):(length(training_scaled)), function(x) training_scaled[x])



# Reshape the input to 3-dim
X_train <- array(X_train, dim=c(nrow(X_train),lookback,1))
dim(X_train)
plot(X_train)
num_samples <- dim(X_train)[1]
num_steps <- dim(X_train)[2]
num_features <- dim(X_train)[3]

c(num_samples, num_steps, num_features)


X_test <- t(sapply(1:(length(testing_scaled)-lookback), function(x) testing_scaled[x:(x+lookback -1)]))

# Reshape the input to 3-dim
X_test <- array(X_test, dim=c(nrow(X_test),lookback,1))
plot(X_train)
dim(X_test)


#########################################################################################################################################
# Modelling

units = 4
batch_size = 1

#es_callback <- callback_early_stopping(monitor='val_mean_absolute_error', min_delta=0, patience=2, verbose=0)


model <- keras_model_sequential() %>%
  layer_lstm(units, batch_input_shape = c(batch_size, num_steps, num_features), return_sequences = TRUE, stateful = TRUE)%>%
  layer_dropout(0.25) %>%
  layer_lstm(units, input_shape=c(num_steps, num_features),  return_sequences = FALSE)%>%
  layer_dropout(0.25) %>%
  layer_dense(units = 1)

model %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam',
  metrics = c('mae')
)
summary(model)

history <- model %>% fit(
  X_train, y_train,
  epochs = 15,
  batch_size = batch_size,
  callback = list(callback_tensorboard("logs/run_a")),
  shiffle = FALSE,
  validation_split = 0.2
)

plot(history)

pred_train <- predict(model, X_train, batch_size = 1)
pred_test <- predict(model, X_test, batch_size = 1)
plot(pred_train)

pred_train2 <- data.frame("time"=snp.close[(1+lookback):n, 1], "Close"=invert_scaling(pred_train, scaler, c(-1, 1)))
pred_test2 <- data.frame("time"=snp.close[(n+1):(N-lookback), 1], "Close"=invert_scaling(pred_test, scaler, c(-1, 1)))
plot(pred_test2)

ggplot()+
  geom_line(data= pred_train2, aes(y=Close, x=Date, color = 'Train')) + 
  geom_line(data= pred_test2, aes(y=Close, x=Date, color = 'Test')) + 
  geom_line(data = snp.close, aes(y=Close, x=Date, color = 'Real')) +
  theme_classic() + 
  labs(title = "S&P500 prediction", subtitle = "Closing Price")

# Calculate RMSE
RMSE <- function(m,o){
  sqrt(mean((m-o)**2))
}
RMSE(pred_test2$Close, snp.close[(n+1+lookback):N, 2]$Close)


tail(pred_train2)
head(pred_test2)

tensorboard("logs")

#########################################################################################################################################
# Market Prediction  2 Features


# Stock market
## S&P 500 Graph from 2010
snp.data <- read_csv("./Documents/Assignment 2/s&p500.csv")
head(snp.data)
snp.close <- snp.data[2517:3599, c(1,2,3,5)]  # Date & Closing price & Volume
head(snp.close) # 2010-01-04
tail(snp.close) # 2014-04-23


# Data Split
N = nrow(snp.close)
n = round(N *0.8, digits = 0)
train = snp.close[1:n, 2:4]
test  = snp.close[(n+1):N, 2:4]
head(train,2)


Scaled.f1 = scale_data(train$Close, test$Close, c(-1, 1))
Scaled.f2 = scale_data(train$High, test$High, c(-1, 1))
Scaled.f3 = scale_data(train$Open, test$Open, c(-1, 1))

full.diff <- diff(snp.close$Close, differences = 1)  # Difference
full.lags <- lag_transform(full.diff)  # Lag
train.diff = full.lags[1:n, ]
test.diff  = full.lags[(n+1):N, ]
diff.scale <- scale_data(train.diff, test.diff, c(-1, 1))

train.diff <- diff.scale$scaled_train
test.diff <- diff.scale$scaled_test

training_scaled.f1 <- Scaled.f1$scaled_train
training_scaled.f2 <- Scaled.f2$scaled_train
training_scaled.f3 <- Scaled.f3$scaled_train
#train.diff = lag_transform(diff(Scaled.f1$scaled_train, differences = 1), 1)

testing_scaled.f1 <- Scaled.f1$scaled_test
testing_scaled.f2 <- Scaled.f2$scaled_test
testing_scaled.f3 <- Scaled.f3$scaled_test
#test.diff = lag_transform(diff(Scaled.f1$scaled_test, differences = 1), 1)


lookback <- 30

X_train.f1 <- t(sapply(1:(length(training_scaled.f1)-lookback), function(x) training_scaled.f1[x:(x+lookback -1)]))
X_train.f2 <- t(sapply(1:(length(training_scaled.f2)-lookback), function(x) training_scaled.f2[x:(x+lookback -1)]))
X_train.f3 <- t(sapply(1:(length(training_scaled.f3)-lookback), function(x) training_scaled.f3[x:(x+lookback -1)]))
X_train.f4 <- t(sapply(1:(length(train.diff$`x-1`)-lookback), function(x) train.diff$`x-1`[x:(x+lookback-1)]))

# Predict Closing 
y_train <- sapply((lookback +1):(length(training_scaled.f1)), function(x) training_scaled.f1[x])
y_train <- sapply((lookback+1):(length(train.diff$x)), function(x) train.diff$x[x])


# Reshape the input to 3-dim
num_features = 4
X_train <- array(X_train, dim=c(nrow(X_train),lookback,num_features))
X_train[,,1] <- X_train.f1
X_train[,,2] <- X_train.f2
X_train[,,3] <- X_train.f3
X_train[,,4] <- X_train.f4

num_samples <- dim(X_train)[1]
num_steps <- dim(X_train)[2]
num_features <- dim(X_train)[3]

c(num_samples, num_steps, num_features)

X_test.f1 <- t(sapply(1:(length(testing_scaled.f1)-lookback), function(x) testing_scaled.f1[x:(x+lookback -1)]))
X_test.f2 <- t(sapply(1:(length(testing_scaled.f2)-lookback), function(x) testing_scaled.f2[x:(x+lookback -1)]))
X_test.f3 <- t(sapply(1:(length(testing_scaled.f3)-lookback), function(x) testing_scaled.f3[x:(x+lookback -1)]))
X_test.f4 <- t(sapply(1:(length(test.diff$`x-1`)-lookback), function(x) test.diff$`x-1`[x:(x+lookback-1)]))


# Reshape the input to 3-dim
X_test <- array(X_test.f1, dim=c(nrow(X_test.f1),lookback, num_features))
X_test[,,1] <- X_test.f1
X_test[,,2] <- X_test.f2
X_test[,,3] <- X_test.f3
X_test[,,4] <- X_test.f4


plot(X_train)
dim(X_test)


#########################################################################################################################################
# Modelling

units = 4
batch_size = 1

#es_callback <- callback_early_stopping(monitor='val_mean_absolute_error', min_delta=0, patience=2, verbose=0)


model <- keras_model_sequential() %>%
  layer_lstm(units, batch_input_shape = c(batch_size, num_steps, num_features), return_sequences = TRUE, stateful = TRUE)%>%
  layer_dropout(0.25) %>%
  layer_lstm(units, input_shape=c(num_steps, num_features),  return_sequences = FALSE)%>%
  layer_dropout(0.25) %>%
  layer_dense(units = 1)

model %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam',
  metrics = c('mae')
)
summary(model)

history <- model %>% fit(
  X_train, y_train,
  epochs = 10,
  batch_size = batch_size,
  #callback = list(callback_tensorboard("logs/run_a")),
  shiffle = FALSE,
  validation_split = 0.2
)

plot(history)

pred_train <- predict(model, X_train, batch_size = 1)
pred_test <- predict(model, X_test, batch_size = 1)
plot(pred_train)

pred_train2 <- data.frame("time"=snp.close[(1+lookback):n, 1], "Close"=invert_scaling(pred_train, scaler, c(-1, 1)))
pred_test2 <- data.frame("time"=snp.close[(n+1):(N-lookback), 1], "Close"=invert_scaling(pred_test, scaler, c(-1, 1)))
plot(pred_test2)

p.coh <- ggplot()+
  geom_line(data= pred_train2, aes(y=Close, x=Date, color = 'Train')) + 
  geom_line(data= pred_test2, aes(y=Close, x=Date, color = 'Test')) + 
  geom_line(data = snp.close, aes(y=Close, x=Date, color = 'Real')) +
  theme_classic() + 
  labs(title = "S&P500 prediction", subtitle = "Closing Price")
p.coh

RMSE(pred_test2$Close, snp.close[(n+1+lookback):N, ]$Close)


invert_scaling(pred_train, scaler, c(-1, 1)) + snp.close$Close[(1+lookback):n]-1500
invert_scaling(pred_test, scaler, c(-1, 1)) + snp.close$Close[(n+1):(N-lookback)]-1500

pred_train3 <- data.frame("time"=snp.close[(1+lookback):n, 1], 
                          "Close"=invert_scaling(pred_train, scaler, c(-1, 1)) + snp.close$Close[(1+lookback):n]-1600)
pred_test3 <- data.frame("time"=snp.close[(n+1):(N-lookback), 1], 
                         "Close"=invert_scaling(pred_test, scaler, c(-1, 1)) + snp.close$Close[(n+1):(N-lookback)]-1500)

p.coh2 <- ggplot()+
  geom_line(data= pred_train3, aes(y=Close, x=Date, color = 'Train')) + 
  geom_line(data= pred_test3, aes(y=Close, x=Date, color = 'Test')) + 
  geom_line(data = snp.close, aes(y=Close, x=Date, color = 'Real')) +
  theme_classic() + 
  labs(title = "S&P500 prediction", subtitle = "Closing Price")
p.coh2

invert_scaling(pred_train, scaler, c(-1, 1)) -1500

RMSE(pred_test3$Close, snp.close[(n+1+lookback):N, ]$Close)


# Calculate RMSE
RMSE <- function(m,o){
  sqrt(mean((m-o)**2))
}


tail(pred_train2)
head(pred_test2)



#########################################################################################################################################
#########################################################################################################################################
# Final Modelling, All combined

# Stock market
## S&P 500 Graph from 2010
snp.data <- read_csv("./Documents/Assignment 2/s&p500.csv")
head(snp.data)
snp.close <- snp.data[2517:3599, c(1,2,3,5)]  # Date & Closing price & Volume
head(snp.close) # 2010-01-04
tail(snp.close) # 2014-04-23


# Data Split
N = nrow(snp.close)
n = round(N *0.8, digits = 0)
train = snp.close[1:n, 2:4]
test  = snp.close[(n+1):N, 2:4]
head(train,2)


Scaled.f1 = scale_data(train$Close, test$Close, c(-1, 1))
Scaled.f2 = scale_data(train$High, test$High, c(-1, 1))
Scaled.f3 = scale_data(train$Open, test$Open, c(-1, 1))

full.diff <- diff(snp.close$Close, differences = 1)  # Difference
full.lags <- lag_transform(full.diff)  # Lag
train.diff = full.lags[1:n, ]
test.diff  = full.lags[(n+1):N, ]
diff.scale <- scale_data(train.diff, test.diff, c(-1, 1))

train.diff <- diff.scale$scaled_train
test.diff <- diff.scale$scaled_test

training_scaled.f1 <- Scaled.f1$scaled_train
training_scaled.f2 <- Scaled.f2$scaled_train
training_scaled.f3 <- Scaled.f3$scaled_train
#train.diff = lag_transform(diff(Scaled.f1$scaled_train, differences = 1), 1)

testing_scaled.f1 <- Scaled.f1$scaled_test
testing_scaled.f2 <- Scaled.f2$scaled_test
testing_scaled.f3 <- Scaled.f3$scaled_test
#test.diff = lag_transform(diff(Scaled.f1$scaled_test, differences = 1), 1)


lookback <- 30

X_train.f1 <- t(sapply(1:(length(training_scaled.f1)-lookback), function(x) training_scaled.f1[x:(x+lookback -1)]))
X_train.f2 <- t(sapply(1:(length(training_scaled.f2)-lookback), function(x) training_scaled.f2[x:(x+lookback -1)]))
X_train.f3 <- t(sapply(1:(length(training_scaled.f3)-lookback), function(x) training_scaled.f3[x:(x+lookback -1)]))
X_train.f4 <- t(sapply(1:(length(train.diff$`x-1`)-lookback), function(x) train.diff$`x-1`[x:(x+lookback-1)]))

# Predict Closing 
y_train <- sapply((lookback +1):(length(training_scaled.f1)), function(x) training_scaled.f1[x])
y_train <- sapply((lookback+1):(length(train.diff$x)), function(x) train.diff$x[x])


# Reshape the input to 3-dim
num_features = 4
X_train <- array(X_train, dim=c(nrow(X_train),lookback,num_features))
X_train[,,1] <- X_train.f1
X_train[,,2] <- X_train.f2
X_train[,,3] <- X_train.f3
X_train[,,4] <- X_train.f4

num_samples <- dim(X_train)[1]
num_steps <- dim(X_train)[2]
num_features <- dim(X_train)[3]

c(num_samples, num_steps, num_features)

X_test.f1 <- t(sapply(1:(length(testing_scaled.f1)-lookback), function(x) testing_scaled.f1[x:(x+lookback -1)]))
X_test.f2 <- t(sapply(1:(length(testing_scaled.f2)-lookback), function(x) testing_scaled.f2[x:(x+lookback -1)]))
X_test.f3 <- t(sapply(1:(length(testing_scaled.f3)-lookback), function(x) testing_scaled.f3[x:(x+lookback -1)]))
X_test.f4 <- t(sapply(1:(length(test.diff$`x-1`)-lookback), function(x) test.diff$`x-1`[x:(x+lookback-1)]))


# Reshape the input to 3-dim
X_test <- array(X_test.f1, dim=c(nrow(X_test.f1),lookback, num_features))
X_test[,,1] <- X_test.f1
X_test[,,2] <- X_test.f2
X_test[,,3] <- X_test.f3
X_test[,,4] <- X_test.f4


plot(X_train)
dim(X_test)


#########################################################################################################################################
# Modelling

units = 4
batch_size = 1

#es_callback <- callback_early_stopping(monitor='val_mean_absolute_error', min_delta=0, patience=2, verbose=0)


model <- keras_model_sequential() %>%
  layer_lstm(units, batch_input_shape = c(batch_size, num_steps, num_features), return_sequences = TRUE, stateful = TRUE)%>%
  layer_dropout(0.25) %>%
  layer_lstm(units, input_shape=c(num_steps, num_features),  return_sequences = FALSE)%>%
  layer_dropout(0.25) %>%
  layer_dense(units = 1)

model %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam',
  metrics = c('mae')
)
summary(model)

history <- model %>% fit(
  X_train, y_train,
  epochs = 10,
  batch_size = batch_size,
  #callback = list(callback_tensorboard("logs/run_a")),
  shiffle = FALSE,
  validation_split = 0.2
)

plot(history)

pred_train <- predict(model, X_train, batch_size = 1)
pred_test <- predict(model, X_test, batch_size = 1)
plot(pred_train)

pred_train2 <- data.frame("time"=snp.close[(1+lookback):n, 1], "Close"=invert_scaling(pred_train, scaler, c(-1, 1)))
pred_test2 <- data.frame("time"=snp.close[(n+1):(N-lookback), 1], "Close"=invert_scaling(pred_test, scaler, c(-1, 1)))
plot(pred_test2)

p.coh <- ggplot()+
  geom_line(data= pred_train2, aes(y=Close, x=Date, color = 'Train')) + 
  geom_line(data= pred_test2, aes(y=Close, x=Date, color = 'Test')) + 
  geom_line(data = snp.close, aes(y=Close, x=Date, color = 'Real')) +
  theme_classic() + 
  labs(title = "S&P500 prediction", subtitle = "Closing Price")
p.coh

RMSE(pred_test2$Close, snp.close[(n+1+lookback):N, ]$Close)


invert_scaling(pred_train, scaler, c(-1, 1)) + snp.close$Close[(1+lookback):n]-1500
invert_scaling(pred_test, scaler, c(-1, 1)) + snp.close$Close[(n+1):(N-lookback)]-1500

pred_train3 <- data.frame("time"=snp.close[(1+lookback):n, 1], 
                          "Close"=invert_scaling(pred_train, scaler, c(-1, 1)) + snp.close$Close[(1+lookback):n]-1600)
pred_test3 <- data.frame("time"=snp.close[(n+1):(N-lookback), 1], 
                         "Close"=invert_scaling(pred_test, scaler, c(-1, 1)) + snp.close$Close[(n+1):(N-lookback)]-1500)

p.coh2 <- ggplot()+
  geom_line(data= pred_train3, aes(y=Close, x=Date, color = 'Train')) + 
  geom_line(data= pred_test3, aes(y=Close, x=Date, color = 'Test')) + 
  geom_line(data = snp.close, aes(y=Close, x=Date, color = 'Real')) +
  theme_classic() + 
  labs(title = "S&P500 prediction", subtitle = "Closing Price")
p.coh2

invert_scaling(pred_train, scaler, c(-1, 1)) -1500

RMSE(pred_test3$Close, snp.close[(n+1+lookback):N, ]$Close)


# Calculate RMSE
RMSE <- function(m,o){
  sqrt(mean((m-o)**2))
}


tail(pred_train2)
head(pred_test2)


