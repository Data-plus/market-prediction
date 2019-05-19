# Final Modelling, All combined

#########################################################################################################################################
# Function
#########################################################################################################################################

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

# Calculate RMSE
RMSE <- function(m,o){
  sqrt(mean((m-o)**2))
}




#########################################################################################################################################
# Stock market
#########################################################################################################################################

## S&P 500 Graph from 2010
snp.data <- read_csv("./Documents/Assignment 2/s&p500.csv")
head(snp.data)
snp.close <- snp.data[2517:3599, c(1,2,3,5)]  # Date & Closing price & Volume
head(snp.close) # 2010-01-04
tail(snp.close) # 2014-04-23



#########################################################################################################################################
# Sentiment
#########################################################################################################################################

# Reading Data
data <- read_csv("./Documents/Assignment 2/news/abcnews-date-text.csv")
#data_sample <- data[870690:1103663, 1:2]
data_sample <- data[510790:870420, 1:2]

corpus <- Corpus(VectorSource(data_sample$headline_text))
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeWords, stopwords('english'))
corpus <- tm_map(corpus, stemDocument)
corpus <- tm_map(corpus, content_transformer(stripWhitespace))
data_sample_text <- data.frame(text=sapply(corpus,identity), stringsAsFactors = F)
data_sample <- data.frame(cbind("publish_date"=data_sample$publish_date, "headline_text"=data_sample_text))

# Sentiment
data_sample$publish_date <- ymd(data_sample$publish_date)
sent <- c()

d <- unique(snp.close$Date)

i <- 1
while (i <= length(d)){
  sent[[i]] <- mean(get_sentiment(data_sample[which(data_sample$publish_date == d[i]),]$text, method="nrc", lang = "english"))
  i <- i+1
}
# Convert to DF
sent.df <- data.frame("Date"=d, "Sentiment" = sent)
# Drop NA
sent.df <- sent.df[complete.cases(sent.df$Sentiment), ]
head(sent.df)



#########################################################################################################################################
## Gold
#########################################################################################################################################
gold.data <- read_csv("./Documents/Assignment 2/WGC-GOLD_DAILY_USD.csv")
gold.data$Date <- as.Date(gold.data$Date, "%d/%m/%Y")
gold.data < -gold.data[!(gold.data$Date %in% gold.data$Date[!(gold.data$Date %in% snp.close$Date)]),]
gold.data <- gold.data[gold.data$Date <= '2014-04-23' & gold.data$Date >= '2010-01-04',]
head(gold.data)

#########################################################################################################################################
# WTI price data
#########################################################################################################################################
#install.packages("rjson")
library("rjson")
json_file <- "./Documents/Assignment 2/wti-daily_json.json"
json_data <- fromJSON(paste(readLines(json_file), collapse=""))
wti <- do.call(rbind, lapply(json_data, data.frame))
wti$Date <- as.Date(wti$Date, "%Y-%m-%d")
wti <- wti[wti$Date <= '2014-04-23' & wti$Date >= '2010-01-04',]
wti <- wti[!(wti$Date %in% wti$Date[!(wti$Date %in% snp.close$Date)]),]

head(wti)

#dr.wti <- dailyReturn(xts(wti$Price, wti$Date))







#########################################################################################################################################
# Data Processing
#########################################################################################################################################

# Data Concat
## Just to make it easier to see
combined <- cbind(snp.close, "WTI"=wti$Price, "Gold"=gold.data$Value, "Sentiment"=sent.df$Sentiment)
head(combined)

#install.packages("ggcorrplot")
library(ggcorrplot)
ggcorrplot(cor(combined[,4:ncol(combined)]), hc.order = TRUE,
           outline.col = "white",
           ggtheme = ggplot2::theme_gray, lab = TRUE,
           colors = c("#6D9EC1", "white", "#E46726"))


# Data Split
N = nrow(combined)
n = round(N *0.8, digits = 0)
train = combined[1:n, 2:ncol(combined)]
test  = combined[(n+1):N, 2:ncol(combined)]
head(train,2)

# Features
Scaled.f1 = scale_data(train$Close, test$Close, c(-1, 1))
Scaled.f2 = scale_data(train$High, test$High, c(-1, 1))
Scaled.f3 = scale_data(train$Open, test$Open, c(-1, 1))
Scaled.f4 = scale_data(train$WTI, test$WTI, c(-1, 1))
Scaled.f5 = scale_data(train$Gold, test$Gold, c(-1, 1))
Scaled.f6 = scale_data(train$Sentiment, test$Sentiment, c(-1, 1))

# Differences
full.diff <- diff(combined$Close, differences = 1)  # Difference
full.lags <- lag_transform(full.diff)  # Lag
train.diff = full.lags[1:n, ]  # Split
test.diff  = full.lags[(n+1):N, ]  # Split
diff.scale <- scale_data(train.diff, test.diff, c(-1, 1))  # Scaling
train.diff <- diff.scale$scaled_train  # Assign
test.diff <- diff.scale$scaled_test  # Assign


training_scaled.f1 <- Scaled.f1$scaled_train
training_scaled.f2 <- Scaled.f2$scaled_train
training_scaled.f3 <- Scaled.f3$scaled_train
training_scaled.f4 <- Scaled.f4$scaled_train
training_scaled.f5 <- Scaled.f5$scaled_train
training_scaled.f6 <- Scaled.f6$scaled_train

testing_scaled.f1 <- Scaled.f1$scaled_test
testing_scaled.f2 <- Scaled.f2$scaled_test
testing_scaled.f3 <- Scaled.f3$scaled_test
testing_scaled.f4 <- Scaled.f4$scaled_test
testing_scaled.f5 <- Scaled.f5$scaled_test
testing_scaled.f6 <- Scaled.f6$scaled_test


lookback <- 30

X_train.f1 <- t(sapply(1:(length(training_scaled.f1)-lookback), function(x) training_scaled.f1[x:(x+lookback -1)]))
X_train.f2 <- t(sapply(1:(length(training_scaled.f2)-lookback), function(x) training_scaled.f2[x:(x+lookback -1)]))
X_train.f3 <- t(sapply(1:(length(training_scaled.f3)-lookback), function(x) training_scaled.f3[x:(x+lookback -1)]))
X_train.f4 <- t(sapply(1:(length(training_scaled.f4)-lookback), function(x) training_scaled.f4[x:(x+lookback -1)]))
X_train.f5 <- t(sapply(1:(length(training_scaled.f5)-lookback), function(x) training_scaled.f5[x:(x+lookback -1)]))
X_train.f6 <- t(sapply(1:(length(training_scaled.f6)-lookback), function(x) training_scaled.f6[x:(x+lookback -1)]))
X_train.fd <- t(sapply(1:(length(train.diff$`x-1`)-lookback), function(x) train.diff$`x-1`[x:(x+lookback-1)]))

# Predict Closing 
#y_train <- sapply((lookback +1):(length(training_scaled.f1)), function(x) training_scaled.f1[x]) # Absolute
y_train <- sapply((lookback+1):(length(train.diff$x)), function(x) train.diff$x[x]) # Change

# Reshape the input to 3-dim
num_features = 7
X_train <- array(X_train, dim=c(nrow(X_train),lookback,num_features))
X_train[,,1] <- X_train.f1
X_train[,,2] <- X_train.f2
X_train[,,3] <- X_train.f3
X_train[,,4] <- X_train.f4
X_train[,,5] <- X_train.f5
X_train[,,6] <- X_train.f6
X_train[,,7] <- X_train.fd

num_samples <- dim(X_train)[1]
num_steps <- dim(X_train)[2]
num_features <- dim(X_train)[3]
c(num_samples, num_steps, num_features)

X_test.f1 <- t(sapply(1:(length(testing_scaled.f1)-lookback), function(x) testing_scaled.f1[x:(x+lookback -1)]))
X_test.f2 <- t(sapply(1:(length(testing_scaled.f2)-lookback), function(x) testing_scaled.f2[x:(x+lookback -1)]))
X_test.f3 <- t(sapply(1:(length(testing_scaled.f3)-lookback), function(x) testing_scaled.f3[x:(x+lookback -1)]))
X_test.f4 <- t(sapply(1:(length(testing_scaled.f4)-lookback), function(x) testing_scaled.f4[x:(x+lookback -1)]))
X_test.f5 <- t(sapply(1:(length(testing_scaled.f5)-lookback), function(x) testing_scaled.f5[x:(x+lookback -1)]))
X_test.f6 <- t(sapply(1:(length(testing_scaled.f6)-lookback), function(x) testing_scaled.f6[x:(x+lookback -1)]))
X_test.fd <- t(sapply(1:(length(test.diff$`x-1`)-lookback), function(x) test.diff$`x-1`[x:(x+lookback-1)]))


# Reshape the input to 3-dim
X_test <- array(X_test.f1, dim=c(nrow(X_test.f1),lookback, num_features))
X_test[,,1] <- X_test.f1
X_test[,,2] <- X_test.f2
X_test[,,3] <- X_test.f3
X_test[,,4] <- X_test.f4
X_test[,,4] <- X_test.f5
X_test[,,4] <- X_test.f6
X_test[,,4] <- X_test.fd
dim(X_test)


#########################################################################################################################################
# Modelling
#########################################################################################################################################

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
  callback = list(callback_tensorboard("logs/run_c")),
  shiffle = FALSE,
  validation_split = 0.2
)

history
tensorboard("logs")

# Prediction
pred_train <- predict(model, X_train, batch_size = 1)
pred_test <- predict(model, X_test, batch_size = 1)
plot(pred_train)

pred_train2 <- data.frame("time"=snp.close[(1+lookback):n, 1], 
                          "Close"=invert_scaling(pred_train, scaler, c(-1, 1)) + snp.close$Close[(1+lookback):n]-1600)
pred_test2 <- data.frame("time"=snp.close[(n+1):(N-lookback), 1], 
                         "Close"=invert_scaling(pred_test, scaler, c(-1, 1)) + snp.close$Close[(n+1):(N-lookback)]-1600)

p.coh2 <- ggplot()+
  geom_line(data= pred_train2, aes(y=Close, x=Date, color = 'Train')) + 
  geom_line(data= pred_test2, aes(y=Close, x=Date, color = 'Test')) + 
  geom_line(data = snp.close, aes(y=Close, x=Date, color = 'Real')) +
  theme_classic() + 
  labs(title = "S&P500 prediction", subtitle = "Closing Price")
p.coh2


RMSE(pred_test2$Close, snp.close[snp.close$Date >= "2013-06-13" & snp.close$Date <= "2014-03-11",]$Close)


