install.packages("keras")
install.packages("tidyr")
library(keras)
library(tensorflow)
library(tidyr)

data <- read.table('D:/Uni/2019-1/FIT5147 Visualisation/Assignment 2/Stocks/aapl.us.txt', header=TRUE, sep=',', dec='.', stringsAsFactors=FALSE)
head(data)
str(data)

# Only first 2 columns
df <- data[1:2]
df <- na.omit(df)

# Min max scaling
mm_normalize <- function(x)
{
  return((x- min(x)) /(max(x)-min(x)))
}

# To get a vector, use apply instead of lapply
df[2] <- as.data.frame(lapply(df[2], mm_normalize))
summary(df)


df_train <- df[1:floor(nrow(df)*0.8),2]
df_test <- df[ceiling(nrow(df)*0.8):nrow(df),2]


x_train <- matrix(nrow=length(df_train)-59, ncol = 60)
y_train <- matrix(nrow=length(df_train)-59, ncol = 1)

for (i in (60:length(df_train))){
  x_train[i-59,] <- matrix(df_train[(i-59):i])
  y_train[i-59] <- matrix(df_train[(i-59)])
}


n_timesteps <- length(x_train[1,])
n_predictions <- n_timesteps
batch_size <- length(x_train[,1])



# specify required arguments
batch_size = 60            # must be a common factor of both the train and test samples
units = 1                     # can adjust this, in model tuninig phase
dim(x_train) <- c(batch_size, n_timesteps, 1)
dim(x_train)



model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(60, 60, 1), stateful= TRUE)%>%
  layer_dense(units = 1)
summary(model)



model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)


Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}


