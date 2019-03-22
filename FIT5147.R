install.packages("keras")
install.packages("tidyr")
library(keras)
library(tidyr)

data <- read.csv('AAPL-2.csv', header=TRUE, sep=',', dec='.', stringsAsFactors=FALSE)
df <- cbind(data[1], data.frame(sapply(data[2:7], function(x) as.numeric(as.character(x)))))
head(df)
str(df)

# Only first 2 columns
df2 <- df[1:2]
df2 <- na.omit(df2)

# Min max scaling
normalize <- function(x)
{
  return((x- min(x)) /(max(x)-min(x)))
}

# To get a vector, use apply instead of lapply
df2[2] <- as.data.frame(lapply(df2[2], normalize))
summary(df2)


array_reshape(df, c(2, 2))