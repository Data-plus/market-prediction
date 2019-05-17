# install.packages("readr")
# install.packages("tidyverse")
#install.packages("tidyquant")
library(syuzhet) #for sentiments analysis
library(readr)
library(dplyr)
library(tidyverse)
library(ggplot2)
library(tm)
library(tidyquant)

library(keras)
install_keras(tensorflow = "gpu") # gpu version must be used


# Reading Data
data <- read_csv("news/abcnews-date-text.csv")

# create new columns: year and month
data$publish_date <- ymd(data$publish_date)
data$publish_year <- year(data$publish_date)
data$publish_month <- month(data$publish_date, label=TRUE)


tail(data)
dim(data)
data_sample <- data[data$publish_year==2017,]
#data_sample <- data[870690:1103663, ]
head(data_sample)




#===================================================================================================================================
# Text cleansing
corpus <- Corpus(VectorSource(data_sample$headline_text))
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeWords, stopwords('english'))
corpus <- tm_map(corpus, stemDocument)
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

s.2017 <- df.cleaned
s.2017['year']=2017

ss <- rbind(ss ,s.2017)

qplot(sentiment, data=ss, weight=count, geom="bar",fill=sentiment)+ggtitle("Entire News headlines sentiment analysis")

df.result


# Yearly sentiment analysis
# Change it to date
ss

ss %>%
  ggplot() +
  geom_bar(aes(y=count, x=sentiment, fill=sentiment),stat="identity") +
  scale_fill_brewer(palette = "Spectral") +
  coord_flip() +
  theme(axis.text.x = element_text(angle = 15)) +
  facet_wrap(~year)





#===================================================================================================================================


# Stock market
## S&P 500 Graph from 2010
snp.data <- read_csv("S&P.csv")
head(snp.data)
snp.close <- snp.data[1:nrow(snp.data), c(1,5)]  # Date & Closing price

head(snp.close)

snp.data[1:nrow(snp.data),1:7] %>% # Only from 2010
  ggplot(aes(x = Date, y= Close, open=Open, high = High, low = Low, close = Close)) + 
  geom_candlestick() +
  labs(title = "S&P500 Candle Stick",
       subtitle = "GLM 5 Smoothing",
       y = "Closing Price", x = "") + stat_smooth(formula=y~poly(x,5), method="glm") + theme_light()
  

head(snp.close)

#===================================================================================================================================

sandp.data <- read_csv("sandp500_final.csv")
sandp.data
sandp.data$date <- as.Date(sandp.data$date, "%d/%m/%Y")


## Add Size Column
df.s[df.s$`Market Cap` > 300000000000,] # Mega Cap
df.s[df.s$`Market Cap` > 10000000000,]  # Large Cap
df.s[df.s$`Market Cap` < 10000000000,]  # Small Cap

sandp.data$Size <- ifelse(sandp.data$`Market Cap` >=300000000000, "Mega Cap", 
                          ifelse(sandp.data$`Market Cap` >=10000000000, "Large Cap", "Small Cap"))



# Find return, industry wisely similar?

sandp.data[which(sandp.data$date == '2014-04-25'),]  # 472 stocks were traded since 25.04.2014
sandp.data[which(sandp.data$date == '2019-04-23'),]  # 505 stocks were traded since 23.04.2019



# Using intersect to find ticker symbols
traded <- intersect(sandp.data[which(sandp.data$date == '2019-04-23'),]$Ticker, 
                    sandp.data[which(sandp.data$date == '2014-04-25'),]$Ticker)

# Traded in both start, end print everything on start date.
df.s <- sandp.data[which(sandp.data$Ticker %in% traded & sandp.data$date == '2014-04-25'),][1:472, c(2,3,4,5,7,9,13,15)]
df.e <- sandp.data[which(sandp.data$Ticker %in% traded & sandp.data$date == '2019-04-23'),][1:472, c(2,9,13)]

df.s
df.e


# Percentage change (return)
(df.e$close - df.s$close)/df.s$close


# Take closing price for 'A'
library(quantmod)
A <- sandp.data[which(sandp.data$Ticker %in% traded & sandp.data$Ticker == 'AAPL' & sandp.data$date <= '2019-04-23'),]
a <- Delt(sandp.data[which(sandp.data$Ticker %in% traded & sandp.data$Ticker == 'AAPL'),]$close)
ts <- xts(A$close, A$date)
mr <- monthlyReturn(ts)


# Loop over all stocks
# Empty DF
df.km <-data.frame("mean.mr."=double(), "SD"=double(), "Name"=character(),
                   "Sector"=character(), "Size"=character(), stringsAsFactors=FALSE)

# Loop
for (i in (1: length(traded))){
  sname <- traded[i]
  A <- sandp.data[which(sandp.data$Ticker %in% traded & sandp.data$Ticker == sname & sandp.data$date <= '2019-04-23'),]
  a <- Delt(sandp.data[which(sandp.data$Ticker %in% traded & sandp.data$Ticker == sname),]$close)
  ts <- xts(A$close, A$date)
  mr <- monthlyReturn(ts)
  
  df.km_temp <- data.frame(sum(mr), "SD" = sd(mr), "Name"=c(sname), "Sector"=A$Sector[1], "Size"=A$`Size`[1])
  df.km <- rbind(df.km, df.km_temp)
  
}



df.km

df.kmeans <- kmeans(df.km[,1:2], centers = 3) # 8 sectors
summary(scale(df.km[,1:2]))

df.km$cluster <- as.factor(df.kmeans$cluster)
ggplot(data = df.km, aes(x = sum.mr., y = SD, colour = cluster)) + geom_point()
table(df.km$Size, df.km$cluster)



install.packages("NbClust")
library(NbClust)

nc <- NbClust(scale(df.km[,1:2]), min.nc = 2, max.nc = 15, method = "kmeans")
par(mfrow=c(1,1))
barplot(table(nc$Best.n[1,]),
        xlab="Numer of Clusters", ylab="Number of Criteria",
        main="Number of Clusters Chosen", col = 'light blue')

#===================================================================================================================================
# Correlation
## S&P Return
a <- Delt(snp.data$`Adj Close`)
mr <- monthlyReturn(xts(snp.data$`Adj Close`, snp.data$Date))

# Sentiment
mean(get_sentiment(data_sample$headline_text[1:10], method="nrc", lang = "english"))
data_sample$publish_date <- ymd(data_sample$publish_date)
unique(unlist(data_sample$publish_date))

# Unique Date from stock market data
d <- unique(sandp.data[which(sandp.data$Ticker %in% traded),]$date)

# Loop over headline, for each date from stock data, find sentiment
sent <- c()
i <- 1
while (i <= length(d)){
  sent[[i]] <- mean(get_sentiment(data_sample[which(data_sample$publish_date == d[i]),]$headline_text, method="nrc", lang = "english"))
  i <- i+1
}

# Convert to DF
sent.df <- data.frame("Date"=d, "Sentiment" = sent)
# Drop NA
sent.df <- sent.df[complete.cases(sent.df$Sentiment), ]


snp.ext <- snp.data[which(snp.data$Date >= "2014-04-25" & snp.data$Date <= "2017-12-29"),]
dr <- dailyReturn(xts(snp.ext$`Adj Close`, snp.ext$Date))
d1 <- cbind("sentiment"=sent.df$Sentiment, "close"=snp.ext$Close)
d1 <- data.frame(d1)

# Correlation heat map
# Daily News Sentiment
sent.df<-sent.df[!(sent.df$Date=="2017-07-03"),]
nrow(sent.df)


# S&P Daily return
snp.ext<-snp.ext[!(snp.ext$Date=="2017-07-03"),]
dr.snp <- dailyReturn(xts(snp.ext$Close, snp.ext$Date))
data.frame(dr.snp)



ggplot() + geom_bar(data = sent.df, aes(x=sent.df$Date, y=sent.df$Sentiment, col='Sentiment'), stat="identity") +
  geom_line(data = dr.snp, aes(x=sent.df$Date, y=dr.snp$daily.returns*10, col='Return')) + 
  scale_color_manual(name="", 
                   values = c("Sentiment"="#0ED0D0", "Return"="#E86666")) 


# WTI price data
#install.packages("rjson")
library("rjson")
json_file <- "wti-daily_json.json"
json_data <- fromJSON(paste(readLines(json_file), collapse=""))
wti <- do.call(rbind, lapply(json_data, data.frame))
wti <- wti[7143:8070,]

wti$Date <- as.Date(wti$Date, "%Y-%m-%d")
snp.ext$Date[!(snp.ext$Date %in% wti$Date)]
dr.wti <- dailyReturn(xts(wti$Price, wti$Date))


# Top Companies, Mega Cap
cname <- unique(sandp.data[sandp.data$Size == "Mega Cap", ]$Ticker)
# APPLE
apple <- sandp.data[sandp.data$Ticker == cname[1] & sandp.data$date <= '2017-12-29',]
apple <- apple[!(apple$date=="2017-07-03"),]
dr.apple <- dailyReturn(xts(apple$close, apple$date))
# Amazon
AMZN <- sandp.data[sandp.data$Ticker == cname[2] & sandp.data$date <= '2017-12-29',]
AMZN <- AMZN[!(AMZN$date=="2017-07-03"),]
dr.AMZN <- dailyReturn(xts(AMZN$close, AMZN$date))
# BRK -B
BRKB <- sandp.data[sandp.data$Ticker == cname[3] & sandp.data$date <= '2017-12-29',]
BRKB <- BRKB[!(BRKB$date=="2017-07-03"),]
dr.BRKB <- dailyReturn(xts(BRKB$close, BRKB$date))
# Facebook
FB <- sandp.data[sandp.data$Ticker == cname[4] & sandp.data$date <= '2017-12-29',]
FB <- FB[!(FB$date=="2017-07-03"),]
dr.FB <- dailyReturn(xts(FB$close, FB$date))
# Google
GOOG <- sandp.data[sandp.data$Ticker == cname[5] & sandp.data$date <= '2017-12-29',]
GOOG <- GOOG[!(GOOG$date=="2017-07-03"),]
dr.GOOG <- dailyReturn(xts(GOOG$close, GOOG$date))
# JNJ
JNJ <- sandp.data[sandp.data$Ticker == cname[7] & sandp.data$date <= '2017-12-29',]
JNJ <- JNJ[!(JNJ$date=="2017-07-03"),]
dr.JNJ <- dailyReturn(xts(JNJ$close, JNJ$date))
# JPM
JPM <- sandp.data[sandp.data$Ticker == cname[8] & sandp.data$date <= '2017-12-29',]
JPM <- JPM[!(JPM$date=="2017-07-03"),]
dr.JPM <- dailyReturn(xts(JPM$close, JPM$date))
# MSFT
MSFT <- sandp.data[sandp.data$Ticker == cname[9] & sandp.data$date <= '2017-12-29',]
MSFT <- MSFT[!(MSFT$date=="2017-07-03"),]
dr.MSFT <- dailyReturn(xts(MSFT$close, MSFT$date))
# V
VISA <- sandp.data[sandp.data$Ticker == cname[10] & sandp.data$date <= '2017-12-29',]
VISA <- VISA[!(VISA$date=="2017-07-03"),]
dr.VISA <- dailyReturn(xts(VISA$close, VISA$date))
# XOM
XOM <- sandp.data[sandp.data$Ticker == cname[11] & sandp.data$date <= '2017-12-29',]
XOM <- XOM[!(XOM$date=="2017-07-03"),]
dr.XOM <- dailyReturn(xts(XOM$close, XOM$date))

# Gold Price
gold.data <- read_csv("WGC-GOLD_DAILY_USD.csv")
gold.data$Date <- as.Date(gold.data$Date, "%d/%m/%Y")
gold.data<-gold.data[!(gold.data$Date %in% gold.data$Date[!(gold.data$Date %in% snp.ext$Date)]),]
gold.data <- gold.data[gold.data$Date <= '2017-12-29' & gold.data$Date >= '2014-04-25',]
dr.gold <- dailyReturn(xts(gold.data$Value, gold.data$Date))

length(dr.gold)
length(dr.XOM)



# Correlation
combined <- data.frame(dr.wti$daily.returns, dr.snp$daily.returns, 
                       sent.df$Sentiment, dr.apple$daily.returns,
                       dr.AMZN$daily.returns, dr.BRKB$daily.returns,
                       dr.FB$daily.returns, dr.GOOG$daily.returns,
                       dr.JNJ$daily.returns, dr.JPM$daily.returns,
                       dr.MSFT$daily.returns, dr.VISA$daily.returns,
                       dr.XOM$daily.returns, dr.gold$daily.returns)

colnames(combined) <- c("WTI", "S&P", "Sentiment", "Apple", "Amazon", "BRK-B", "Facebook", 
                        "Google", "Johnson&J", "JP Morgan", "Microsoft", "Visa", "Exon",
                        "Gold")

#install.packages("ggcorrplot")
library(ggcorrplot)
ggcorrplot(cor(combined), hc.order = TRUE, lab = TRUE,
           outline.col = "white",
           ggtheme = ggplot2::theme_gray,
           colors = c("#6D9EC1", "white", "#E46726"))




# S&P Daily return
snp.ext<-snp.ext[!(snp.ext$Date=="2017-07-03"),]
dr.snp <- monthlyReturn(xts(snp.ext$Close, snp.ext$Date))

# WTI price data
dr.wti <- monthlyReturn(xts(wti$Price, wti$Date))


# Top Companies, Mega Cap
cname <- unique(sandp.data[sandp.data$Size == "Mega Cap", ]$Ticker)
# APPLE
apple <- sandp.data[sandp.data$Ticker == cname[1] & sandp.data$date <= '2017-12-29',]
apple <- apple[!(apple$date=="2017-07-03"),]
dr.apple <- monthlyReturn(xts(apple$close, apple$date))
# Amazon
AMZN <- sandp.data[sandp.data$Ticker == cname[2] & sandp.data$date <= '2017-12-29',]
AMZN <- AMZN[!(AMZN$date=="2017-07-03"),]
dr.AMZN <- monthlyReturn(xts(AMZN$close, AMZN$date))
# BRK -B
BRKB <- sandp.data[sandp.data$Ticker == cname[3] & sandp.data$date <= '2017-12-29',]
BRKB <- BRKB[!(BRKB$date=="2017-07-03"),]
dr.BRKB <- monthlyReturn(xts(BRKB$close, BRKB$date))
# Facebook
FB <- sandp.data[sandp.data$Ticker == cname[4] & sandp.data$date <= '2017-12-29',]
FB <- FB[!(FB$date=="2017-07-03"),]
dr.FB <- monthlyReturn(xts(FB$close, FB$date))
# Google
GOOG <- sandp.data[sandp.data$Ticker == cname[5] & sandp.data$date <= '2017-12-29',]
GOOG <- GOOG[!(GOOG$date=="2017-07-03"),]
dr.GOOG <- monthlyReturn(xts(GOOG$close, GOOG$date))
# JNJ
JNJ <- sandp.data[sandp.data$Ticker == cname[7] & sandp.data$date <= '2017-12-29',]
JNJ <- JNJ[!(JNJ$date=="2017-07-03"),]
dr.JNJ <- monthlyReturn(xts(JNJ$close, JNJ$date))
# JPM
JPM <- sandp.data[sandp.data$Ticker == cname[8] & sandp.data$date <= '2017-12-29',]
JPM <- JPM[!(JPM$date=="2017-07-03"),]
dr.JPM <- monthlyReturn(xts(JPM$close, JPM$date))
# MSFT
MSFT <- sandp.data[sandp.data$Ticker == cname[9] & sandp.data$date <= '2017-12-29',]
MSFT <- MSFT[!(MSFT$date=="2017-07-03"),]
dr.MSFT <- monthlyReturn(xts(MSFT$close, MSFT$date))
# V
VISA <- sandp.data[sandp.data$Ticker == cname[10] & sandp.data$date <= '2017-12-29',]
VISA <- VISA[!(VISA$date=="2017-07-03"),]
dr.VISA <- monthlyReturn(xts(VISA$close, VISA$date))
# XOM
XOM <- sandp.data[sandp.data$Ticker == cname[11] & sandp.data$date <= '2017-12-29',]
XOM <- XOM[!(XOM$date=="2017-07-03"),]
dr.XOM <- monthlyReturn(xts(XOM$close, XOM$date))

# Gold Price
gold.data <- read_csv("WGC-GOLD_DAILY_USD.csv")
gold.data$Date <- as.Date(gold.data$Date, "%d/%m/%Y")
gold.data<-gold.data[!(gold.data$Date %in% gold.data$Date[!(gold.data$Date %in% snp.ext$Date)]),]
gold.data <- gold.data[gold.data$Date <= '2017-12-29' & gold.data$Date >= '2014-04-25',]
dr.gold <- monthlyReturn(xts(gold.data$Value, gold.data$Date))


# Correlation
combined.m <- data.frame(dr.wti$monthly.returns, dr.snp$monthly.returns, 
                       dr.apple$monthly.returns,
                       dr.AMZN$monthly.returns, dr.BRKB$monthly.returns,
                       dr.FB$monthly.returns, dr.GOOG$monthly.returns,
                       dr.JNJ$monthly.returns, dr.JPM$monthly.returns,
                       dr.MSFT$monthly.returns, dr.VISA$monthly.returns,
                       dr.XOM$monthly.returns, dr.gold$monthly.returns)

colnames(combined.m) <- c("WTI", "S&P", "Apple", "Amazon", "BRK-B", "Facebook", 
                        "Google", "Johnson&J", "JP Morgan", "Microsoft", "Visa", "Exon", "Gold")

ggcorrplot(cor(combined.m), hc.order = TRUE,
           outline.col = "white",
           ggtheme = ggplot2::theme_gray, lab = TRUE,
           colors = c("#6D9EC1", "white", "#E46726"))

