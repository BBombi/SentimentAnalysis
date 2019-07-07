# Settings -----
if(require("pacman")=="FALSE"){
  install.packages("pacman")
}
pacman::p_load(ggplot2, rstudioapi, plyr, purrr, readr, plotly, caret,
               RColorBrewer, caretEnsemble, parallel, doMC, randomForest, 
               DescTools, ggpubr, ggthemes, corrplot, C50, e1071, car, 
               prettydoc, RColorBrewer)

current_path <- getActiveDocumentContext()$path
setwd(dirname(dirname(current_path)))
rm(current_path)
registerDoMC(cores = detectCores()-1)

# Loadding the data ----
galaxy <- list(c())
iphone <- list(c())
galaxy$df <- read.csv("datasets/galaxy_smallmatrix_labeled_9d.csv")
iphone$df <- read.csv("datasets/iphone_smallmatrix_labeled_8d.csv")

  # Checking Correlations ----
Apple <- c(grep("iphone", names(iphone$df[1:58]), value=T), "ios")
Samsung <- c(grep("samsung", names(galaxy$df[1:58]), value=T), "googleandroid")
corrplot(cor(iphone$df[!duplicated(iphone$df), Apple]))
corrplot(cor(galaxy$df[!duplicated(galaxy$df), Samsung]))

# Plotting distributions ----
ggplot(iphone$df, aes(x=iphonesentiment)) + 
  geom_histogram(aes(y=..density..), colour="black", 
                 fill=brewer.pal(name = "RdBu", n=6), bins = 6) +
  labs(title = "iPhone Sentiment") + xlab(NULL)

ggplot(galaxy$df, aes(x=galaxysentiment)) + 
  geom_histogram(aes(y=..density..), colour="black", 
                 fill=brewer.pal(name = "YlGn", n=6), bins = 6) +
  labs(title = "Galaxy Sentiment") + xlab(NULL)

 # Removing duplicates ----
iphone$newdf <- iphone$df[!duplicated(iphone$df),]
galaxy$newdf <- galaxy$df[!duplicated(galaxy$df),]

# Factoring dependant variables ----
galaxy$newdf$galaxysentiment <- as.factor(galaxy$newdf$galaxysentiment)
iphone$newdf$iphonesentiment <- as.factor(iphone$newdf$iphonesentiment)

# Spliting the data ----
set.seed(123)
inTrain <- createDataPartition(y = iphone$newdf$iphonesentiment, 
                               p = 0.7, list = FALSE)
iphone$train <- iphone$newdf[inTrain,]
iphone$test <- iphone$newdf[-inTrain,]

set.seed(123)
inTrain <- createDataPartition(y = galaxy$newdf$galaxysentiment, 
                               p = 0.7, list = FALSE)
galaxy$train <- galaxy$newdf[inTrain,]
galaxy$test <- galaxy$newdf[-inTrain,]

 # Principal Component Analysis----
iphone$preprocessParams <- preProcess(iphone$train[,-ncol(iphone$train)], 
                                      method="pca", 
                                      thresh = 0.90)
print(iphone$preprocessParams)
iphone$train.pca <- predict(iphone$preprocessParams, iphone$train[,-59])
iphone$train.pca$iphonesentiment <- iphone$train$iphonesentiment
iphone$test.pca <- predict(iphone$preprocessParams, iphone$test[,-59])
iphone$test.pca$iphonesentiment <- iphone$test$iphonesentiment
iphone$df.pca <- predict(iphone$preprocessParams, iphone$df[,-59])
iphone$df.pca$iphonesentiment <- iphone$df$iphonesentiment

galaxy$preprocessParams <- preProcess(galaxy$train[,-ncol(galaxy$train)], 
                                      method="pca", 
                                      thresh = 0.9)
print(galaxy$preprocessParams)
galaxy$train.pca <- predict(galaxy$preprocessParams, galaxy$train[,-59])
galaxy$train.pca$galaxysentiment <- galaxy$train$galaxysentiment
galaxy$test.pca <- predict(galaxy$preprocessParams, galaxy$test[,-59])
galaxy$test.pca$galaxysentiment <- galaxy$test$galaxysentiment
galaxy$df.pca <- predict(galaxy$preprocessParams, galaxy$df[,-59])
galaxy$df.pca$galaxysentiment <- galaxy$df$galaxysentiment

# Models for Iphone ----
iphone$C5.0 <- train(iphonesentiment ~ .,
                     data = iphone$train.pca,
                     method = "C5.0",
                     trControl = trainControl(method = "repeatedcv",
                                              number = 10,
                                              repeats = 10, 
                                              returnResamp="all",
                                              verboseIter = TRUE),
                     metric = "Kappa",
                     preProcess = c("center", "scale"))

saveRDS(iphone$C5.0, "Models/iPhoneC5.0.rds")

iphone$svm <- svm(iphonesentiment ~.,
                  data = iphone$train.pca)
  
  # C5.0
postResample(predict(iphone$C5.0, newdata = iphone$df.pca),
             iphone$df.pca$iphonesentiment)
  # SVM
postResample(predict(iphone$svm, newdata = iphone$df.pca),
             iphone$df.pca$iphonesentiment)
 
#Models for Galaxy ----
galaxy$C5.0 <- train(galaxysentiment ~ .,
                     data = galaxy$train.pca,
                     method = "C5.0",
                     trControl = trainControl(method = "repeatedcv",
                                              number = 5,
                                              repeats = 5, 
                                              returnResamp="all",
                                              verboseIter = TRUE),
                     metric = "Kappa",
                     preProcess = c("center", "scale"))

saveRDS(galaxy$C5.0, "Models/GalaxyC5.0.rds")

galaxy$svm <- svm(galaxysentiment ~.,
                  data = galaxy$train.pca)

# C5.0
postResample(predict(galaxy$C5.0, newdata = galaxy$df.pca),
             galaxy$df.pca$galaxysentiment)
# SVM
postResample(predict(galaxy$svm, newdata = galaxy$df.pca),
             galaxy$df.pca$galaxysentiment)

# Loading the real Testset and applying the models ----
RealTest <- read.csv("datasets/allfactors.csv")

RealTest <- RealTest[2:59]

iphone$realtest.pca <- predict(iphone$preprocessParams, RealTest)
galaxy$realtest.pca <- predict(galaxy$preprocessParams, RealTest)
RealTest$iphonesentiment <- predict(iphone$C5.0, newdata = iphone$realtest.pca)
RealTest$galaxysentiment <- predict(galaxy$svm, newdata = galaxy$realtest.pca)

summary(RealTest[which(RealTest$galaxysentiment == RealTest$iphonesentiment),60])

sum(as.numeric(RealTest$iphonesentiment))/nrow(RealTest)-1
sum(as.numeric(RealTest$galaxysentiment))/nrow(RealTest)-1

ggplot(RealTest, aes(x=iphonesentiment)) + 
  geom_bar(colour="black", fill=brewer.pal(name = "RdBu", n=5)) +
  labs(title = "iPhone Sentiment") + xlab(NULL)

ggplot(RealTest, aes(x=galaxysentiment)) + 
  geom_bar(colour="black", fill=brewer.pal(name = "YlGn", n=4)) +
  labs(title = "Galaxy Sentiment") + xlab(NULL)