# Settings -----
if(require("pacman")=="FALSE"){
  install.packages("pacman")
}
pacman::p_load(ggplot2, rstudioapi, plyr, purrr, readr, plotly, caret,
               RColorBrewer, caretEnsemble, parallel, doMC, randomForest, 
               DescTools, ggpubr, ggthemes, corrplot, C50, e1071, car)

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
corrplot(cor(iphone$df[!duplicated(iphone$df), ]))
corrplot(cor(galaxy$df[!duplicated(galaxy$df), ]))

 # Removing multicolinearity and duplicates ----
iphone$newdf <- iphone$df[!duplicated(iphone$df), ]
# iphone$newdf <-iphone$newdf[ ,c((which(vif(lm(iphonesentiment ~ ., 
#                               data = iphone$df[!duplicated(iphone$df), ])) 
#                        <= 5)),59)]
galaxy$newdf <- galaxy$df[!duplicated(galaxy$df), ]
# galaxy$newdf <-   galaxy$newdf[ ,c((which(vif(lm(galaxysentiment ~ ., 
#                              data = galaxy$df[!duplicated(galaxy$df), ])) 
#                        <= 5)),59)]
# Factoring dependant variables ----
galaxy$df$galaxysentiment <- as.factor(galaxy$df$galaxysentiment)
iphone$df$iphonesentiment <- as.factor(iphone$df$iphonesentiment)
galaxy$newdf$galaxysentiment <- as.factor(galaxy$newdf$galaxysentiment)
iphone$newdf$iphonesentiment <- as.factor(iphone$newdf$iphonesentiment)

# Spliting the data ----
set.seed(123)
inTrain <- createDataPartition(y = iphone$newdf$iphonesentiment, 
                               p = 0.75, list = FALSE)
iphone$train <- iphone$newdf[inTrain,]
iphone$test <- iphone$newdf[-inTrain,]

inTrain <- createDataPartition(y = galaxy$newdf$galaxysentiment, 
                               p = 0.75, list = FALSE)
galaxy$train <- galaxy$newdf[inTrain,]
galaxy$test <- galaxy$newdf[-inTrain,]

 # Principal Component Analysis----
iphone$preprocessParams <- preProcess(iphone$train[,-ncol(iphone$train)], 
                                      method="pca", 
                                      thresh = 0.88)
print(iphone$preprocessParams)
iphone$train.pca <- predict(iphone$preprocessParams, iphone$train[,-59])
iphone$train.pca$iphonesentiment <- iphone$train$iphonesentiment
iphone$test.pca <- predict(iphone$preprocessParams, iphone$test[,-59])
iphone$test.pca$iphonesentiment <- iphone$test$iphonesentiment
iphone$df.pca <- predict(iphone$preprocessParams, iphone$df[,-59])
iphone$df.pca$iphonesentiment <- iphone$df$iphonesentiment

galaxy$preprocessParams <- preProcess(galaxy$train[,-ncol(galaxy$train)], 
                                      method="pca", 
                                      thresh = 0.88)
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

iphone$svm <- svm(iphonesentiment ~.,
                  data = iphone$train.pca)
iphone$pred_svm <- predict(iphone$svm, newdata = iphone$test.pca)
iphone$accuracy_svm <- postResample(iphone$pred_svm, 
                                    iphone$test.pca$iphonesentiment)
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
                                              number = 10,
                                              repeats = 10, 
                                              returnResamp="all",
                                              verboseIter = TRUE),
                     metric = "Kappa",
                     preProcess = c("center", "scale"))

galaxy$svm <- svm(galaxysentiment ~.,
                  data = galaxy$train.pca)


# C5.0
postResample(predict(galaxy$C5.0, newdata = galaxy$df.pca),
             galaxy$df.pca$galaxysentiment)
# SVM
postResample(predict(galaxy$svm, newdata = galaxy$df.pca),
             galaxy$df.pca$galaxysentiment)

# Loading the real Testset and applying the models ----
RealTest <- rbind(read.csv("datasets/Alejo_combinedFile.csv"),
                  read.csv("datasets/concatenated_factors_borja.csv"),
                  read.csv("datasets/concatenated_factors1.csv"),
                  read.csv("datasets/concatenated_factors_0_100.csv"),
                  read.csv("datasets/concatenated_factors2.csv"),
                  read.csv("datasets/LargeMatrix.csv"))