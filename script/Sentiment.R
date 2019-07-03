# Settings -----
if(require("pacman")=="FALSE"){
  install.packages("pacman")
}
pacman::p_load(ggplot2, rstudioapi, plyr, purrr, readr, plotly, caret,
               RColorBrewer, caretEnsemble, parallel, doMC, randomForest, 
               DescTools, ggpubr, ggthemes, corrplot, C50, e1071)

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
# Apple <- grep("iphone", names(galaxy$df), value=T) # "iphonesentiment" excluded
# Samsung <- grep("samsung", names(iphone$df), value=T) # "galaxysentiment" excluded

corrplot(cor(iphone$df[!duplicated(iphone$df), ]))
corrplot(cor(galaxy$df[!duplicated(galaxy$df), ]))

# Factoring dependant variables ----
galaxy$df$galaxysentiment <- as.factor(galaxy$df$galaxysentiment)
iphone$df$iphonesentiment <- as.factor(iphone$df$iphonesentiment)

# Removing duplicates: -----
galaxy$newdf <- galaxy$df[!duplicated(galaxy$df), ]
iphone$newdf <- iphone$df[!duplicated(iphone$df), ]

# Spliting the data ----
set.seed(123)
# iphone$sample <- iphone$df %>% dplyr::group_by(iphonesentiment) %>%
#   dplyr::sample_n(ifelse(dplyr::n() < 350, dplyr::n(), 350))
inTrain <- createDataPartition(y = iphone$newdf$iphonesentiment, 
                               p = 0.7, list = FALSE)
iphone$train <- iphone$newdf[inTrain,]
  # iphone$sample %>% dplyr::group_by(iphonesentiment) %>%
  # dplyr::sample_frac(.70)
iphone$test <- iphone$df[-inTrain,]
  # dplyr::anti_join(iphone$df, iphone$train)
galaxy$train <- galaxy$df %>% dplyr::group_by(iphonesentiment) %>% 
  dplyr::sample_n(ifelse(dplyr::n() > 350, 350, dplyr::n()))
galaxy$test <- galaxy$df[2001:nrow(galaxy$df), ]

 # Principal Component Analysis----
iphone$preprocessParams <- preProcess(iphone$train[,-59], 
                                      method=c("center", "scale", "pca", "nzv"), 
                                      thresh = 0.95)
print(iphone$preprocessParams)
iphone$train.pca <- predict(iphone$preprocessParams, iphone$train[,-59])
iphone$train.pca$iphonesentiment <- iphone$train$iphonesentiment
iphone$test.pca <- predict(iphone$preprocessParams, iphone$test[,-59])
iphone$test.pca$iphonesentiment <- iphone$test$iphonesentiment
iphone$df.pca <- predict(iphone$preprocessParams, iphone$df[,-59])
iphone$df.pca$iphonesentiment <- iphone$df$iphonesentiment
# # Recursive Feature Elimination ----
# iphoneRFE <- rfe(iphone$sample[,1:58], 
#                  iphone$sample$iphonesentiment, 
#                  sizes=(1:58), 
#                  rfeControl=rfeControl(functions = rfFuncs,
#                                        method = "repeatedcv",
#                                        repeats = 5,
#                                        verbose = FALSE))
# plot(iphoneRFE, type=c("g", "o"))

# Models ----
models <- c("C5.0", "rf", "knn")
for (i in models) {
  iphone[[i]] <- train(iphonesentiment~., # y = iphone$train$iphonesentiment,
                       # x = iphone$train[,predictors(iphoneRFE)],
                       data = iphone$train.pca, 
                       method = i,
                       trControl = trainControl(method = "cv",
                                                number = 5,
                                                verboseIter = TRUE,
                                                sampling = "up"),
                       metric = "Kappa")
  
  iphone[[paste0("pred_",i)]] <- predict(iphone[[i]], newdata = iphone$test.pca)
  iphone[[paste0("conf_mtx_",i)]] <-table(iphone[[paste0("pred_",i)]], 
                                          iphone$test.pca$iphonesentiment)
  iphone[[paste0("accuracy_",i)]] <- postResample(iphone[[paste0("pred_",i)]],
                                                  iphone$test.pca$iphonesentiment)
  
}
rm(i, models)

iphone$svm <- svm(iphonesentiment ~.,
                  data = iphone$train.pca)
iphone$pred_svm <- predict(iphone$svm, newdata = iphone$test.pca)
iphone$accuracy_svm <- postResample(iphone$pred_svm, 
                                    iphone$test.pca$iphonesentiment)
  # RF
postResample(predict(iphone$rf, newdata = iphone$df.pca),
             iphone$df.pca$iphonesentiment)
  # k-NN
postResample(predict(iphone$knn, newdata = iphone$df.pca),
             iphone$df.pca$iphonesentiment)
  # C5.0
postResample(predict(iphone$C5.0, newdata = iphone$df.pca),
             iphone$df.pca$iphonesentiment)
  # SVM
postResample(predict(iphone$svm, newdata = iphone$df.pca),
             iphone$df.pca$iphonesentiment)
