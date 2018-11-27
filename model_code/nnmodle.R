gbm_yhat <- do.call(c,lapply(gbm_set,function(x){x$predictions$yhat}))
xgb_yhat <- do.call(c,lapply(xgb_set,function(x){x$predictions$yhat}))
rngr_yhat <- do.call(c,lapply(rngr_set,function(x){x$predictions$yhat}))

# create Feature Set
L1FeatureSet <- list()

L1FeatureSet$train$id <- do.call(c,lapply(gbm_set,function(x){x$predictions$ID}))
L1FeatureSet$train$y <- do.call(c,lapply(gbm_set,function(x){x$predictions$y}))
predictors <- data.frame(gbm_yhat,xgb_yhat,rngr_yhat)
predictors_rank <- t(apply(predictors,1,rank))
colnames(predictors_rank) <- paste0("rank_",names(predictors))
L1FeatureSet$train$predictors <- predictors #cbind(predictors,predictors_rank)

L1FeatureSet$test$id <- gbm_submission[,"Id"]
L1FeatureSet$test$predictors <- data.frame(gbm_yhat=test_gbm_yhat,
                                           xgb_yhat=test_xgb_yhat,
                                           rngr_yhat=test_rngr_yhat)
```


### Neural Net Model


# set caret training parameters
CARET.TRAIN.PARMS <- list(method="nnet") 

CARET.TUNE.GRID <-  NULL  # NULL provides model specific default tuning parameters

# model specific training parameter
CARET.TRAIN.CTRL <- trainControl(method="repeatedcv",
                                 number=5,
                                 repeats=1,
                                 verboseIter=FALSE)

CARET.TRAIN.OTHER.PARMS <- list(trControl=CARET.TRAIN.CTRL,
                                maximize=FALSE,
                                tuneGrid=CARET.TUNE.GRID,
                                tuneLength=7,
                                metric="RMSE")

MODEL.SPECIFIC.PARMS <- list(verbose=FALSE,linout=TRUE,trace=FALSE) #NULL # Other model specific parameters


# train the model
set.seed(825)
l1_nnet_mdl <- do.call(train,c(list(x=L1FeatureSet$train$predictors,y=L1FeatureSet$train$y),
                               CARET.TRAIN.PARMS,
                               MODEL.SPECIFIC.PARMS,
                               CARET.TRAIN.OTHER.PARMS))

l1_nnet_mdl
cat("Average CV rmse:",mean(l1_nnet_mdl$resample$RMSE),"\n")

test_l1_nnet_yhat <- predict(l1_nnet_mdl,newdata = L1FeatureSet$test$predictors,type = "raw")
l1_nnet_submission <- cbind(Id=L1FeatureSet$test$id,SalePrice=exp(test_l1_nnet_yhat))
colnames(l1_nnet_submission) <- c("Id","SalePrice")

write.csv(l1_nnet_submission,file="l1_nnet_submission.csv",row.names=FALSE)
