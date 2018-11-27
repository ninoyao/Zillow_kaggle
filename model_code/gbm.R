## GBM
# set caret training parameters
CARET.TRAIN.PARMS <- list(method="gbm")   

CARET.TUNE.GRID <-  expand.grid(n.trees=100, 
                                interaction.depth=10, 
                                shrinkage=0.1,
                                n.minobsinnode=10)

MODEL.SPECIFIC.PARMS <- list(verbose=0) #NULL # Other model specific parameters

# model specific training parameter
CARET.TRAIN.CTRL <- trainControl(method="none",
                                 verboseIter=FALSE,
                                 classProbs=FALSE)

CARET.TRAIN.OTHER.PARMS <- list(trControl=CARET.TRAIN.CTRL,
                                tuneGrid=CARET.TUNE.GRID,
                                metric="RMSE")

# generate features for Level 1
gbm_set <- llply(data_folds,trainOneFold,L0FeatureSet1)

# final model fit
gbm_mdl <- do.call(train,
                   c(list(x=L0FeatureSet1$train$predictors,y=L0FeatureSet1$train$y),
                     CARET.TRAIN.PARMS,
                     MODEL.SPECIFIC.PARMS,
                     CARET.TRAIN.OTHER.PARMS))

# CV Error Estimate
cv_y <- do.call(c,lapply(gbm_set,function(x){x$predictions$y}))
cv_yhat <- do.call(c,lapply(gbm_set,function(x){x$predictions$yhat}))
rmse(cv_y,cv_yhat)
## [1] 0.1309466
cat("Average CV rmse:",mean(do.call(c,lapply(gbm_set,function(x){x$score}))))
## Average CV rmse: 0.1304117
# create test submission.
# A prediction is made by averaging the predictions made by using the models
# fitted for each fold.

test_gbm_yhat <- predict(gbm_mdl,newdata = L0FeatureSet1$test$predictors,type = "raw")
gbm_submission <- cbind(Id=L0FeatureSet1$test$id,SalePrice=exp(test_gbm_yhat))
write.csv(gbm_submission,file="gbm_sumbission2.csv",row.names=FALSE)
