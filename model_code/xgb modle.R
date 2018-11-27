# set caret training parameters
CARET.TRAIN.PARMS <- list(method="xgbTree")   

CARET.TUNE.GRID <- expand.grid(nrounds=600, max_depth=40, eta=0.03, gamma=0.01, colsample_bytree=0.4, min_child_weight=1, subsample = 1)

MODEL.SPECIFIC.PARMS <- list(verbose=0) #NULL # Other model specific parameters

# model specific training parameter
CARET.TRAIN.CTRL <- trainControl(method="none",
                                 verboseIter=FALSE,
                                 classProbs=FALSE)

CARET.TRAIN.OTHER.PARMS <- list(trControl=CARET.TRAIN.CTRL,
                                tuneGrid=CARET.TUNE.GRID,
                                metric="RMSE")

# generate Level 1 features
xgb_set <- llply(data_folds,trainOneFold,L0FeatureSet2)

# final model fit
xgb_mdl <- do.call(train,
                   c(list(x=L0FeatureSet2$train$predictors,y=L0FeatureSet2$train$y),
                     CARET.TRAIN.PARMS,
                     MODEL.SPECIFIC.PARMS,
                     CARET.TRAIN.OTHER.PARMS))

# CV Error Estimate
cv_y <- do.call(c,lapply(xgb_set,function(x){x$predictions$y}))
cv_yhat <- do.call(c,lapply(xgb_set,function(x){x$predictions$yhat}))
rmse(cv_y,cv_yhat)
## [1] 0.1292102
cat("Average CV rmse:",mean(do.call(c,lapply(xgb_set,function(x){x$score}))))
## Average CV rmse: 0.1284355
# create test submission.
# A prediction is made by averaging the predictions made by using the models
# fitted for each fold.

test_xgb_yhat <- predict(xgb_mdl,newdata = L0FeatureSet2$test$predictors,type = "raw")
xgb_submission <- cbind(Id=L0FeatureSet2$test$id,SalePrice=exp(test_xgb_yhat))

write.csv(xgb_submission,file="xgb_sumbission.csv",row.names=FALSE)

