# set caret training parameters
CARET.TRAIN.PARMS <- list(method="ranger")   

CARET.TUNE.GRID <-  expand.grid(splitrule = "extratrees",mtry=2*as.integer(sqrt(ncol(L0FeatureSet1$train$predictors))))

MODEL.SPECIFIC.PARMS <- list(verbose=0,num.trees=500) #NULL # Other model specific parameters

# model specific training parameter
CARET.TRAIN.CTRL <- trainControl(method="none",
                                 verboseIter=FALSE,
                                 classProbs=FALSE)

CARET.TRAIN.OTHER.PARMS <- list(trControl=CARET.TRAIN.CTRL,
                                tuneGrid=CARET.TUNE.GRID,
                                metric="RMSE")

# generate Level 1 features
rngr_set <- llply(data_folds,trainOneFold,L0FeatureSet1)
rngr_set = gbm_set
# final model fit
rngr_mdl <- do.call(train,
                    c(list(x=L0FeatureSet1$train$predictors,y=L0FeatureSet1$train$y),
                      CARET.TRAIN.PARMS,
                      MODEL.SPECIFIC.PARMS,
                      CARET.TRAIN.OTHER.PARMS))

# CV Error Estimate
cv_y <- do.call(c,lapply(rngr_set,function(x){x$predictions$y}))
cv_yhat <- do.call(c,lapply(rngr_set,function(x){x$predictions$yhat}))
rmse(cv_y,cv_yhat)
cat("Average CV rmse:",mean(do.call(c,lapply(rngr_set,function(x){x$score}))))

# create test submission.
# A prediction is made by averaging the predictions made by using the models
# fitted for each fold.

test_rngr_yhat <- predict(rngr_mdl,newdata = L0FeatureSet1$test$predictors,type = "raw")
rngr_submission <- cbind(Id=L0FeatureSet1$test$id,SalePrice=exp(test_rngr_yhat))

write.csv(rngr_submission,file="rngr_sumbission.csv",row.names=FALSE)


