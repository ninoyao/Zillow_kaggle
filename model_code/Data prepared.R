library(knitr)
library(caret)
library(plyr)
library(dplyr)
library(xgboost)
library(ranger)
library(nnet)
library(Metrics)
library(ggplot2)
knitr::opts_chunk$set(echo = TRUE, message=FALSE, warning=FALSE)
train.raw <- read.csv(file.path("train.csv"),stringsAsFactors = FALSE)
test.raw <- read.csv(file.path("test.csv"), stringsAsFactors = FALSE)

# incorporate results of Boruta analysis
CONFIRMED_ATTR <- c("MSSubClass","MSZoning","LotArea","LotShape","LandContour","Neighborhood",
                    "BldgType","HouseStyle","OverallQual","OverallCond","YearBuilt",
                    "YearRemodAdd","Exterior1st","Exterior2nd","MasVnrArea","ExterQual",
                    "Foundation","BsmtQual","BsmtCond","BsmtFinType1","BsmtFinSF1",
                    "BsmtFinType2","BsmtUnfSF","TotalBsmtSF","HeatingQC","CentralAir",
                    "X1stFlrSF","X2ndFlrSF","GrLivArea","BsmtFullBath","FullBath","HalfBath",
                    "BedroomAbvGr","KitchenAbvGr","KitchenQual","TotRmsAbvGrd","Functional",
                    "Fireplaces","FireplaceQu","GarageType","GarageYrBlt","GarageFinish",
                    "GarageCars","GarageArea","GarageQual","GarageCond","PavedDrive","WoodDeckSF",
                    "OpenPorchSF","Fence")

TENTATIVE_ATTR <- c("Alley","LandSlope","Condition1","RoofStyle","MasVnrType","BsmtExposure",
                    "Electrical","EnclosedPorch","SaleCondition")

REJECTED_ATTR <- c("LotFrontage","Street","Utilities","LotConfig","Condition2","RoofMatl",
                   "ExterCond","BsmtFinSF2","Heating","LowQualFinSF","BsmtHalfBath",
                   "X3SsnPorch","ScreenPorch","PoolArea","PoolQC","MiscFeature","MiscVal",
                   "MoSold","YrSold","SaleType")


PREDICTOR_ATTR <- c(CONFIRMED_ATTR,TENTATIVE_ATTR,REJECTED_ATTR)

# Determine data types in the data set
data_types <- sapply(PREDICTOR_ATTR,function(x){class(train.raw[[x]])})
unique_data_types <- unique(data_types)

# Separate attributes by data type
DATA_ATTR_TYPES <- lapply(unique_data_types,function(x){ names(data_types[data_types == x])})
names(DATA_ATTR_TYPES) <- unique_data_types


# create folds for training
set.seed(13)
data_folds <- createFolds(train.raw$SalePrice, k=5)

