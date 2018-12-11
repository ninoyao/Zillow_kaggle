#In this simple model I want to use linear regression on few selected features

#Read the train file
tr <- read.csv("train.csv")
te <- read.csv("test.csv")

#Bath <- pmax(1,tr$FullBath)
#Bed <- pmin(pmax(1, tr$BedroomAbvGr), 4)
LogLotArea <- log10(tr$LotArea)
LogArea <- log10(tr$GrLivArea)
Age <- pmax(0.0, tr$YrSold - pmax(tr$YearBuilt, tr$YearRemodAdd))
#New <- as.factor(Age == 0.0)
Quality <- tr$OverallQual
Neighborhood <- as.factor(tr$Neighborhood)
Zoning <- as.factor(tr$MSZoning)
Style <- as.factor(tr$HouseStyle)
Condition <- tr$OverallCond

LogPrice <- log10(tr$SalePrice)

simplemodel <- lm(LogPrice ~ LogArea + LogLotArea + Age + Quality + Condition + Neighborhood + Zoning + Style)


#Bath <- pmax(1,te$FullBath)
#Bed <- pmin(pmax(1, te$BedroomAbvGr), 4)
LogArea <- log10(te$GrLivArea)
LogLotArea <- log10(te$LotArea)
Age <- pmax(0.0, te$YrSold - pmax(te$YearBuilt, te$YearRemodAdd))
#New <- as.factor(Age == 0.0)
Quality <- te$OverallQual
Neighborhood <- as.factor(te$Neighborhood)
Zoning <- as.factor(te$MSZoning)
Zoning[is.na(Zoning)] <- "RL"
Subclass <- as.factor(te$MSSubClass)
Style <- as.factor(te$HouseStyle)
Condition <- te$OverallCond

fewfeatures <- data.frame(LogArea, LogLotArea, Age, Quality, Condition, Neighborhood, Zoning, Style)
sapply(1:ncol(fewfeatures), function(i) anyNA(fewfeatures[,i]))

test.LogPrice <- predict(simplemodel, fewfeatures)
SalePrice <- 10.0**test.LogPrice
Id <- te$Id


submission <- data.frame(Id, SalePrice)

write.csv(submission, "lmsubmission.csv", row.names = F)
