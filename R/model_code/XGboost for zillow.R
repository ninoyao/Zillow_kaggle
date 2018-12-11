## XGBoost model for Zillow Home Value Prediction competition

#load packages and data
install.packages("purrr")
packages <- c("readr", "dplyr", "purrr", "xgboost", "lubridate")
purrr::walk(packages, library, character.only = TRUE, warn.conflicts = FALSE)

##
## Set important constants
##

train_filename <- "/Users/nino/Desktop/Kaggle/Zillow/train_2016_v2.csv"
properties_filename <- "/Users/nino/Desktop/Kaggle/Zillow/properties_2016.csv"

sample_submission_filename <- "/Users/nino/Desktop/Kaggle/Zillow/sample_submission.csv"
results_path <- "./"

target_variable <- "logerror"
target_months <- c("201610", "201611", "201612", "201710", "201711", "201712")
train_variables <- c(target_variable)

SEED = 100

get_feature_names <- function(train, test, train_variables = train_variables){
  ## Get list of variables used in training
  var_names <- intersect(names(train), names(test))
  var_names <- setdiff(var_names, train_variables)
  return(var_names)
}



create_features <- function(data, properties, train = TRUE, te_data = NULL){
  
  if(!train){
    cat("Creating features for each property.\n")
    
    ## First convert character features to numeric values
    
    ctypes <- sapply(data, class)
    cidx <- which(ctypes %in% c("character", "factor"))
    
    for(i in cidx){
      data[[i]] <- as.integer(factor(data[[i]]))
    }
    
    
    ## This is where you add all the fancy features you want to add
    output <- data
    
    output$bathroomspersf <- output$bathroomcnt / output$calculatedfinishedsquarefeet
    
    return(output)
  } else {
    cat("Extracting features from", round(nrow(data)/1000,0), "thousand examples.\n")
    
    dat_year <- year(data$transactiondate)
    dat_month <- month(data$transactiondate)
    data$transactiondate <- NULL
    data$yearmonth <- paste0(dat_year, sprintf("%02d", dat_month))
    
    data2 <- data %>%
      left_join(te_data, by = c("parcelid"))
    
    return(data2)
  }
}



run_xgboost_model <- function(tr, target_variable, var_names, n_rounds, early_stop, eta, m_depth, seed){
  
  ## Create folds
  
  set.seed(seed)
  idx_tr <- sample(seq(nrow(tr)), floor(.8 * nrow(tr)) )
  gc()
  
  tr_len <- length(idx_tr)
  vl_len <- nrow(tr) - tr_len
  
  cat("Training with", tr_len , "sales, validating with", vl_len ,"sales\n")
  
  target <- tr[[target_variable]]
  
  dtrain <- xgb.DMatrix(data.matrix(tr[idx_tr,var_names]), label=target[idx_tr], missing=NA)
  dval <- xgb.DMatrix(data.matrix(tr[-idx_tr,var_names]), label=target[-idx_tr], missing=NA)
  
  
  watchlist <- watchlist <- list(train = dtrain, eval = dval)
  
  
  param <- list(  objective           = "reg:linear", 
                  booster             = "gbtree",
                  eval_metric         = "mae",
                  eta                 = eta, 
                  max_depth           = m_depth, 
                  subsample           = 0.5,
                  colsample_bytree    = 0.5,
                  min_child_weight    = 4,
                  maximize            = FALSE
                  
  )
  
  xgb <- xgb.train( params                = param, 
                    data                  = dtrain,
                    nrounds               = n_rounds, 
                    verbose               = 1,
                    print_every_n         = 10L,
                    early_stopping_rounds = early_stop,
                    watchlist             = watchlist
  )
  return(xgb)
}


write_submission_file <- function(model, te, var_names, target_months, results_path){
  sub_name <- paste(results_path, "submission_",
                    model$best_score,"_", Sys.time(),".csv", sep = "")
  sub_name <- gsub(":", "-", sub_name)
  sub_name <- gsub(" ", "_", sub_name)
  
  sub_data <- predict_monthly_errors(model, te, var_names, target_months)
  
  write_csv(sub_data, sub_name)
  return(sub_name)
}


read_and_prepare_data <- function(filename, train = TRUE, te_data = NULL){
  if(train){
    cat("Reading transaction data.\n")
    colspec <- cols(
      parcelid = col_integer(),
      logerror = col_double(),
      transactiondate = col_date(format = "")
    )
  } else {
    cat("Reading property data.\n")
    colspec <- cols(
      .default = col_integer(),
      architecturalstyletypeid = col_character(),
      bathroomcnt = col_double(),
      bedroomcnt = col_double(),
      calculatedbathnbr = col_double(),
      calculatedfinishedsquarefeet = col_double(),
      fips = col_character(),
      hashottuborspa = col_character(),
      lotsizesquarefeet = col_double(),
      pooltypeid10 = col_character(),
      pooltypeid2 = col_character(),
      propertycountylandusecode = col_character(),
      propertyzoningdesc = col_character(),
      rawcensustractandblock = col_character(),
      roomcnt = col_double(),
      typeconstructiontypeid = col_character(),
      yearbuilt = col_double(),
      fireplaceflag = col_character(),
      structuretaxvaluedollarcnt = col_double(),
      taxvaluedollarcnt = col_double(),
      landtaxvaluedollarcnt = col_double(),
      taxamount = col_double(),
      taxdelinquencyflag = col_character(),
      censustractandblock = col_double()
    )
  }
  
  
  dat <- read_csv(filename, col_types = colspec)
  output <- create_features(dat, train = train, te_data = te_data)
  return(output)
}


predict_monthly_errors <- function(model, te, var_names, target_months){
  
  # Prepopulate results object
  sub_mat <- matrix(0, nrow = nrow(te), ncol = 1 + length(target_months))
  sub_dat <- as.data.frame(sub_mat)
  names(sub_dat) <- c("ParcelId", target_months)
  sub_dat$ParcelId <- te$parcelid
  
  for(tmonth in target_months){
    te$yearmonth <- tmonth
    dte <- xgb.DMatrix(data.matrix(te[var_names]), missing=NA)
    sub_dat[tmonth] <- round(predict(model, dte), 4)
  }
  
  return(sub_dat)
}

##Run the model and make the predictions file
te <- read_and_prepare_data(properties_filename, train = FALSE)
tr <- read_and_prepare_data(train_filename, train = TRUE, te)
var_names <- get_feature_names(tr, tr, train_variables)
model <- run_xgboost_model(tr, target_variable, var_names,
                           n_rounds = 1000, 
                           early_stop = 50, 
                           eta = 0.1, 
                           m_depth = 4,
                           seed = SEED)

write_submission_file(model, te, var_names, target_months, results_path)

summary(tr$airconditioningtypeid)
class(tr$airconditioningtypeid)