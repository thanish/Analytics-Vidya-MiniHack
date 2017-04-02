train = read.csv('train_63qYitG.csv')
test_prod = read.csv('test_XaoFywY.csv')
str(train)
summary(train)


#Outliers removal from train_prod
train = train[train$Var1<110,]
train = train[!is.na(train$Var1),]
train = train[train$Var2<75,]
train = train[train$Var3<110,]
train = train[train$Life_Style_Index < 4,]
train = train[!is.na(train$Life_Style_Index),]

#combining the train an test
test_prod$Surge_Pricing_Type = NA
train_test_prod = rbind(train, test_prod)

#Filling up the NA's in factors
train_test_prod$Type_of_Cab = as.character(train_test_prod$Type_of_Cab)
train_test_prod$Type_of_Cab[train_test_prod$Type_of_Cab == ''] = 'other'
train_test_prod$Type_of_Cab = as.factor(train_test_prod$Type_of_Cab)

train_test_prod$Confidence_Life_Style_Index = as.character(train_test_prod$Confidence_Life_Style_Index)
train_test_prod$Confidence_Life_Style_Index[train_test_prod$Confidence_Life_Style_Index == ''] = 'other'
train_test_prod$Confidence_Life_Style_Index = as.factor(train_test_prod$Confidence_Life_Style_Index)

train_test_prod$Customer_Since_Months[is.na(train_test_prod$Customer_Since_Months)] = mean(train_test_prod$Customer_Since_Months, na.rm = T)
train_test_prod$Life_Style_Index[is.na(train_test_prod$Life_Style_Index)] = mean(train_test_prod$Life_Style_Index, na.rm = T)
train_test_prod$Var1[is.na(train_test_prod$Var1)] = mean(train_test_prod$Var1, na.rm = T)

#Splitting back to train an test prod
train = train_test_prod[!is.na(train_test_prod$Surge_Pricing_Type), ]
test_prod = train_test_prod[is.na(train_test_prod$Surge_Pricing_Type),]
train$Trip_ID = NULL

#Convert the predictor to factor
train$Surge_Pricing_Type = as.factor(as.character(train$Surge_Pricing_Type))

#train and test split
split = sample(nrow(train)*0.6)
train_local = train[split,]
test_local = train[-split,]

evalacc <- function(preds, real) 
{ 
  labels <- getinfo(real, "label")
  acc_table = table(preds, labels)
  Acc = sum(diag(acc_table))/nrow(test_local)
  return(list(metric = "Acc", value = Acc))
}

#XGBoosting model
library(xgboost)
library(dummies)
train_xgb_local = train
test_xgb_local = test_local
test_xgb_prod = test_prod

train_xgb_local$Type_of_Cab = as.numeric(train_xgb_local$Type_of_Cab)
train_xgb_local$Confidence_Life_Style_Index = as.numeric(train_xgb_local$Confidence_Life_Style_Index)
train_xgb_local$Destination_Type = as.numeric(train_xgb_local$Destination_Type)
train_xgb_local$Gender = as.numeric(train_xgb_local$Gender)

test_xgb_local$Type_of_Cab = as.numeric(test_xgb_local$Type_of_Cab)
test_xgb_local$Confidence_Life_Style_Index = as.numeric(test_xgb_local$Confidence_Life_Style_Index)
test_xgb_local$Destination_Type = as.numeric(test_xgb_local$Destination_Type)
test_xgb_local$Gender = as.numeric(test_xgb_local$Gender)

test_xgb_prod$Type_of_Cab = as.numeric(test_xgb_prod$Type_of_Cab)
test_xgb_prod$Confidence_Life_Style_Index = as.numeric(test_xgb_prod$Confidence_Life_Style_Index)
test_xgb_prod$Destination_Type = as.numeric(test_xgb_prod$Destination_Type)
test_xgb_prod$Gender = as.numeric(test_xgb_prod$Gender)

train_xgb_indep = train_xgb_local[,!colnames(train_xgb_local) %in% c('Surge_Pricing_Type')]
train_xgb_label= train_xgb_local[,'Surge_Pricing_Type']
test_xgb_indep = test_xgb_local[,!colnames(test_xgb_local) %in% c('Surge_Pricing_Type')]
test_xgb_label = test_xgb_local[,'Surge_Pricing_Type']
test_xgb_indep_prod = test_xgb_prod[,!colnames(test_xgb_prod) %in% c('Trip_ID','Surge_Pricing_Type')]

dtrain_local = xgb.DMatrix(data = as.matrix(train_xgb_indep), label = as.matrix(train_xgb_label))
dtest_local  = xgb.DMatrix(data = as.matrix(test_xgb_indep) , label = as.matrix(test_xgb_label))

watchlist <- list(test=dtest_local, train=dtrain_local)
xgb.local.model = xgb.train(data = dtrain_local,
                            watchlist=watchlist,
                            num_class = 4,
                            nround = 28,
                            max_depth = 6,
                            eta=0.3,
                            objective = 'multi:softmax',
                            colsample_bytree = 0.8,
                            #subsample = 0.8,
                            #early.stop.round = 10,
                            #maximize = T,
                            eval_metric= evalacc
)
xgb.local.pred = predict(xgb.local.model, newdata = as.matrix(test_xgb_indep))
acc_table_xgb = table(test_local$Surge_Pricing_Type, xgb.local.pred)
acc_table_xgb
acc_xgb = sum(diag(acc_table_xgb))/nrow(test_local)
acc_xgb
xgb.importance(mode = xgb.local.model, feature_names = colnames(train_xgb_indep))

#On prod
xgb.prod.pred = predict(xgb.local.model, newdata = as.matrix(test_xgb_indep_prod))

sub = data.frame(Trip_ID = test_prod$Trip_ID, Surge_Pricing_Type = xgb.prod.pred)
write.csv(sub, row.names=F, 'XGB24.csv')




