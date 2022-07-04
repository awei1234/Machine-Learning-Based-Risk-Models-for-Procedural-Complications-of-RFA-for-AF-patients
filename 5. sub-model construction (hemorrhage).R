setwd("/Users/weiyuna/Desktop/工作/合作项目/上海胸科/20211103导出数据/房颤/补充结局-不包含瓣膜手术")
library(caret)
library(Rmisc)
library(MASS)
library(pROC)
library(Matrix)
library(grid)
library(xgboost)
library(terra)
library(modEvA)
library(OptimalCutpoints)
library(randomForest)
library(e1071)


### logistic regression (stepwise variable)
set.seed(12)
index = sample(nrow(data3),round(0.7*nrow(data3)), )
data.train = data3[index,]
data.test = data3[-index,]

fit.log.step5 <- glm(outcome_b ~ Gender + AF_category + Age_factor + LAD_factor  + CREA_factor + DD_factor + TnI_factor +
                       AST_factor + Antiplatelet.agents + Hypertension + Diabetes , data=data.train,family=binomial)

# training set 
data.train$pred.train <- predict(fit.log.step5, data.train,type="response")
roc.train.step5 <- roc(data.train$outcome_b, data.train$pred.train,ci=TRUE,print.auc = TRUE, levels = c(0,1), direction = "<")
roc.train.step5
# plot(roc.train.step5,ci=TRUE,print.auc = TRUE,lty=2)
pre_cutoff<-optimal.cutpoints(X = "pred.train",status = "outcome_b", tag.healthy = 0, methods = "Youden",
                              data = data.train,ci.fit = TRUE,conf.level = 0.95)
summary(pre_cutoff)

caret::confusionMatrix(as.factor(as.numeric(data.train$pred.train>0.01119605 )), as.factor(data.train$outcome_b), positive = "1")

## test set
pred.test <- predict(fit.log.step5, newdata = data.test, type="response")
roc.test.step5 <- roc(data.test$outcome_b, pred.test,ci=TRUE,smooth=TRUE,print.auc = TRUE,levels = c(0,1), direction = "<")
roc.test.step5
# plot(roc.test.step5,ci=TRUE,print.auc = TRUE,lty=2)
caret::confusionMatrix(as.factor(as.numeric(pred.test>0.01119605)), as.factor(data.test$outcome_b), positive = "1") 

# brier score
val.prob(pred.test,as.numeric(data.test$outcome_b),cex = 0.8)


###############################################################################################################
## k-fold cross validation
set.seed(1234)
folds <- createMultiFolds(y=data3$outcome_b,k=5,times=20)   

c_train <- 0
acc_train <- 0
se_train <- 0
spe_train <- 0
ppv_train <- 0
npv_train <- 0

c_test <- 0
acc_test <- 0
se_test <- 0
spe_test <- 0
ppv_test <- 0
npv_test <- 0

# brier_train <- 0
# brier_test <- 0



for (i in 1:100){
  
  data.train <- data3[folds[[i]],]
  data.test <- data3[-folds[[i]],]
  
  Outcome <- "outcome_b"
  
  CandidateVariables <- c("Gender", "AF_category", "Age_factor","LAD_factor","CREA_factor","DD_factor",
                          "TnI_factor","AST_factor","Antiplatelet.agents", "Hypertension","Diabetes")
  
  
  Formula <- formula(paste(paste(Outcome,"~", collapse=" "), 
                           paste(CandidateVariables, collapse=" + ")))
  
  model.step <- glm(Formula, data= data.train,family=binomial)
  
  data.train$p_prediction <- predict(model.step, data.train, type="response")
  data.test$p_prediction <- predict(model.step, data.test, type="response")
  
  roc.train <- roc(data.train$outcome_b, data.train$p_prediction,levels = c(0,1), direction = "<")
  roc.test <- roc(data.test$outcome_b, data.test$p_prediction,levels = c(0,1), direction = "<")
  
  cm_train <- caret::confusionMatrix(as.factor(as.numeric(data.train$p_prediction>0.006 )), as.factor(data.train$outcome_b), positive = "1")
  cm_test <- caret::confusionMatrix(as.factor(as.numeric(data.test$p_prediction>0.006)), as.factor(data.test$outcome_b), positive = "1") 
  
  c_train[i] <- roc.train$auc
  acc_train[i] <- cm_train$overall
  se_train[i] <- cm_train$byClass[1]
  spe_train[i] <- cm_train$byClass[2]
  ppv_train[i] <- cm_train$byClass[3]
  npv_train[i] <- cm_train$byClass[4]
  brier_train[i] <- mean((data.train$p_prediction-as.numeric(data.train$outcome_b))^2)
  
  c_test[i] <- roc.test$auc
  acc_test[i] <- cm_test$overall
  se_test[i] <- cm_test$byClass[1]
  spe_test[i] <- cm_test$byClass[2]
  ppv_test[i] <- cm_test$byClass[3]
  npv_test[i] <- cm_test$byClass[4]
  brier_test[i] <- mean((data.test$p_prediction-as.numeric(data.test$outcome_b))^2)
}

summary(c_train) 
summary(acc_train)
summary(se_train)
summary(spe_train)
summary(ppv_train)
summary(npv_train)
summary(c_test)    
summary(acc_test)
summary(se_test)
summary(spe_test)
summary(ppv_test)
summary(npv_test)
round(CI(c_train,ci=0.95),3)
round(CI(acc_train,ci=0.95),3)
round(CI(se_train,ci=0.95),3)
round(CI(spe_train,ci=0.95),3)
round(CI(ppv_train,ci=0.95),3)
round(CI(npv_train,ci=0.95),3)
# round(CI(brier_train,ci=0.95),3)

round(CI(c_test,ci=0.95),3)
round(CI(acc_test,ci=0.95),3)
round(CI(se_test,ci=0.95),3)
round(CI(spe_test,ci=0.95),3)
round(CI(ppv_test,ci=0.95),3)
round(CI(npv_test,ci=0.95),3)
# round(CI(brier_test,ci=0.95),3)



############# Random forest model
data_rf3 <- subset(data3,select = c(outcome_b,AF_category:Age,Weight:LAD,LVEF:UA,TT:CREA,Ccr:CK,NT_proBNP:Prior.CABG))

set.seed(12)
index = sample(nrow(data_rf3),round(0.7*nrow(data_rf3)), )
data_rf3_train = data_rf3[index,]
data_rf3_test = data_rf3[-index,]

# fit_rf3 <- randomForest(outcome_b~.,data = data_rf3_train,importance = TRUE)  ## all features

fit_rf3 <- randomForest(outcome_b~DD+ALB+HR+TnI+CHA2DS2_VACS+FBG+Age+AST+TT+Ccr,data = data_rf3_train,importance = TRUE) ##gini top 10 

# fit_rf3 <- randomForest(outcome_b~DD+ALB+HR+TnI+CHA2DS2_VACS+FBG+Age+AST+TT+Ccr+CREA+CK+SBP+LDH+GLU+
#                           NT_proBNP+HAS_BLED+LVEDD+PTINR+BMI,data = data_rf3_train,importance = TRUE) ##gini top 20   

# fit_rf3 <- randomForest(outcome_b~DD+ALB+HR+TnI+CHA2DS2_VACS+FBG+Age+AST+TT+Ccr+CREA+CK+SBP+LDH+GLU+
#                           NT_proBNP+HAS_BLED+LVEDD+PTINR+BMI+DBP+LVESD+UA+TSH+LVEF+Weight+LAD+CCB+Heparin+Stroke,
#                         data = data_rf3_train,importance = TRUE) ##gini top 30  

# training set
data_rf3_train$pred <- predict(fit_rf3, data_rf3_train,type="prob")[,2]
roc_rf3_train <- roc(data_rf3_train$outcome_b, data_rf3_train$pred,ci=TRUE,print.auc = TRUE,levels = c(0,1), direction = "<")
roc_rf3_train
# plot(roc_rf3_train,ci=TRUE,print.auc = TRUE,lty=2)

pre_cutoff<-optimal.cutpoints(X = "pred",status = "outcome_b", tag.healthy = 0, methods = "Youden",
                              data = roc_rf3_train,ci.fit = TRUE,conf.level = 0.95)
summary(pre_cutoff)

caret::confusionMatrix(as.factor(as.numeric(data_rf3_train$pred>0.594)), as.factor(data_rf3_train$outcome_b), positive = "1")

# test set
data_rf3_test$pred <- predict(fit_rf3, newdata = data_rf3_test,type="prob")[,2]
roc_rf3_test <- roc(data_rf3_test$outcome_b, data_rf3_test$pred,ci=TRUE,smooth=TRUE,print.auc = TRUE) #levels = c(0,1), direction = "<")
roc_rf3_test
plot(roc_rf3_test,ci=TRUE,print.auc = TRUE,lty=2)

caret::confusionMatrix(as.factor(as.numeric(data_rf3_test$pred>0.024)), as.factor(data_rf3_test$outcome_b), positive = "1") 

### feature importance
importance_rf3 <- data.frame(fit_rf3$importance)
importance_rf3 <- importance_rf3[order(importance_rf3$MeanDecreaseGini,decreasing = TRUE),]
head(importance_rf3)
# varImpPlot(fit_rf3,n.var=min(30,nrow(fit_r1f$importance)),main="Top 30 varible importance")  


###############################################################################################################
## k-fold cross validation
set.seed(1234)   
folds <- createMultiFolds(y=data_rf3$outcome_b,k=5,times=10)   

c_train <- 0
acc_train <- 0
se_train <- 0
spe_train <- 0
ppv_train <- 0
npv_train <- 0

cutoff <- 0
c_test <- 0
acc_test <- 0
se_test <- 0
spe_test <- 0
ppv_test <- 0
npv_test <- 0

brier_train <- 0
brier_test <- 0


for (i in 1:50){
  
  data.train <- data_rf3[folds[[i]],]
  data.test <- data_rf3[-folds[[i]],]
  
  # Outcome <- "outcome_b"
  
  # CandidateVariables <- c("DD","ALB","HR","TnI", "CHA2DS2_VACS","FBG","Age","AST", "TT","Ccr")  ## TOP 10
  # 
  # CandidateVariables <- c("DD","ALB","HR","TnI", "CHA2DS2_VACS","FBG","Age","AST", "TT","Ccr",
  #                         "CREA","CK","SBP","LDH","GLU","NT_proBNP","HAS_BLED","LVEDD","PTINR","BMI")  ## TOP 20
  
  # CandidateVariables <- c("DD","ALB","HR","TnI", "CHA2DS2_VACS","FBG","Age","AST", "TT","Ccr",
  #                         "CREA","CK","SBP","LDH","GLU","NT_proBNP","HAS_BLED","LVEDD","PTINR","BMI",
  #                         "DBP","LVESD","UA","TSH","LVEF","Weight","LAD","CCB","Heparin","Stroke") ## TOP 30
  
  Formula <- formula(paste(paste(Outcome,"~", collapse=" "), 
                           paste(CandidateVariables, collapse=" + ")))
  
  # model.rf <- randomForest(Formula, data= data.train,importance = TRUE)
  model.rf <- randomForest(outcome_b~., data= data.train,importance = TRUE)
  
  data.train$p_prediction <- predict(model.rf, data.train, type="prob")[,2]
  data.test$p_prediction <- predict(model.rf, data.test, type="prob")[,2]
  
  pre_cutoff<-optimal.cutpoints(X = "p_prediction",status = "outcome_b", tag.healthy = 0, methods = "Youden",
                                data = data.test,ci.fit = TRUE,conf.level = 0.95)
  
  roc.train <- roc(data.train$outcome_b, data.train$p_prediction,levels = c(0,1), direction = "<")
  roc.test <- roc(data.test$outcome_b, data.test$p_prediction,levels = c(0,1), direction = "<")
  
  cm_train <- caret::confusionMatrix(as.factor(as.numeric(data.train$p_prediction>pre_cutoff$Youden$Global$optimal.cutoff$cutoff )), as.factor(data.train$outcome_b), positive = "1")
  cm_test <- caret::confusionMatrix(as.factor(as.numeric(data.test$p_prediction>pre_cutoff$Youden$Global$optimal.cutoff$cutoff)), as.factor(data.test$outcome_b), positive = "1") 
  
  c_train[i] <- roc.train$auc
  acc_train[i] <- cm_train$overall
  se_train[i] <- cm_train$byClass[1]
  spe_train[i] <- cm_train$byClass[2]
  ppv_train[i] <- cm_train$byClass[3]
  npv_train[i] <- cm_train$byClass[4]
  brier_train[i] <- mean((data.train$p_prediction-as.numeric(data.train$outcome_b))^2)
  cutoff[i] <- pre_cutoff$Youden$Global$optimal.cutoff$cutoff
  
  c_test[i] <- roc.test$auc
  acc_test[i] <- cm_test$overall
  se_test[i] <- cm_test$byClass[1]
  spe_test[i] <- cm_test$byClass[2]
  ppv_test[i] <- cm_test$byClass[3]
  npv_test[i] <- cm_test$byClass[4]
  brier_test[i] <- mean((data.test$p_prediction-as.numeric(data.test$outcome_b))^2)
  
}
round(CI(c_train,ci=0.95),3)
round(CI(acc_train,ci=0.95),3)
round(CI(se_train,ci=0.95),3)
round(CI(spe_train,ci=0.95),3)
round(CI(ppv_train,ci=0.95),3)
round(CI(npv_train,ci=0.95),3)
# round(CI(brier_train,ci=0.95),3)

round(CI(cutoff,ci=0.95),3)
round(CI(c_test,ci=0.95),3)
round(CI(acc_test,ci=0.95),3)
round(CI(se_test,ci=0.95),3)
round(CI(spe_test,ci=0.95),3)
round(CI(ppv_test,ci=0.95),3)
round(CI(npv_test,ci=0.95),3)
# round(CI(brier_test,ci=0.95),3)




############# gbm model
data_gbm3 <- subset(data3,select = c(outcome_b,AF_category:Age,Weight:LAD,LVEF:UA,TT:CREA,Ccr:CK,NT_proBNP:Prior.CABG))
set.seed(1234)
index = sample(nrow(data_gbm3),round(0.7*nrow(data_gbm3)), )
data_gbm3_train = data_gbm3[index,]
data_gbm3_test = data_gbm3[-index,]

## hyper-parameter selection
data_gbm3_train$outcome_b <- ifelse(data_gbm3_train$outcome_b == "1","yes","no")
data_gbm3_test$outcome_b <- ifelse(data_gbm3_test$outcome_b == "1","yes","no")

grid = expand.grid(
  interaction.depth = c(1,3,5,9), 
  n.trees = (1:10)*50, 
  shrinkage = c(0.01,0.05,0.1,0.2),
  n.minobsinnode = c(10,20)
)
grid

cntrl = trainControl(
  method = "cv",
  number = 5,
  # repeats = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)


set.seed(1234)
train.gbm3 = train(
  outcome_b ~.,
  data = data_gbm3_train,
  trControl = cntrl,
  tuneGrid = grid,
  method = "gbm",
  metric = "ROC",
  # metric = "Kappa"
)

train.gbm3

## visualization
trellis.par.set(caretTheme())
plot(train.gbm3, metric = "ROC")
plot(train.gbm3, metric = "ROC", plotType = "level",
     scales = list(x = list(rot = 90)))

## feature importance
gbmImp3 <-varImp(train.gbm3,n.trees = 50, interaction.depth =1, shrinkage = 0.05,
                 n.minobsinnode = 10)
plot(gbmImp3, top = 10)


## select the best hyper-parameter and train the model
data_gbm3 <- subset(data3,select = c(outcome_b,AF_category:Age,Weight:LAD,LVEF:UA,TT:CREA,Ccr:CK,NT_proBNP:Prior.CABG))
set.seed(16)
index = sample(nrow(data_gbm3),round(0.7*nrow(data_gbm3)), )
data_gbm3_train = data_gbm3[index,]
data_gbm3_test = data_gbm3[-index,]

## the function of gbm can not predict with propability, so chose the train function of caret package
grid = expand.grid(interaction.depth = 1, n.trees = 50, shrinkage = 0.05,n.minobsinnode = 10)
cntrl = trainControl(method = "cv",number = 5)

fit_gbm3 = train(
  outcome_b ~ DD+TnI+CHA2DS2_VACS+Age+HAS_BLED+Ccr+CREA+Antiplatelet.agents+ALB+FBG,
  data = data_gbm3_train,trControl = cntrl,tuneGrid = grid,method = "gbm",metric = "Kappa")

# fit_gbm3 <- gbm(outcome_b ~.,data = data_gbm3_train,distribution = "adaboost",n.trees = 50,interaction.depth = 1,
#                 shrinkage = 0.05,n.minobsinnode = 10)  ## all variable

# fit_gbm3 <- gbm(outcome_b~DD+TnI+CHA2DS2_VACS+Age+HAS_BLED+Ccr+CREA+Antiplatelet.agents+ALB+FBG,
#                 data = data_gbm3_train,distribution = "adaboost",n.trees = 50,interaction.depth = 1,
#                 shrinkage = 0.05,n.minobsinnode = 10)  # top10

# fit_gbm3 <- gbm(outcome_b~DD+TnI+CHA2DS2_VACS+Age+HAS_BLED+Ccr+CREA+Antiplatelet.agents+ALB+
#                   FBG+AST+HR+CKD+DBP+TT+BMI+SBP+CHD+Anticoagulants+Ever.smoking,
#                 data = data_gbm3_train,distribution = "adaboost",n.trees = 50,interaction.depth = 1,
#                 shrinkage = 0.05,n.minobsinnode = 10)  # top20
# 
# fit_gbm3 <- gbm(outcome_b~DD+TnI+CHA2DS2_VACS+Age+HAS_BLED+Ccr+CREA+Antiplatelet.agents+ALB+
#                   FBG+AST+HR+CKD+DBP+TT+BMI+SBP+CHD+Anticoagulants+Ever.smoking+Warfarin+LVEF+
#                   UA+Hyperlipidemia+Heart.failure+Clopidogrel+Hypertension+ACEI_ARB+LVESD+Prior.CABG,
#                 data = data_gbm3_train,distribution = "adaboost",n.trees = 50,interaction.depth = 1,
#                 shrinkage = 0.05,n.minobsinnode = 10)  # top30

# training set
data_gbm3_train$pred <- predict(fit_gbm3, data_gbm3_train,type = "prob")[,2]
roc_gbm3_train <- roc(data_gbm3_train$outcome_b, data_gbm3_train$pred,ci=TRUE,print.auc = TRUE,levels = c(0,1), direction = "<")
roc_gbm3_train
# plot(roc_gbm3_train,ci=TRUE,print.auc = TRUE,lty=2)

pre_cutoff<-optimal.cutpoints(X = "pred",status = "outcome_b", tag.healthy = 0, methods = "Youden",
                              data = data_gbm3_train,ci.fit = TRUE,conf.level = 0.95)
summary(pre_cutoff)

caret::confusionMatrix(as.factor(as.numeric(data_gbm3_train$pred>4.433149e-03)), as.factor(data_gbm3_train$outcome_b), positive = "1")

# test set
data_gbm3_test$pred <- predict(fit_gbm3, newdata = data_gbm3_test,type = "prob")[,2]
roc_gbm3_test <- roc(data_gbm3_test$outcome_b, data_gbm3_test$pred,ci=TRUE,smooth=TRUE,print.auc = TRUE) #levels = c(0,1), direction = "<")
roc_gbm3_test
plot(roc_gbm3_test,ci=TRUE,print.auc = TRUE,lty=2)

caret::confusionMatrix(as.factor(as.numeric(data_gbm3_test$pred>4.433149e-03)), as.factor(data_gbm3_test$outcome_b), positive = "1") 



###############################################################################################################
## k-fold cross validation
data_gbm3 <- subset(data3,select = c(outcome_b,AF_category:Age,Weight:LAD,LVEF:UA,TT:CREA,Ccr:CK,NT_proBNP:Prior.CABG))
set.seed(1234)   
folds <- createMultiFolds(y=data_gbm3$outcome_b,k=5,times=10)   

c_train <- 0
acc_train <- 0
se_train <- 0
spe_train <- 0
ppv_train <- 0
npv_train <- 0
cutoff <- 0

c_test <- 0
acc_test <- 0
se_test <- 0
spe_test <- 0
ppv_test <- 0
npv_test <- 0

brier_train <- 0
brier_test <- 0


for (i in 1:50){
  
  data.train <- data_gbm3[folds[[i]],]
  data.test <- data_gbm3[-folds[[i]],]
  
  grid = expand.grid(interaction.depth = 1, n.trees = 50, shrinkage = 0.05,n.minobsinnode = 10)
  
  cntrl = trainControl(method = "cv",number = 5)
  
  model.gbm = train(
    outcome_b ~ DD+TnI+CHA2DS2_VACS+Age+HAS_BLED+Ccr+CREA+Antiplatelet.agents+ALB+FBG,
    data = data.train,trControl = cntrl,tuneGrid = grid,method = "gbm",metric = "Kappa")
  
  # Outcome <- "outcome_b"
  
  # CandidateVariables <- c("DD","TnI","CHA2DS2_VACS","Age","HAS_BLED","Ccr","CREA","Antiplatelet.agents","ALB","FBG")  # top10
  
  # CandidateVariables <- c("DD","TnI","CHA2DS2_VACS","Age","HAS_BLED","Ccr","CREA","Antiplatelet.agents","ALB","FBG",
  #                         "AST","HR","CKD","DBP","TT","BMI","SBP","CHD","Anticoagulants","Ever.smoking")  # top20
  
  # CandidateVariables <- c("DD","TnI","CHA2DS2_VACS","Age","HAS_BLED","Ccr","CREA","Antiplatelet.agents","ALB","FBG",
  #                         "AST","HR","CKD","DBP","TT","BMI","SBP","CHD","Anticoagulants","Ever.smoking",
  #                         "Warfarin","LVEF","UA","Hyperlipidemia","Heart.failure","Clopidogrel",
  #                         "Hypertension","ACEI_ARB","LVESD","Prior.CABG")  # top30

  # Formula <- formula(paste(paste(Outcome,"~", collapse=" "),
  #                          paste(CandidateVariables, collapse=" + ")))
  
  # model.gbm <- gbm(Formula, data= data.train,distribution = "adaboost",n.trees = 50,interaction.depth = 1,shrinkage = 0.05,n.minobsinnode = 10)
  # model.gbm <- gbm(outcome_b~., data= data.train,distribution = "adaboost",n.trees = 50,interaction.depth = 1,shrinkage = 0.05,n.minobsinnode = 10)
  
  data.train$p_prediction <- predict(model.gbm, data.train, type="prob")[,2]
  data.test$p_prediction <- predict(model.gbm, data.test, type="prob")[,2]
  
  pre_cutoff<-optimal.cutpoints(X = "p_prediction",status = "outcome_b", tag.healthy = 0, methods = "Youden",
                                data = data.train,ci.fit = TRUE,conf.level = 0.95)
  
  roc.train <- roc(data.train$outcome_b, data.train$p_prediction,levels = c(0,1), direction = "<")
  roc.test <- roc(data.test$outcome_b, data.test$p_prediction,levels = c(0,1), direction = "<")
  
  cm_train <- caret::confusionMatrix(as.factor(as.numeric(data.train$p_prediction>pre_cutoff$Youden$Global$optimal.cutoff$cutoff )), as.factor(data.train$outcome_b), positive = "1")
  cm_test <- caret::confusionMatrix(as.factor(as.numeric(data.test$p_prediction>pre_cutoff$Youden$Global$optimal.cutoff$cutoff)), as.factor(data.test$outcome_b), positive = "1") 
  
  c_train[i] <- roc.train$auc
  acc_train[i] <- cm_train$overall
  se_train[i] <- cm_train$byClass[1]
  spe_train[i] <- cm_train$byClass[2]
  ppv_train[i] <- cm_train$byClass[3]
  npv_train[i] <- cm_train$byClass[4]
  brier_train[i] <- mean((data.train$p_prediction-as.numeric(data.train$outcome_b))^2)
  
  cutoff[i] <- pre_cutoff$Youden$Global$optimal.cutoff$cutoff
  c_test[i] <- roc.test$auc
  acc_test[i] <- cm_test$overall
  se_test[i] <- cm_test$byClass[1]
  spe_test[i] <- cm_test$byClass[2]
  ppv_test[i] <- cm_test$byClass[3]
  npv_test[i] <- cm_test$byClass[4]
  brier_test[i] <- mean((data.test$p_prediction-as.numeric(data.test$outcome_b))^2)
  
}
round(CI(c_train,ci=0.95),3)
round(CI(acc_train,ci=0.95),3)
round(CI(se_train,ci=0.95),3)
round(CI(spe_train,ci=0.95),3)
round(CI(ppv_train,ci=0.95),3)
round(CI(npv_train,ci=0.95),3)
# round(CI(brier_train,ci=0.95),3)
round(CI(cutoff,ci=0.95),3)
round(CI(c_test,ci=0.95),3)
round(CI(acc_test,ci=0.95),3)
round(CI(se_test,ci=0.95),3)
round(CI(spe_test,ci=0.95),3)
round(CI(ppv_test,ci=0.95),3)
round(CI(npv_test,ci=0.95),3)
# round(CI(brier_test,ci=0.95),3)



#######################################################################################################
#### xgboost model
data<-read.csv("AF_new.csv")
data.xgb<-subset(data,select = c(patient_id,outcome:UA,Ca:CREA,Ccr:CK,NT_proBNP:Prior.CABG))
names(data.xgb)
data.xgb$Persistent_AF<-ifelse(data.xgb$AF_category=="2",1,0)
data.xgb$Chronic_AF<-ifelse(data.xgb$AF_category=="3",1,0)

## hyper-parameter selection
set.seed(1234)
index = sample(nrow(data.xgb),round(0.7*nrow(data.xgb)), )
data.train.xg = data.xgb[index,]
data.test.xg = data.xgb[-index,]
summary(data.train.xg)
summary(data.test.xg)

data.train.xg$outcome_h <- ifelse(data.train.xg$outcome_h == "1","yes","no")
data.test.xg$outcome_h <- ifelse(data.test.xg$outcome_h == "1","yes","no")

grid = expand.grid(
  nrounds = c(100,200,500),
  colsample_bytree = c(0.5, 0.7, 0.8, 0.9, 1),
  min_child_weight = c(1, 3, 5, 7),
  eta = c(0.01, 0.015, 0.025, 0.05, 0.1), #0.3 is default,
  gamma = c(1,0.5,0.1,0.01),
  # lambda = c(0.01, 0.1,1),
  subsample = c(0.7),
  max_depth = c(3, 5, 6, 7, 10)
)
grid

cntrl = trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  returnData = FALSE,
  returnResamp = "final",
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)


set.seed(1234)
train.xgb = train(
  x = data.train.xg[,c(6:7,9:71)],
  y = as.factor(data.train.xg[, 4]),
  trControl = cntrl,
  tuneGrid = grid,
  method = "xgbTree",
  metric = "ROC"
  # metric = "Kappa"
)

train.xgb



#######################################################################################################
## xgboost model (k-fold cross validation)
set.seed(1234)
folds <- createMultiFolds(y=data.xgb$outcome_b,k=5,times=20)   

c_train <- 0
acc_train <- 0
se_train <- 0
spe_train <- 0
ppv_train <- 0
npv_train <- 0
brier_train <- 0

c_test <- 0
acc_test <- 0
se_test <- 0
spe_test <- 0
ppv_test <- 0
npv_test <- 0
brier_test <- 0

for (i in 1:100){
  data.train.xg <- data.xgb[folds[[i]],]
  data.test.xg <- data.xgb[-folds[[i]],]
  
  ## data preprocessing
  
  # data.train.xg1<-data.matrix(data.train.xg[,c("Persistent_AF","Chronic_AF", "Age","LAD", "CREA","DD",
  #                                              "TnI","AST","Antiplatelet.agents","Hypertension","Diabetes","Gender")])  ## stepwise feature
  
  # data.train.xg1<-data.matrix(data.train.xg[,c("DD","ALB","TT","Age","HR","TnI","CHA2DS2_VACS","DBP","GLU","Ccr")])  ## top10
  # 
  # data.train.xg1<-data.matrix(data.train.xg[,c("DD","ALB","TT","Age","HR","TnI","CHA2DS2_VACS","DBP","GLU","Ccr","FBG", "CK","LDH","UA","LVEDD")])  ## top15
  
  # data.train.xg1<-data.matrix(data.train.xg[,c("DD","ALB","TT","Age","HR","TnI","CHA2DS2_VACS","DBP","GLU","Ccr","FBG", "CK","LDH","UA",
  #                                            "LVEDD", "AST","Weight","HAS_BLED","BMI","NT_proBNP")])  ## top20
  # 
  # data.train.xg1<-data.matrix(data.train.xg[,c("DD","ALB","TT","Age","HR","TnI","CHA2DS2_VACS","DBP","GLU","Ccr","FBG", "CK","LDH","UA",
  #                                              "LVEDD", "AST","Weight","HAS_BLED","BMI","NT_proBNP","LAD","CREA","Clopidogrel",
  #                                              "PTINR","LVESD", "CCB","TSH","CHD","SBP","LVEF")])  ## top30
  
  data.train.xg1<-data.matrix(data.train.xg[,c(6:7,9:23,27:71)]) ## all features
  
  data.train.xg2<-Matrix(data.train.xg1,sparse = T)
  data.train.xg_y<-data.train.xg$outcome_b
  traindata.xg<-list(data=data.train.xg2,label=data.train.xg_y)
  dtrain<-xgb.DMatrix(data = traindata.xg$data,label=traindata.xg$label)
  
  
  # data.test.xg1<-data.matrix(data.test.xg[,c("Persistent_AF","Chronic_AF", "Age","LAD", "CREA","DD",
  #                                            "TnI","AST","Antiplatelet.agents","Hypertension","Diabetes","Gender")])  ## stepwise feature
  
  # data.test.xg1<-data.matrix(data.test.xg[,c("DD","ALB","TT","Age","HR","TnI","CHA2DS2_VACS","DBP","GLU","Ccr")])  ## top10
  # 
  # data.test.xg1<-data.matrix(data.test.xg[,c("DD","ALB","TT","Age","HR","TnI","CHA2DS2_VACS","DBP","GLU","Ccr","FBG", "CK","LDH","UA","LVEDD")])  ## top15
  
  # data.test.xg1<-data.matrix(data.test.xg[,c("DD","ALB","TT","Age","HR","TnI","CHA2DS2_VACS","DBP","GLU","Ccr","FBG", "CK","LDH","UA",
  #                                            "LVEDD", "AST","Weight","HAS_BLED","BMI","NT_proBNP")])  ## top20
  # 
  # data.test.xg1<-data.matrix(data.test.xg[,c("DD","ALB","TT","Age","HR","TnI","CHA2DS2_VACS","DBP","GLU","Ccr","FBG", "CK","LDH","UA",
  #                                            "LVEDD", "AST","Weight","HAS_BLED","BMI","NT_proBNP","LAD","CREA","Clopidogrel",
  #                                            "PTINR","LVESD", "CCB","TSH","CHD","SBP","LVEF")])    ## top30
  
  data.test.xg1<-data.matrix(data.test.xg[,c(6:7,9:23,27:71)])   ## all features
  
  data.test.xg2<-Matrix(data.test.xg1,sparse = T)
  data.test.xg_y<-data.test.xg$outcome_b
  testdata.xg<-list(data=data.test.xg2,label=data.test.xg_y)
  dtest<-xgb.DMatrix(data = testdata.xg$data,label=testdata.xg$label)
  
  model_xgb_chuxue <-xgboost(data = dtrain,objective="binary:logistic", booster="gbtree",nrounds = 200, max_depth = 3, eta = 0.01, gamma =
                             0.01, colsample_bytree = 0.5, min_child_weight = 1 , subsample = 0.7)
  
  ## output the result of feature importance
  # importance_chuxue<-xgb.importance(data.train.xg2@Dimnames[[2]],model = model_xgb_chuxue)
  # xgb.ggplot.importance(importance_chuxue,rel_to_first = TRUE,n_clusters = 3, measure = "Cover")
  # xgb.ggplot.importance(importance_chuxue,rel_to_first = TRUE,n_clusters = 3, measure = "Gain",top_n = 20)
  # xgb.ggplot.shap.summary(data = traindata.xg$data, model = model_xgb_chuxue,target_class = 1,top_n = 20)
  
  ## predict
  pre.xgb.train <-predict(model_xgb_chuxue,dtrain)
  pre.xgb.test <-predict(model_xgb_chuxue,newdata = dtest)
  
  roc.xgb.train <- roc(response = data.train.xg_y, predictor = pre.xgb.train,ci=TRUE,print.auc = TRUE)
  roc.xgb.test <- roc(response = data.test.xg_y, predictor = pre.xgb.test,ci=TRUE,print.auc = TRUE)
  
  cm.xgb_train <- caret::confusionMatrix(as.factor(as.numeric(pre.xgb.train>0.073 )), as.factor(data.train.xg_y), positive = "1")
  cm.xgb_test <- caret::confusionMatrix(as.factor(as.numeric(pre.xgb.test>0.073)), as.factor(data.test.xg_y), positive = "1") 
  
  c_train[i] <- roc.xgb.train$auc
  acc_train[i] <- cm.xgb_train$overall
  se_train[i] <- cm.xgb_train$byClass[1]
  spe_train[i] <- cm.xgb_train$byClass[2]
  ppv_train[i] <- cm.xgb_train$byClass[3]
  npv_train[i] <- cm.xgb_train$byClass[4]
  
  
  c_test[i] <- roc.xgb.test$auc
  acc_test[i] <- cm.xgb_test$overall
  se_test[i] <- cm.xgb_test$byClass[1]
  spe_test[i] <- cm.xgb_test$byClass[2]
  ppv_test[i] <- cm.xgb_test$byClass[3]
  npv_test[i] <- cm.xgb_test$byClass[4]
  
  brier_train[i] <- mean((pre.xgb.train-as.numeric(data.train.xg_y))^2)
  brier_test[i] <- mean((pre.xgb.test-as.numeric(data.test.xg_y))^2)
  
}

summary(c_train) 
summary(acc_train)
summary(se_train)
summary(spe_train)
summary(ppv_train)
summary(npv_train)
summary(c_test)    
summary(acc_test)
summary(se_test)
summary(spe_test)
summary(ppv_test)
summary(npv_test)
round(CI(c_train,ci=0.95),3)
round(CI(acc_train,ci=0.95),3)
round(CI(se_train,ci=0.95),3)
round(CI(spe_train,ci=0.95),3)
round(CI(ppv_train,ci=0.95),3)
round(CI(npv_train,ci=0.95),3)
round(CI(brier_train,ci=0.95),3)

round(CI(c_test,ci=0.95),3)
round(CI(acc_test,ci=0.95),3)
round(CI(se_test,ci=0.95),3)
round(CI(spe_test,ci=0.95),3)
round(CI(ppv_test,ci=0.95),3)
round(CI(npv_test,ci=0.95),3)
round(CI(brier_test,ci=0.95),3)



#######################################################################################################
#### xgboost model
data<-read.csv("AF_new.csv")
data.xgb<-subset(data,select = c(patient_id,outcome:UA,Ca:CREA,Ccr:CK,NT_proBNP:Prior.CABG))
names(data.xgb)

data.xgb$Persistent_AF<-ifelse(data.xgb$AF_category=="2",1,0)
data.xgb$Chronic_AF<-ifelse(data.xgb$AF_category=="3",1,0)
summary(data.xgb)

#######################################################################################################
## xgboost model with the top10 features
set.seed(999)
index = sample(nrow(data.xgb),round(0.7*nrow(data.xgb)), )
data.train.xg = data.xgb[index,]
data.test.xg = data.xgb[-index,]

# data preparation
data.train.xg1<-data.matrix(data.train.xg[,c("DD","ALB","TT","Age","HR","TnI","CHA2DS2_VACS","DBP","FBG","Ccr")])
data.train.xg2<-Matrix(data.train.xg1,sparse = T)
data.train.xg_y<-data.train.xg$outcome_b
traindata.xg<-list(data=data.train.xg2,label=data.train.xg_y)
dtrain<-xgb.DMatrix(data = traindata.xg$data,label=traindata.xg$label)

data.test.xg1<-data.matrix(data.test.xg[,c("DD","ALB","TT","Age","HR","TnI","CHA2DS2_VACS","DBP","FBG","Ccr")])
data.test.xg2<-Matrix(data.test.xg1,sparse = T)
data.test.xg_y<-data.test.xg$outcome_b
testdata.xg<-list(data=data.test.xg2,label=data.test.xg_y)
dtest<-xgb.DMatrix(data = testdata.xg$data,label=testdata.xg$label)

## fit the model
model_xgb9<-xgboost(data = dtrain,objective="binary:logistic", booster="gbtree",nrounds = 200, max_depth = 3, eta = 0.01, gamma =
                      0.01, colsample_bytree = 0.5, min_child_weight = 1 , subsample = 0.7)

## output the result of feature importance

# importance9<-xgb.importance(data.train.xg2@Dimnames[[2]],model = model_xgb9)
# xgb.ggplot.importance(importance9,rel_to_first = TRUE)
# xgb.plot.importance(importance9,rel_to_first = TRUE,xlab = "Relative importance")
# xgb.ggplot.shap.summary(data = traindata.xg$data, model = model_xgb9,target_class = 1)

## predict
pre_xgb_train9<-predict(model_xgb9,dtrain)
pre_xgb_test9<-predict(model_xgb9,newdata = dtest)

## model evaluation
xgb.cf_train9<-caret::confusionMatrix(as.factor(as.numeric(pre_xgb_train9>0.08)),as.factor(data.train.xg_y), positive = "1")
xgb.cf_train9
roc_xgb_train9 <- roc(response = data.train.xg_y, predictor = pre_xgb_train9,ci=TRUE,print.auc = TRUE,levels = c(0,1), direction = "<")
roc_xgb_train9
# plot(roc_xgb_train9,ci=TRUE,print.auc = TRUE,lty=2)

xgb.cf_test9<-caret::confusionMatrix(as.factor(as.numeric(pre_xgb_test9>0.08)),as.factor(data.test.xg_y), positive = "1")
xgb.cf_test9
roc_xgb_test9 <- roc(response = data.test.xg_y, predictor = pre_xgb_test9,smooth=TRUE,ci=TRUE,print.auc = TRUE,levels = c(0,1), direction = "<")
roc_xgb_test9
# plot(roc_xgb_test9,ci=TRUE,print.auc = TRUE,lty=2)



### ROC curve of four models
par(mfrow=c(1,1))
plot(roc.test.step5,ci=TRUE,lty=2,col="black",legacy.axes=T)
plot(roc_rf3_test,add=TRUE,ci=TRUE,lty=2,col="blue")
plot(roc_gbm3_test,add=TRUE,ci=TRUE,lty=2,col="purple")
plot(roc_xgb_test9,add=TRUE,ci=TRUE,lty=2,col="red")
abline(h = seq(0,1,0.1), v = seq(0,1,0.1), col = "lightgrey", lty = 1,lwd = 0.3)
legend("bottomright",legend = c("AUC (95% CI)",
                                "LR: 0.74 (0.72-0.76)",
                                "RF: 0.78 (0.75-0.80)",
                                "GBM: 0.75 (0.71-0.78)",
                                "XGBoost: 0.82 (0.80-0.84)"),
       col = c("white","black","blue","purple","red"),lty = 2,lwd = 1,cex = 0.8, text.font = 2,box.lwd = "none")
legend("topleft",legend = "Hemorrhage/hematoma", text.font = 2,box.lwd = "none")




## combined roc
par(mfrow=c(1,3))

## any complication
plot(roc.test.step1,ci=TRUE,lty=2,col="black",legacy.axes=T)
plot(roc_rf1_test,add=TRUE,ci=TRUE,lty=2,col="blue")
plot(roc_gbm1_test,add=TRUE,ci=TRUE,lty=2,col="purple")
plot(roc_xgb_test3,add=TRUE,ci=TRUE,lty=2,col="red")
abline(h = seq(0,1,0.1), v = seq(0,1,0.1), col = "lightgrey", lty = 1,lwd = 0.3)
legend("bottomright",legend = c("AUC (95% CI)",
                                "LR: 0.66 (0.64-0.67)",
                                "RF: 0.71 (0.69-0.72)",
                                "GBM: 0.72 (0.70-0.74)",
                                "XGBoost: 0.73 (0.70-0.75)"),
       col = c("white","black","blue","purple","red"),lty = 2,lwd = 1,cex = 0.8, text.font = 2,box.lwd = "none")
legend("topleft",legend = "Any complication", text.font = 2,box.lwd = "none")

## cardiac effusion
plot(roc.test.step2,ci=TRUE,lty=2,col="black",legacy.axes=T)
plot(roc_rf2_test,add=TRUE,ci=TRUE,lty=2,col="blue")
plot(roc_gbm2_test,add=TRUE,ci=TRUE,lty=2,col="purple")
plot(roc_xgb_test5,add=TRUE,ci=TRUE,lty=2,col="red")
abline(h = seq(0,1,0.1), v = seq(0,1,0.1), col = "lightgrey", lty = 1,lwd = 0.3)
legend("bottomright",legend = c("AUC (95% CI)",
                                "LR: 0.66 (0.64-0.68)",
                                "RF: 0.62 (0.59-0.66)",
                                "GBM: 0.64 (0.61-0.68)",
                                "XGBoost: 0.71 (0.69-0.73)"),
       col = c("white","black","blue","purple","red"),lty = 2,lwd = 1,cex = 0.8, text.font = 2,box.lwd = "none")
legend("topleft",legend = "Cardiac effusion/tamponade", text.font = 2,box.lwd = "none")

## hemorrhage
plot(roc.test.step5,ci=TRUE,lty=2,col="black",legacy.axes=T)
plot(roc_rf3_test,add=TRUE,ci=TRUE,lty=2,col="blue")
plot(roc_gbm3_test,add=TRUE,ci=TRUE,lty=2,col="purple")
plot(roc_xgb_test9,add=TRUE,ci=TRUE,lty=2,col="red")
abline(h = seq(0,1,0.1), v = seq(0,1,0.1), col = "lightgrey", lty = 1,lwd = 0.3)
legend("bottomright",legend = c("AUC (95% CI)",
                                "LR: 0.74 (0.72-0.76)",
                                "RF: 0.78 (0.75-0.80)",
                                "GBM: 0.75 (0.71-0.78)",
                                "XGBoost: 0.82 (0.80-0.84)"),
       col = c("white","black","blue","purple","red"),lty = 2,lwd = 1,cex = 0.8, text.font = 2,box.lwd = "none")
legend("topleft",legend = "Hemorrhage/hematoma", text.font = 2,box.lwd = "none")

