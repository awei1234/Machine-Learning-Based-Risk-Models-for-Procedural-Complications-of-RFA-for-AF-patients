setwd("/Users/weiyuna/Desktop/工作/合作项目/上海胸科/20211103导出数据/房颤/补充结局-不包含瓣膜手术")
# install.packages("modEvA")
# install.packages("terra")
# install.packages("kernlab")
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
library(gbm)
# library(kernlab)



### logistic regression (stepwise variable)
set.seed(123)
index = sample(nrow(data3),round(0.7*nrow(data3)), )
data.train = data3[index,]
data.test = data3[-index,]

fit.log.step1 <- glm(outcome ~ Gender + Age_factor + BMI_factor + HR_factor + LAD_factor + CHA2DS2_VACS_factor  + 
                       CREA_factor + TnI_factor + AST_factor + NT_proBNP_factor , data=data.train,family=binomial)

data.train$pred <- predict(fit.log.step1, data.train,type="response")
roc.train.step1 <- roc(data.train$outcome, data.train$pred,ci=TRUE,print.auc = TRUE,levels = c(0,1), direction = "<")
roc.train.step1
# plot(roc.train.step1,ci=TRUE,print.auc = TRUE,lty=2)

pre_cutoff<-optimal.cutpoints(X = "pred",status = "outcome", tag.healthy = 0, methods = "Youden",
                              data = data.train,ci.fit = TRUE,conf.level = 0.95)
summary(pre_cutoff)
caret::confusionMatrix(as.factor(as.numeric(data.train$pred>0.024 )), as.factor(data.train$outcome), positive = "1")

data.test$pred <- predict(fit.log.step1, newdata = data.test, type="response")
roc.test.step1 <- roc(data.test$outcome, data.test$pred,ci=TRUE,print.auc = TRUE,smooth=TRUE,levels = c(0,1), direction = "<")
roc.test.step1
# plot(roc.test.step1,ci=TRUE,print.auc = TRUE,lty=2)
caret::confusionMatrix(as.factor(as.numeric(data.test$pred>0.024)), as.factor(data.test$outcome), positive = "1") 


## brier score
brier <- function(df){
  n = nrow(df)
  df$delta_p_sq <- (df$pred-(as.numeric(df$outcome)-1))^2
  brier <- sum(df$delta_p_sq)/n
  return(brier)
}      

val.prob(data.test$pred,as.numeric(data.test$outcome),cex = 0.8)

## PR Curve
AUC(model = fit.log.step1, obs = data.test$outcome, pred = data.test$pred, simplif = TRUE, interval = 0.01,
    FPR.limits = c(0, 1), curve = "PR", method = "rank", plot = TRUE, diag = TRUE, 
    diag.col = "black", diag.lty = 1, curve.col = "blue", curve.lty = 2, curve.lwd = 2, plot.values = TRUE, 
    plot.digits = 2, plot.preds = FALSE, grid = FALSE,
    xlab = "Recall" , ylab = "Precision")



###############################################################################################################
###### k-fold cross validation
set.seed(1234)   
folds <- createMultiFolds(y=data3$outcome,k=5,times=20)   

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

brier_train <- 0
brier_test <- 0


for (i in 1:100){
  
  data.train <- data3[folds[[i]],]
  data.test <- data3[-folds[[i]],]
  
  Outcome <- "outcome"
  
  CandidateVariables <- c("Gender","Age_factor","BMI_factor","HR_factor", "LAD_factor","CHA2DS2_VACS_factor","CREA_factor","TnI_factor", "AST_factor"
                          ,"NT_proBNP_factor") 
  
  Formula <- formula(paste(paste(Outcome,"~", collapse=" "), 
                           paste(CandidateVariables, collapse=" + ")))
  
  model.step <- glm(Formula, data= data.train,family=binomial)
  
  data.train$p_prediction <- predict(model.step, data.train, type="response")
  data.test$p_prediction <- predict(model.step, data.test, type="response")
  
  roc.train <- roc(data.train$outcome, data.train$p_prediction,levels = c(0,1), direction = "<")
  roc.test <- roc(data.test$outcome, data.test$p_prediction,levels = c(0,1), direction = "<")
  
  cm_train <- caret::confusionMatrix(as.factor(as.numeric(data.train$p_prediction>0.025 )), as.factor(data.train$outcome), positive = "1")
  cm_test <- caret::confusionMatrix(as.factor(as.numeric(data.test$p_prediction>0.025)), as.factor(data.test$outcome), positive = "1") 
  
  c_train[i] <- roc.train$auc
  acc_train[i] <- cm_train$overall
  se_train[i] <- cm_train$byClass[1]
  spe_train[i] <- cm_train$byClass[2]
  ppv_train[i] <- cm_train$byClass[3]
  npv_train[i] <- cm_train$byClass[4]
  brier_train[i] <- mean((data.train$p_prediction-as.numeric(data.train$outcome))^2)
  
  c_test[i] <- roc.test$auc
  acc_test[i] <- cm_test$overall
  se_test[i] <- cm_test$byClass[1]
  spe_test[i] <- cm_test$byClass[2]
  ppv_test[i] <- cm_test$byClass[3]
  npv_test[i] <- cm_test$byClass[4]
  brier_test[i] <- mean((data.test$p_prediction-as.numeric(data.test$outcome))^2)
  
}
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
data_rf1 <- subset(data3,select = c(outcome,AF_category:Age,Weight:LAD,LVEF:UA,TT:CREA,Ccr:CK,NT_proBNP:Prior.CABG))

set.seed(12)
index = sample(nrow(data_rf1),round(0.7*nrow(data_rf1)), )
data_rf1_train = data_rf1[index,]
data_rf1_test = data_rf1[-index,]

# fit_rf1 <- randomForest(outcome~.,data = data_rf1_train,importance = TRUE)  ## all variables

fit_rf1 <- randomForest(outcome~DD+HR+FBG+Ccr+NT_proBNP+Age+ALB+CK+TSH+CREA,data = data_rf1_train,importance = TRUE) ##gini top 10 

# fit_rf1 <- randomForest(outcome~DD+HR+FBG+Ccr+NT_proBNP+Age+ALB+CK+TSH+CREA+GLU+TnI+TT+LDH+AST+UA+CHA2DS2_VACS+
#                           BMI+LVEDD+SBP,data = data_rf1_train,importance = TRUE) ##gini top 20   
# 
# fit_rf1 <- randomForest(outcome~DD+HR+FBG+Ccr+NT_proBNP+Age+ALB+CK+TSH+CREA+GLU+TnI+TT+LDH+AST+UA+CHA2DS2_VACS+
#                           BMI+LVEDD+SBP+LVESD+Weight+PTINR+DBP+LVEF+LAD+HAS_BLED+AF_category+Statins+Rivaroxaban,importance = TRUE) ##gini top 30  

# train
data_rf1_train$pred <- predict(fit_rf1, data_rf1_train,type="prob")[,2]
roc_rf1_train <- roc(data_rf1_train$outcome, data_rf1_train$pred,ci=TRUE,print.auc = TRUE,levels = c(0,1), direction = "<")
roc_rf1_train
# plot(roc_rf1_train,ci=TRUE,print.auc = TRUE,lty=2)

pre_cutoff<-optimal.cutpoints(X = "pred",status = "outcome", tag.healthy = 0, methods = "Youden",
                              data = data_rf1_train,ci.fit = TRUE,conf.level = 0.95)
summary(pre_cutoff)

caret::confusionMatrix(as.factor(as.numeric(data_rf1_train$pred>0.024)), as.factor(data_rf1_train$outcome), positive = "1")

# test
data_rf1_test$pred <- predict(fit_rf1, newdata = data_rf1_test,type="prob")[,2]
roc_rf1_test <- roc(data_rf1_test$outcome, data_rf1_test$pred,ci=TRUE,smooth=TRUE,print.auc = TRUE) #levels = c(0,1), direction = "<")
roc_rf1_test
plot(roc_rf1_test,ci=TRUE,print.auc = TRUE,lty=2)

caret::confusionMatrix(as.factor(as.numeric(data_rf1_test$pred>0.024)), as.factor(data_rf1_test$outcome), positive = "1") 

### feature importance
importance_rf1 <- data.frame(fit_rf1$importance)
importance_rf1 <- importance_rf1[order(importance_rf1$MeanDecreaseGini,decreasing = TRUE),]
head(importance_rf1)



###############################################################################################################
###### k-fold cross validation
set.seed(1234)   
folds <- createMultiFolds(y=data_rf1$outcome,k=5,times=10)   

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
  
  data.train <- data_rf1[folds[[i]],]
  data.test <- data_rf1[-folds[[i]],]
  
  # Outcome <- "outcome"
  
  # CandidateVariables <- c("DD","HR","FBG","Ccr", "NT_proBNP","Age","ALB","CK", "TSH","CREA")  # top10
  
  # CandidateVariables <- c("DD","HR","FBG","Ccr", "NT_proBNP","Age","ALB","CK", "TSH","CREA","GLU","TnI","TT","LDH","AST",
  #                         "UA","CHA2DS2_VACS","BMI","LVEDD","SBP")  # top20
  
  # CandidateVariables <- c("DD","HR","FBG","Ccr", "NT_proBNP","Age","ALB","CK", "TSH","CREA","GLU","TnI","TT","LDH","AST",
  #                         "UA","CHA2DS2_VACS","BMI","LVEDD","SBP","LVESD","Weight","PTINR","DBP","LVEF","LAD","HAS_BLED",
  #                         "AF_category","Statins","Rivaroxaban")  # top30
  
  # Formula <- formula(paste(paste(Outcome,"~", collapse=" "), 
  #                          paste(CandidateVariables, collapse=" + ")))
  
  # model.rf <- randomForest(Formula, data= data.train,importance = TRUE)
  model.rf <- randomForest(outcome~., data= data.train,importance = TRUE)
  
  data.train$p_prediction <- predict(model.rf, data.train, type="prob")[,2]
  data.test$p_prediction <- predict(model.rf, data.test, type="prob")[,2]
  
  pre_cutoff<-optimal.cutpoints(X = "p_prediction",status = "outcome", tag.healthy = 0, methods = "Youden",
                                data = data.train,ci.fit = TRUE,conf.level = 0.95)
  
  roc.train <- roc(data.train$outcome, data.train$p_prediction,levels = c(0,1), direction = "<")
  roc.test <- roc(data.test$outcome, data.test$p_prediction,levels = c(0,1), direction = "<")
  
  cm_train <- caret::confusionMatrix(as.factor(as.numeric(data.train$p_prediction>pre_cutoff$Youden$Global$optimal.cutoff$cutoff )), as.factor(data.train$outcome), positive = "1")
  cm_test <- caret::confusionMatrix(as.factor(as.numeric(data.test$p_prediction>pre_cutoff$Youden$Global$optimal.cutoff$cutoff)), as.factor(data.test$outcome), positive = "1") 
  
  c_train[i] <- roc.train$auc
  acc_train[i] <- cm_train$overall
  se_train[i] <- cm_train$byClass[1]
  spe_train[i] <- cm_train$byClass[2]
  ppv_train[i] <- cm_train$byClass[3]
  npv_train[i] <- cm_train$byClass[4]
  brier_train[i] <- mean((data.train$p_prediction-as.numeric(data.train$outcome))^2)
  
  cutoff[i] <- pre_cutoff$Youden$Global$optimal.cutoff$cutoff
  c_test[i] <- roc.test$auc
  acc_test[i] <- cm_test$overall
  se_test[i] <- cm_test$byClass[1]
  spe_test[i] <- cm_test$byClass[2]
  ppv_test[i] <- cm_test$byClass[3]
  npv_test[i] <- cm_test$byClass[4]
  brier_test[i] <- mean((data.test$p_prediction-as.numeric(data.test$outcome))^2)
  
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
data_gbm1 <- subset(data3,select = c(outcome,AF_category:Age,Weight:LAD,LVEF:UA,TT:CREA,Ccr:CK,NT_proBNP:Prior.CABG))
set.seed(1234)
index = sample(nrow(data_gbm1),round(0.7*nrow(data_gbm1)), )
data_gbm1_train = data_gbm1[index,]
data_gbm1_test = data_gbm1[-index,]

## hyper-parameter selection
data_gbm1_train$outcome <- ifelse(data_gbm1_train$outcome == "1","yes","no")
data_gbm1_test$outcome <- ifelse(data_gbm1_test$outcome == "1","yes","no")

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
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)


set.seed(1234)
train.gbm1 = train(
  outcome ~.,
  data = data_gbm1_train,
  trControl = cntrl,
  tuneGrid = grid,
  method = "gbm",
  metric = "ROC",
  # metric = "Kappa"
)

train.gbm1

## visualization
trellis.par.set(caretTheme())
plot(train.gbm1, metric = "ROC")

plot(train.gbm1, metric = "ROC", plotType = "level",
     scales = list(x = list(rot = 90)))

## feature importance
gbmImp1 <-varImp(train.gbm1,n.trees = 300, interaction.depth =1, shrinkage = 0.01,
       n.minobsinnode = 10)

plot(gbmImp1, top = 10)


## select the best hyper-parameter and train the model
data_gbm1 <- subset(data3,select = c(outcome,AF_category:Age,Weight:LAD,LVEF:UA,TT:CREA,Ccr:CK,NT_proBNP:Prior.CABG))
set.seed(555)
index = sample(nrow(data_gbm1),round(0.7*nrow(data_gbm1)), )
data_gbm1_train = data_gbm1[index,]
data_gbm1_test = data_gbm1[-index,]

## the function of gbm can not predict with propability, so chose the train function of caret package
grid = expand.grid(interaction.depth = 1, n.trees = 300, shrinkage = 0.01,n.minobsinnode = 10)
cntrl = trainControl(method = "cv",number = 5)

fit_gbm1 = train(
  outcome ~ DD+Ccr+HR+Age+TnI+FBG+CHA2DS2_VACS+HAS_BLED+ALB+NT_proBNP,
  data = data_gbm1_train,trControl = cntrl,tuneGrid = grid,method = "gbm",metric = "Kappa")  

# fit_gbm1 <- gbm(outcome ~.,data = data_gbm1_train,distribution = "adaboost",n.trees = 300,interaction.depth = 1,
#                 shrinkage = 0.01,n.minobsinnode = 10)  ## all variable

# fit_gbm1 <- gbm(outcome~DD+Ccr+HR+Age+TnI+FBG+CHA2DS2_VACS+HAS_BLED+ALB+NT_proBNP,data = data_gbm1_train,distribution = "adaboost",n.trees = 300,interaction.depth = 1,
#                 shrinkage = 0.01,n.minobsinnode = 10)  # top10

# fit_gbm1 <- gbm(outcome~DD+Ccr+HR+Age+TnI+FBG+CHA2DS2_VACS+HAS_BLED+ALB+NT_proBNP+TSH+
#                   TT+CK+AST+CREA+DBP+BMI+LVEF+GLU+LDH,data = data_gbm1_train,distribution = "adaboost",n.trees = 300,interaction.depth = 1,
#                 shrinkage = 0.01,n.minobsinnode = 10)  # top20

# fit_gbm1 <- gbm(outcome~DD+Ccr+HR+Age+TnI+FBG+CHA2DS2_VACS+HAS_BLED+ALB+NT_proBNP+TSH+
#                   TT+CK+AST+CREA+DBP+BMI+LVEF+GLU+LDH+PTINR+Heart.failure+Ever.smoking+Rivaroxaban+
#                   CHD+Hyperlipidemia+Statins+Dabigatran+Other.antiplatelet.agents+Clopidogrel,
#                 data = data_gbm1_train,distribution = "adaboost",n.trees = 300,interaction.depth = 1,
#                 shrinkage = 0.01,n.minobsinnode = 10)  # top30

# train set
data_gbm1_train$pred <- predict(fit_gbm1, data_gbm1_train,type = "prob")[,2]
roc_gbm1_train <- roc(data_gbm1_train$outcome, data_gbm1_train$pred,ci=TRUE,print.auc = TRUE,levels = c(0,1), direction = "<")
roc_gbm1_train
plot(roc_gbm1_train,ci=TRUE,print.auc = TRUE,lty=2)

pre_cutoff<-optimal.cutpoints(X = "pred",status = "outcome", tag.healthy = 0, methods = "Youden",
                              data = data_gbm1_train,ci.fit = TRUE,conf.level = 0.95)
summary(pre_cutoff)

caret::confusionMatrix(as.factor(as.numeric(data_gbm1_train$pred>0.01467029)), as.factor(data_gbm1_train$outcome), positive = "1")

# test set
data_gbm1_test$pred <- predict(fit_gbm1, newdata = data_gbm1_test,type = "prob")[,2]
roc_gbm1_test <- roc(data_gbm1_test$outcome, data_gbm1_test$pred,ci=TRUE,smooth=TRUE,print.auc = TRUE) #levels = c(0,1), direction = "<")
roc_gbm1_test
plot(roc_gbm1_test,ci=TRUE,print.auc = TRUE,lty=2)

caret::confusionMatrix(as.factor(as.numeric(data_gbm1_test$pred>0.01467029)), as.factor(data_gbm1_test$outcome), positive = "1") 



###############################################################################################################
###### k-fold cross validation
set.seed(1234)   
folds <- createMultiFolds(y=data_gbm1$outcome,k=5,times=10)   

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
  
  data.train <- data_gbm1[folds[[i]],]
  data.test <- data_gbm1[-folds[[i]],]
  
  grid = expand.grid(interaction.depth = 1, n.trees = 300, shrinkage = 0.01,n.minobsinnode = 10)
  
  cntrl = trainControl(method = "cv",number = 5)
  
  model.gbm = train(
    outcome ~ DD+Ccr+HR+Age+TnI+FBG+CHA2DS2_VACS+HAS_BLED+ALB+NT_proBNP,
    data = data.train,trControl = cntrl,tuneGrid = grid,method = "gbm",metric = "Kappa")
  
  # Outcome <- "outcome"
  
  # CandidateVariables <- c("DD","Ccr","HR","Age", "TnI","FBG","CHA2DS2_VACS","HAS_BLED", "ALB","NT_proBNP")  # top10
  
  # CandidateVariables <- c("DD","Ccr","HR","Age", "TnI","FBG","CHA2DS2_VACS","HAS_BLED", "ALB","NT_proBNP",
  #                         "TSH","TT","CK","AST","CREA","DBP","BMI","LVEF","GLU","LDH")  # top20
  
  # CandidateVariables <- c("DD","Ccr","HR","Age", "TnI","FBG","CHA2DS2_VACS","HAS_BLED", "ALB","NT_proBNP",
  #                         "TSH","TT","CK","AST","CREA","DBP","BMI","LVEF","GLU","LDH",
  #                         "PTINR","Heart.failure","Ever.smoking","Rivaroxaban","CHD","Hyperlipidemia",
  #                         "Statins","Dabigatran","Other.antiplatelet.agents","Clopidogrel")  # top30
  
  # Formula <- formula(paste(paste(Outcome,"~", collapse=" "),
  #                          paste(CandidateVariables, collapse=" + ")))
  
  # model.gbm <- gbm(Formula, data= data.train,distribution = "adaboost",n.trees = 300,interaction.depth = 1,shrinkage = 0.01,n.minobsinnode = 10)
  # model.gbm <- gbm(outcome~., data= data.train,distribution = "adaboost",n.trees = 300,interaction.depth = 1,shrinkage = 0.01,n.minobsinnode = 10)
  
  data.train$p_prediction <- predict(model.gbm, data.train, type="prob")[,2]
  data.test$p_prediction <- predict(model.gbm, data.test, type="prob")[,2]
  
  pre_cutoff<-optimal.cutpoints(X = "p_prediction",status = "outcome", tag.healthy = 0, methods = "Youden",
                                data = data.train,ci.fit = TRUE,conf.level = 0.95)
  
  roc.train <- roc(data.train$outcome, data.train$p_prediction,levels = c(0,1), direction = "<")
  roc.test <- roc(data.test$outcome, data.test$p_prediction,levels = c(0,1), direction = "<")
  
  cm_train <- caret::confusionMatrix(as.factor(as.numeric(data.train$p_prediction>pre_cutoff$Youden$Global$optimal.cutoff$cutoff )), as.factor(data.train$outcome), positive = "1")
  cm_test <- caret::confusionMatrix(as.factor(as.numeric(data.test$p_prediction>pre_cutoff$Youden$Global$optimal.cutoff$cutoff)), as.factor(data.test$outcome), positive = "1") 
  
  c_train[i] <- roc.train$auc
  acc_train[i] <- cm_train$overall
  se_train[i] <- cm_train$byClass[1]
  spe_train[i] <- cm_train$byClass[2]
  ppv_train[i] <- cm_train$byClass[3]
  npv_train[i] <- cm_train$byClass[4]
  brier_train[i] <- mean((data.train$p_prediction-as.numeric(data.train$outcome))^2)
  
  cutoff[i] <- pre_cutoff$Youden$Global$optimal.cutoff$cutoff
  c_test[i] <- roc.test$auc
  acc_test[i] <- cm_test$overall
  se_test[i] <- cm_test$byClass[1]
  spe_test[i] <- cm_test$byClass[2]
  ppv_test[i] <- cm_test$byClass[3]
  npv_test[i] <- cm_test$byClass[4]
  brier_test[i] <- mean((data.test$p_prediction-as.numeric(data.test$outcome))^2)
  
}
round(CI(c_train,ci=0.95),3)
round(CI(acc_train,ci=0.95),3)
round(CI(se_train,ci=0.95),3)
round(CI(spe_train,ci=0.95),3)
round(CI(ppv_train,ci=0.95),3)
round(CI(npv_train,ci=0.95),3)
round(CI(brier_train,ci=0.95),3)
round(CI(cutoff,ci=0.95),3)
round(CI(c_test,ci=0.95),3)
round(CI(acc_test,ci=0.95),3)
round(CI(se_test,ci=0.95),3)
round(CI(spe_test,ci=0.95),3)
round(CI(ppv_test,ci=0.95),3)
round(CI(npv_test,ci=0.95),3)
round(CI(brier_test,ci=0.95),3)



#######################################################################################################
#### xgboost model (use the dateset before imputation)
data_xgb<-read.csv("AF_new.csv")
data.xgb<-subset(data_xgb,select = c(patient_id,outcome:UA,Ca:CREA,Ccr:CK,NT_proBNP:Prior.CABG))
names(data.xgb)

data.xgb$Persistent_AF<-ifelse(data.xgb$AF_category=="2",1,0)
data.xgb$Chronic_AF<-ifelse(data.xgb$AF_category=="3",1,0)
summary(data.xgb)


### hyper-parameter selection
set.seed(1234)
index = sample(nrow(data.xgb),round(0.7*nrow(data.xgb)), )
data.train.xg = data.xgb[index,]
data.test.xg = data.xgb[-index,]
summary(data.train.xg)
summary(data.test.xg)

data.train.xg$outcome_h <- ifelse(data.train.xg$outcome_h == "1","yes","no")
data.test.xg$outcome_h <- ifelse(data.test.xg$outcome_h == "1","yes","no")

grid = expand.grid(
  nrounds = c(100,200,300),
  colsample_bytree = c(0.5, 0.8, 1),
  # min_child_weight = c(1, 3, 5, 7),
  eta = c(0.01, 0.05, 0.1,0.5), #0.3 is default,
  gamma = c(1,0.5,0.1,0.01),
  # lambda = c(0.01,0.1, 1),
  subsample = c(0.5,0.7),
  max_depth = c(3, 5, 7,10)
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
  y = as.factor(data.train.xg[, 2]),
  trControl = cntrl,
  tuneGrid = grid,
  method = "xgbTree",
  metric = "ROC"
  # metric = "Kappa"
)

train.xgb



###############################################################################################################
## k-fold cross validation
set.seed(1234)
folds <- createMultiFolds(y=data.xgb$outcome,k=5,times=20)   

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
  
  
  ## data preparation
  
  # data.train.xg1<-data.matrix(data.train.xg[,c("Gender","Age","BMI","HR","LAD","CHA2DS2_VACS","CREA","TnI","AST","NT_proBNP")])  ## stepwise feature 
  
  data.train.xg1<-data.matrix(data.train.xg[,c("DD","HR","Ccr","FBG","TT","Age","ALB","AST","NT_proBNP","DBP")])  ##top10
 
  # data.train.xg1<-data.matrix(data.train.xg[,c("DD","HR","Ccr","FBG","TT","Age","ALB","AST","NT_proBNP","DBP", "HAS_BLED","LVESD","CHA2DS2_VACS","LDH","Weight")])   ###top15
  
  # data.train.xg1<-data.matrix(data.train.xg[,c("DD","HR","Ccr","FBG","TT","Age","ALB","AST","NT_proBNP","DBP", "HAS_BLED","LVESD","CHA2DS2_VACS","LDH","Weight",
  #                                              "GLU","BMI", "LVEF","TSH","LVEDD")])   ###top20
  # 
  # data.train.xg1<-data.matrix(data.train.xg[,c("DD","HR","Ccr","FBG","TT","Age","ALB","AST","NT_proBNP","DBP", "HAS_BLED","LVESD","CHA2DS2_VACS","LDH","Weight",
  #                                              "GLU","BMI", "LVEF","TSH","LVEDD","PTINR","UA","CREA","TnI","LAD","CK","SBP","Statins","CHD","Rivaroxaban")])  ##top30
  
  # data.train.xg1<-data.matrix(data.train.xg[,c(6:7,9:23,27:71)]) ### all features
  
  data.train.xg2<-Matrix(data.train.xg1,sparse = T)
  data.train.xg_y<-data.train.xg$outcome
  traindata.xg<-list(data=data.train.xg2,label=data.train.xg_y)
  dtrain<-xgb.DMatrix(data = traindata.xg$data,label=traindata.xg$label)
  
  
  # data.test.xg1<-data.matrix(data.test.xg[,c("Gender","Age","BMI","HR","LAD","CHA2DS2_VACS","CREA","TnI","AST","NT_proBNP")])  ##stepwise variable
  
  data.test.xg1<-data.matrix(data.test.xg[,c("DD","HR","Ccr","FBG","TT","Age","ALB","AST","NT_proBNP","DBP")])  ##top10
  
  # data.test.xg1<-data.matrix(data.test.xg[,c("DD","HR","Ccr","FBG","TT","Age","ALB","AST","NT_proBNP","DBP", "HAS_BLED","LVESD","CHA2DS2_VACS","LDH","Weight")])   ###top15
  
  # data.test.xg1<-data.matrix(data.test.xg[,c("DD","HR","Ccr","FBG","TT","Age","ALB","AST","NT_proBNP","DBP", "HAS_BLED","LVESD","CHA2DS2_VACS","LDH","Weight",
  #                                              "GLU","BMI", "LVEF","TSH","LVEDD")])  ##top20

  # data.test.xg1<-data.matrix(data.test.xg[,c("DD","HR","Ccr","FBG","TT","Age","ALB","AST","NT_proBNP","DBP", "HAS_BLED","LVESD","CHA2DS2_VACS","LDH","Weight",
  #                                            "GLU","BMI", "LVEF","TSH","LVEDD","PTINR","UA","CREA","TnI","LAD","CK","SBP","Statins","CHD","Rivaroxaban")])  ##top30
  
  # data.test.xg1<-data.matrix(data.test.xg[,c(6:7,9:23,27:71)])   ## all variables
  
  data.test.xg2<-Matrix(data.test.xg1,sparse = T)
  data.test.xg_y<-data.test.xg$outcome
  testdata.xg<-list(data=data.test.xg2,label=data.test.xg_y)
  dtest<-xgb.DMatrix(data = testdata.xg$data,label=testdata.xg$label)
  
  model_xgb_fuhe <-xgboost(data = dtrain,objective="binary:logistic", booster="gbtree",nrounds = 200, max_depth = 3, eta = 0.01, gamma =
                        0.01, colsample_bytree = 0.5, min_child_weight = 1 , subsample = 0.7)
  
  # importance_fuhe<-xgb.importance(data.train.xg2@Dimnames[[2]],model = model_xgb_fuhe)
  # xgb.ggplot.importance(importance_fuhe,rel_to_first = TRUE,n_clusters = 3, measure = "Cover")
  # xgb.ggplot.importance(importance_fuhe,rel_to_first = TRUE,n_clusters = 3, measure = "Gain",top_n = 20)
  # xgb.ggplot.shap.summary(data = traindata.xg$data, model = model_xgb_fuhe,target_class = 1,top_n = 20)
  
  ## predict
  pre.xgb.train <-predict(model_xgb_fuhe,dtrain)
  pre.xgb.test <-predict(model_xgb_fuhe,newdata = dtest)
  
  roc.xgb.train <- roc(response = data.train.xg_y, predictor = pre.xgb.train,ci=TRUE,print.auc = TRUE)
  roc.xgb.test <- roc(response = data.test.xg_y, predictor = pre.xgb.test,ci=TRUE,print.auc = TRUE)
  
  cm.xgb_train <- caret::confusionMatrix(as.factor(as.numeric(pre.xgb.train>0.082 )), as.factor(data.train.xg_y), positive = "1")
  cm.xgb_test <- caret::confusionMatrix(as.factor(as.numeric(pre.xgb.test>0.082)), as.factor(data.test.xg_y), positive = "1") 
  
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
  
  brier_train[i] <- mean(pre.xgb.train-as.numeric(data.train.xg_y))^2
  brier_test[i] <- mean(pre.xgb.test-as.numeric(data.test.xg_y))^2
  
}
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
#### xgboost model with the top 10 features
data<-read.csv("AF_new.csv")
data.xgb<-subset(data,select = c(patient_id,outcome:UA,Ca:CREA,Ccr:CK,NT_proBNP:Prior.CABG))
data.xgb$Persistent_AF<-ifelse(data.xgb$AF_category=="2",1,0)
data.xgb$Chronic_AF<-ifelse(data.xgb$AF_category=="3",1,0)
summary(data.xgb)

set.seed(666)
index = sample(nrow(data.xgb),round(0.7*nrow(data.xgb)), )
data.train.xg = data.xgb[index,]
data.test.xg = data.xgb[-index,]

## data preprocessing
data.train.xg1<-data.matrix(data.train.xg[,c("DD","HR","Ccr","FBG","TT","Age","ALB","AST","NT_proBNP","DBP")])  ##top10
data.train.xg2<-Matrix(data.train.xg1,sparse = T)
data.train.xg_y<-data.train.xg$outcome
traindata.xg<-list(data=data.train.xg2,label=data.train.xg_y)
dtrain<-xgb.DMatrix(data = traindata.xg$data,label=traindata.xg$label)

data.test.xg1<-data.matrix(data.test.xg[,c("DD","HR","Ccr","FBG","TT","Age","ALB","AST","NT_proBNP","DBP")])  ##top10
data.test.xg2<-Matrix(data.test.xg1,sparse = T)
data.test.xg_y<-data.test.xg$outcome
testdata.xg<-list(data=data.test.xg2,label=data.test.xg_y)
dtest<-xgb.DMatrix(data = testdata.xg$data,label=testdata.xg$label)


## train the model
model_xgb3<-xgboost(data = dtrain,objective="binary:logistic", booster="gbtree",nrounds = 200, max_depth = 3, eta = 0.01, gamma =
                      0.01, colsample_bytree = 0.5, min_child_weight = 1, subsample = 0.7)   

importance3<-xgb.importance(data.train.xg2@Dimnames[[2]],model = model_xgb3)
# xgb.ggplot.importance(importance3,rel_to_first = TRUE)
# xgb.plot.importance(importance3,rel_to_first = TRUE,xlab = "Relative importance")

## predict
pre_xgb_train3<-predict(model_xgb3,dtrain)
pre_xgb_test3<-predict(model_xgb3,newdata = dtest)

pre_cutoff<-optimal.cutpoints(X = "pre_xgb_train3",status = "data.train.xg_y", tag.healthy = 0, methods = "Youden",
                              data = data.frame(cbind(pre_xgb_train3,data.train.xg_y)),ci.fit = TRUE,conf.level = 0.95)
summary(pre_cutoff)

## model evaluation
xgb.cf_train3<-caret::confusionMatrix(as.factor(as.numeric(pre_xgb_train3>0.08786933)),as.factor(data.train.xg_y), positive = "1")
xgb.cf_train3
roc_xgb_train3 <- roc(response = data.train.xg_y, predictor = pre_xgb_train3,ci=TRUE,print.auc = TRUE)
roc_xgb_train3
# plot(roc_xgb_train3,ci=TRUE,print.auc = TRUE,lty=2)

xgb.cf_test3<-caret::confusionMatrix(as.factor(as.numeric(pre_xgb_test3>0.08786933)),as.factor(data.test.xg_y), positive = "1")
xgb.cf_test3
roc_xgb_test3 <- roc(response = data.test.xg_y, predictor = pre_xgb_test3,ci=TRUE,smooth=TRUE,print.auc = TRUE)
roc_xgb_test3
plot(roc_xgb_test3,ci=TRUE,print.auc = TRUE,lty=2)


## PR curve
AUC(obs = data.test.xg_y, pred = pre_xgb_test3, simplif = TRUE, interval = 0.005,
    FPR.limits = c(0, 1), curve = "PR", method = "trapezoid", plot = TRUE, diag = TRUE, 
    diag.col = "black", diag.lty = 1, curve.col = "blue", curve.lty = 2, curve.lwd = 2, plot.values = TRUE, 
    plot.digits = 3, plot.preds = FALSE, grid = FALSE,
    xlab = "Recall" , ylab = "Precision")





## ROC curves of four models
par(mfrow=c(1,1))
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
legend("topleft",legend = "Any complications", text.font = 2,box.lwd = "none")





  

