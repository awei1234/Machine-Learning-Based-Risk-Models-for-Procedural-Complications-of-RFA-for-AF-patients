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
set.seed(123)
index = sample(nrow(data3),round(0.7*nrow(data3)), )
data.train = data3[index,]
data.test = data3[-index,]

fit.log.step2 <- glm(outcome_h ~ Gender + Age_factor + BMI_factor  + AST_factor + ALB_factor + NT_proBNP_factor, data=data.train,family=binomial)

# training set 
data.train$pred.train <- predict(fit.log.step2, data.train,type="response")
roc.train.step2 <- roc(data.train$outcome_h, data.train$pred.train,ci=TRUE,print.auc = TRUE) #levels = c(0,1), direction = "<")
roc.train.step2
plot(roc.train.step2,ci=TRUE,print.auc = TRUE,lty=2)

pre_cutoff<-optimal.cutpoints(X = "pred.train",status = "outcome_h", tag.healthy = 0, methods = "Youden",
                              data = data.train,ci.fit = TRUE,conf.level = 0.95)
summary(pre_cutoff)
caret::confusionMatrix(as.factor(as.numeric(data.train$pred.train>0.007 )), as.factor(data.train$outcome_h), positive = "1")

## test set
pred.test <- predict(fit.log.step2, newdata = data.test, type="response")
roc.test.step2 <- roc(data.test$outcome_h, pred.test,smooth=TRUE,ci=TRUE,print.auc = TRUE,levels = c(0,1), direction = "<")
roc.test.step2
plot(roc.test.step2,ci=TRUE,print.auc = TRUE,lty=2)
caret::confusionMatrix(as.factor(as.numeric(pred.test>0.007)), as.factor(data.test$outcome_h), positive = "1") 



###############################################################################################################
## k-fold cross validation
set.seed(1234)   
folds <- createMultiFolds(y=data3$outcome_h,k=5,times=20)   

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

  Outcome <- "outcome_h"
  CandidateVariables <- c("Gender","Age_factor", "BMI_factor","AST_factor","ALB_factor", "NT_proBNP_factor") ###添加与他汀可能有相关性的合并疾病Stroke/PAD/Hyperlipidemia/CHD
  
  Formula <- formula(paste(paste(Outcome,"~", collapse=" "), 
                           paste(CandidateVariables, collapse=" + ")))
  
  model.step <- glm(Formula, data= data.train,family=binomial)
  
  data.train$p_prediction <- predict(model.step, data.train, type="response")
  data.test$p_prediction <- predict(model.step, data.test, type="response")
  
  roc.train <- roc(data.train$outcome_h, data.train$p_prediction,print.auc = TRUE,levels = c(0,1), direction = "<")
  roc.test <- roc(data.test$outcome_h, data.test$p_prediction,print.auc = TRUE,levels = c(0,1), direction = "<")
  
  cm_train <- caret::confusionMatrix(as.factor(as.numeric(data.train$p_prediction>0.009 )), as.factor(data.train$outcome_h), positive = "1")
  cm_test <- caret::confusionMatrix(as.factor(as.numeric(data.test$p_prediction>0.009)), as.factor(data.test$outcome_h), positive = "1") 
  
  c_train[i] <- roc.train$auc
  acc_train[i] <- cm_train$overall
  se_train[i] <- cm_train$byClass[1]
  spe_train[i] <- cm_train$byClass[2]
  ppv_train[i] <- cm_train$byClass[3]
  npv_train[i] <- cm_train$byClass[4]
  brier_train[i] <- mean((data.train$p_prediction-as.numeric(data.train$outcome_h))^2)
  
  c_test[i] <- roc.test$auc
  acc_test[i] <- cm_test$overall
  se_test[i] <- cm_test$byClass[1]
  spe_test[i] <- cm_test$byClass[2]
  ppv_test[i] <- cm_test$byClass[3]
  npv_test[i] <- cm_test$byClass[4]
  brier_test[i] <- mean((data.test$p_prediction-as.numeric(data.test$outcome_h))^2)
  
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


############# Random forest model
data_rf2 <- subset(data3,select = c(outcome_h,AF_category:Age,Weight:LAD,LVEF:UA,TT:CREA,Ccr:CK,NT_proBNP:Prior.CABG))

set.seed(33)
index = sample(nrow(data_rf2),round(0.7*nrow(data_rf2)), )
data_rf2_train = data_rf2[index,]
data_rf2_test = data_rf2[-index,]

# fit_rf2 <- randomForest(outcome_h~.,data = data_rf2_train,importance = TRUE) ##测试集auc=0.5856

fit_rf2 <- randomForest(outcome_h~FBG+NT_proBNP+GLU+Age+CK+ALB+AST+TSH+Ccr+UA,data = data_rf2_train,importance = TRUE) ##gini top 10 测试集auc=0.5649

# fit_rf2 <- randomForest(outcome_h~FBG+NT_proBNP+GLU+Age+CK+ALB+AST+TSH+Ccr+UA+DD+LDH+CREA+BMI+DBP+
#                           TT+LAD+TnI+HR+PTINR,data = data_rf2_train,importance = TRUE) ##gini top 20   测试集auc=0.5844

# fit_rf2 <- randomForest(outcome_h~FBG+NT_proBNP+GLU+Age+CK+ALB+AST+TSH+Ccr+UA+DD+LDH+CREA+BMI+DBP+TT+LAD+TnI+HR+PTINR+
#                           SBP+LVEF+Weight+LVEDD+LVESD+CHA2DS2_VACS+HAS_BLED+AF_category+Rivaroxaban+Hyperlipidemia,
#                         data = data_rf2_train,importance = TRUE) ##gini top 30  测试集auc=0.6204


# training set
data_rf2_train$pred <- predict(fit_rf2, data_rf2_train,type="prob")[,2]
roc_rf2_train <- roc(data_rf2_train$outcome_h, data_rf2_train$pred,ci=TRUE,print.auc = TRUE,levels = c(0,1), direction = "<")
roc_rf2_train
# plot(roc_rf2_train,ci=TRUE,print.auc = TRUE,lty=2)

pre_cutoff<-optimal.cutpoints(X = "pred",status = "outcome_h", tag.healthy = 0, methods = "Youden",
                              data = data_rf2_train,ci.fit = TRUE,conf.level = 0.95)
summary(pre_cutoff)

caret::confusionMatrix(as.factor(as.numeric(data_rf2_train$pred>0.01)), as.factor(data_rf2_train$outcome_h), positive = "1")

# test set
data_rf2_test$pred <- predict(fit_rf2, newdata = data_rf2_test,type="prob")[,2]
roc_rf2_test <- roc(data_rf2_test$outcome_h, data_rf2_test$pred,ci=TRUE,smooth=TRUE,print.auc = TRUE,levels = c(0,1), direction = "<")
roc_rf2_test
plot(roc_rf2_test,ci=TRUE,print.auc = TRUE,lty=2)

caret::confusionMatrix(as.factor(as.numeric(data_rf2_test$pred>0.01)), as.factor(data_rf2_test$outcome_h), positive = "1") 

### feature importance
importance_rf2 <- data.frame(fit_rf2$importance)
importance_rf2 <- importance_rf2[order(importance_rf2$MeanDecreaseGini,decreasing = TRUE),]
head(importance_rf2)
# varImpPlot(fit_rf2,n.var=min(30,nrow(fit_r1f$importance)),main="Top 30 varible importance")  


###############################################################################################################
## k-fold cross validation
set.seed(1234)   
folds <- createMultiFolds(y=data_rf2$outcome_h,k=5,times=10)   

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
  
  data.train <- data_rf2[folds[[i]],]
  data.test <- data_rf2[-folds[[i]],]
  
  # Outcome <- "outcome_h"
  
  # CandidateVariables <- c("FBG","NT_proBNP","GLU","Age", "CK","ALB","AST","TSH", "Ccr","UA")  ## TOP 10
  # 
  # CandidateVariables <- c("FBG","NT_proBNP","GLU","Age","CK","ALB","AST","TSH","Ccr","UA","DD","LDH","CREA","BMI",
  #                         "DBP","TT","LAD","TnI","HR","PTINR")  ## TOP 20
  
  # CandidateVariables <- c("FBG","NT_proBNP","GLU","Age","CK","ALB","AST","TSH","Ccr","UA","DD","LDH","CREA","BMI",
  #                         "DBP","TT","LAD","TnI","HR","PTINR","SBP","LVEF","Weight","LVEDD","LVESD","CHA2DS2_VACS",
  #                         "HAS_BLED","AF_category","Rivaroxaban","Hyperlipidemia") ## TOP 30
  # 
  # Formula <- formula(paste(paste(Outcome,"~", collapse=" "), 
  #                          paste(CandidateVariables, collapse=" + ")))
  
  # model.rf <- randomForest(Formula, data= data.train,importance = TRUE)
  model.rf <- randomForest(outcome_h~., data= data.train,importance = TRUE)
  
  data.train$p_prediction <- predict(model.rf, data.train, type="prob")[,2]
  data.test$p_prediction <- predict(model.rf, data.test, type="prob")[,2]
  
  pre_cutoff<-optimal.cutpoints(X = "p_prediction",status = "outcome_h", tag.healthy = 0, methods = "Youden",
                                data = data.train,ci.fit = TRUE,conf.level = 0.95)
  
  roc.train <- roc(data.train$outcome_h, data.train$p_prediction,levels = c(0,1), direction = "<")
  roc.test <- roc(data.test$outcome_h, data.test$p_prediction,levels = c(0,1), direction = "<")
  
  cm_train <- caret::confusionMatrix(as.factor(as.numeric(data.train$p_prediction>pre_cutoff$Youden$Global$optimal.cutoff$cutoff )), as.factor(data.train$outcome_h), positive = "1")
  cm_test <- caret::confusionMatrix(as.factor(as.numeric(data.test$p_prediction>pre_cutoff$Youden$Global$optimal.cutoff$cutoff)), as.factor(data.test$outcome_h), positive = "1") 
  
  c_train[i] <- roc.train$auc
  acc_train[i] <- cm_train$overall
  se_train[i] <- cm_train$byClass[1]
  spe_train[i] <- cm_train$byClass[2]
  ppv_train[i] <- cm_train$byClass[3]
  npv_train[i] <- cm_train$byClass[4]
  brier_train[i] <- mean((data.train$p_prediction-as.numeric(data.train$outcome_h))^2)
  cutoff[i] <- pre_cutoff$Youden$Global$optimal.cutoff$cutoff
  
  c_test[i] <- roc.test$auc
  acc_test[i] <- cm_test$overall
  se_test[i] <- cm_test$byClass[1]
  spe_test[i] <- cm_test$byClass[2]
  ppv_test[i] <- cm_test$byClass[3]
  npv_test[i] <- cm_test$byClass[4]
  brier_test[i] <- mean((data.test$p_prediction-as.numeric(data.test$outcome_h))^2)
  
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
data_gbm2 <- subset(data3,select = c(outcome_h,AF_category:Age,Weight:LAD,LVEF:UA,TT:CREA,Ccr:CK,NT_proBNP:Prior.CABG))
set.seed(1234)
index = sample(nrow(data_gbm2),round(0.7*nrow(data_gbm2)), )
data_gbm2_train = data_gbm2[index,]
data_gbm2_test = data_gbm2[-index,]

## hyper-parameter selection
data_gbm2_train$outcome_h <- ifelse(data_gbm2_train$outcome_h == "1","yes","no")
data_gbm2_test$outcome_h <- ifelse(data_gbm2_test$outcome_h == "1","yes","no")

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
train.gbm2 = train(
  outcome_h ~.,
  data = data_gbm2_train,
  trControl = cntrl,
  tuneGrid = grid,
  method = "gbm",
  metric = "ROC",
  # metric = "Kappa"
)

train.gbm2

## visualization
trellis.par.set(caretTheme())
plot(train.gbm2, metric = "ROC")

plot(train.gbm2, metric = "ROC", plotType = "level",
     scales = list(x = list(rot = 90)))

## feature importance
gbmImp2 <-varImp(train.gbm2,n.trees = 350, interaction.depth =5, shrinkage = 0.1,
                n.minobsinnode = 10)
plot(gbmImp2, top = 10)

## select the best hyper-parameter and train the model
data_gbm2 <- subset(data3,select = c(outcome_h,AF_category:Age,Weight:LAD,LVEF:UA,TT:CREA,Ccr:CK,NT_proBNP:Prior.CABG))
set.seed(444)
index = sample(nrow(data_gbm2),round(0.7*nrow(data_gbm2)), )
data_gbm2_train = data_gbm2[index,]
data_gbm2_test = data_gbm2[-index,]

## the function of gbm can not predict with propability, so chose the train function of caret package
grid = expand.grid(interaction.depth = 5, n.trees = 350, shrinkage = 0.1,n.minobsinnode = 10)
cntrl = trainControl(method = "cv",number = 5)

fit_gbm2 = train(
  outcome_h ~ NT_proBNP+UA+AST+ALB+DD+Age+GLU+TSH+HR+CK ,
  data = data_gbm2_train,trControl = cntrl,tuneGrid = grid,method = "gbm",metric = "Kappa")

# fit_gbm2 <- gbm(outcome_h ~.,data = data_gbm2_train,distribution = "adaboost",n.trees = 350,interaction.depth = 5,
#                 shrinkage = 0.1,n.minobsinnode = 10)  ## all variable

# fit_gbm2 <- gbm(outcome_h~NT_proBNP+UA+AST+ALB+DD+Age+GLU+TSH+HR+CK,data = data_gbm2_train,distribution = "adaboost",n.trees = 350,interaction.depth = 5,
#                 shrinkage = 0.1,n.minobsinnode = 10)  # top10

# fit_gbm2 <- gbm(outcome_h~NT_proBNP+UA+AST+ALB+DD+Age+GLU+TSH+HR+CK+BMI+Ccr+TT+PTINR+
#                   SBP+FBG+LVEF+LDH+LVEDD+LAD,data = data_gbm2_train,
#                 distribution = "adaboost",n.trees = 350,interaction.depth = 5,
#                 shrinkage = 0.1,n.minobsinnode = 10)  # top20

# fit_gbm2 <- gbm(outcome_h~NT_proBNP+UA+AST+ALB+DD+Age+GLU+TSH+HR+CK+BMI+Ccr+TT+PTINR+
#                   SBP+FBG+LVEF+LDH+LVEDD+LAD+Hyperlipidemia+AF_category+TnI+CREA+DBP+
#                   CHA2DS2_VACS+Gender+Antihypertensive.agents+CHD+Diuretics,
#                 data = data_gbm2_train,distribution = "adaboost",n.trees = 350,interaction.depth = 5,
#                 shrinkage = 0.1,n.minobsinnode = 10)  # top30


# training set
data_gbm2_train$pred <- predict(fit_gbm2, data_gbm2_train,type = "prob")[,2]
roc_gbm2_train <- roc(data_gbm2_train$outcome_h, data_gbm2_train$pred,ci=TRUE,print.auc = TRUE,levels = c(0,1), direction = "<")
roc_gbm2_train
# plot(roc_gbm2_train,ci=TRUE,print.auc = TRUE,lty=2)

pre_cutoff<-optimal.cutpoints(X = "pred",status = "outcome_h", tag.healthy = 0, methods = "Youden",
                              data = data_gbm2_train,ci.fit = TRUE,conf.level = 0.95)
summary(pre_cutoff)

caret::confusionMatrix(as.factor(as.numeric(data_gbm2_train$pred>0.7975908)), as.factor(data_gbm2_train$outcome_h), positive = "1")

# test set
data_gbm2_test$pred <- predict(fit_gbm2, newdata = data_gbm2_test,type = "prob")[,2]
roc_gbm2_test <- roc(data_gbm2_test$outcome_h, data_gbm2_test$pred,ci=TRUE,smooth=TRUE,print.auc = TRUE) #levels = c(0,1), direction = "<")
roc_gbm2_test
plot(roc_gbm2_test,ci=TRUE,print.auc = TRUE,lty=2)

caret::confusionMatrix(as.factor(as.numeric(data_gbm2_test$pred>0.7975908)), as.factor(data_gbm2_test$outcome_h), positive = "1") 



###############################################################################################################
## k-fold cross validation
data_gbm2 <- subset(data3,select = c(outcome_h,AF_category:Age,Weight:LAD,LVEF:UA,TT:CREA,Ccr:CK,NT_proBNP:Prior.CABG))
set.seed(1234)   
folds <- createMultiFolds(y=data_gbm2$outcome_h,k=5,times=10)   

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
  
  data.train <- data_gbm2[folds[[i]],]
  data.test <- data_gbm2[-folds[[i]],]
  
  grid = expand.grid(interaction.depth = 5, n.trees = 350, shrinkage = 0.1,n.minobsinnode = 10)
  
  cntrl = trainControl(method = "cv",number = 5)
  
  model.gbm = train(
    outcome_h ~ NT_proBNP+UA+AST+ALB+DD+Age+GLU+TSH+HR+CK ,
    data = data.train,trControl = cntrl,tuneGrid = grid,method = "gbm",metric = "Kappa")
  
  # Outcome <- "outcome_h"
  
  # CandidateVariables <- c("NT_proBNP","UA","AST","ALB", "DD","Age","GLU","TSH", "HR","CK")  # top10
  
  # CandidateVariables <- c("NT_proBNP","UA","AST","ALB", "DD","Age","GLU","TSH", "HR","CK",
  #                         "BMI","Ccr","TT","PTINR","SBP","FBG","LVEF","LDH","LVEDD","LAD")  # top20
  
  # CandidateVariables <- c("NT_proBNP","UA","AST","ALB", "DD","Age","GLU","TSH", "HR","CK",
  #                         "BMI","Ccr","TT","PTINR","SBP","FBG","LVEF","LDH","LVEDD","LAD",
  #                         "Hyperlipidemia","AF_category","TnI","CREA","DBP","CHA2DS2_VACS",
  #                         "Gender","Antihypertensive.agents","CHD","Diuretics")  # top30
  # 
  # Formula <- formula(paste(paste(Outcome,"~", collapse=" "),
  #                          paste(CandidateVariables, collapse=" + ")))
  
  # model.gbm <- gbm(Formula, data= data.train,distribution = "adaboost",n.trees = 350,interaction.depth = 5,shrinkage = 0.1,n.minobsinnode = 10)
  # model.gbm <- gbm(outcome_h~., data= data.train,distribution = "adaboost",n.trees = 350,interaction.depth = 5,shrinkage = 0.1,n.minobsinnode = 10)
  
  data.train$p_prediction <- predict(model.gbm, data.train, type="prob")[,2]
  data.test$p_prediction <- predict(model.gbm, data.test, type="prob")[,2]
  
  pre_cutoff<-optimal.cutpoints(X = "p_prediction",status = "outcome_h", tag.healthy = 0, methods = "Youden",
                                data = data.test,ci.fit = TRUE,conf.level = 0.95)
  
  roc.train <- roc(data.train$outcome_h, data.train$p_prediction,levels = c(0,1), direction = "<")
  roc.test <- roc(data.test$outcome_h, data.test$p_prediction,levels = c(0,1), direction = "<")
  
  cm_train <- caret::confusionMatrix(as.factor(as.numeric(data.train$p_prediction>pre_cutoff$Youden$Global$optimal.cutoff$cutoff )), as.factor(data.train$outcome_h), positive = "1")
  cm_test <- caret::confusionMatrix(as.factor(as.numeric(data.test$p_prediction>pre_cutoff$Youden$Global$optimal.cutoff$cutoff)), as.factor(data.test$outcome_h), positive = "1") 
  
  c_train[i] <- roc.train$auc
  acc_train[i] <- cm_train$overall
  se_train[i] <- cm_train$byClass[1]
  spe_train[i] <- cm_train$byClass[2]
  ppv_train[i] <- cm_train$byClass[3]
  npv_train[i] <- cm_train$byClass[4]
  brier_train[i] <- mean((data.train$p_prediction-as.numeric(data.train$outcome_h))^2)
  
  cutoff[i] <- pre_cutoff$Youden$Global$optimal.cutoff$cutoff
  c_test[i] <- roc.test$auc
  acc_test[i] <- cm_test$overall
  se_test[i] <- cm_test$byClass[1]
  spe_test[i] <- cm_test$byClass[2]
  ppv_test[i] <- cm_test$byClass[3]
  npv_test[i] <- cm_test$byClass[4]
  brier_test[i] <- mean((data.test$p_prediction-as.numeric(data.test$outcome_h))^2)
  
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
summary(data.xgb)

set.seed(1234)
index = sample(nrow(data.xgb),round(0.7*nrow(data.xgb)), )
data.train.xg = data.xgb[index,]
data.test.xg = data.xgb[-index,]

data.train.xg$outcome_h <- ifelse(data.train.xg$outcome_h == "1","yes","no")
data.test.xg$outcome_h <- ifelse(data.test.xg$outcome_h == "1","yes","no")

### hyper-parameter selection
grid = expand.grid(
  nrounds = c(75,100,200,500),
  colsample_bytree = c(0.5, 0.7, 1),
  min_child_weight = c(1, 3, 5),
  eta = c(0.001,0.005,0.01, 0.05, 0.1), #0.3 is default,
  gamma = c(1,0.5,0.1,0.01,0.005,0.001),
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
  # classProbs = TRUE,
  # summaryFunction = twoClassSummary
)

train.xgb = train(
  x = data.train.xg[,c(6:7,9:71)],
  y = as.factor(data.train.xg[, 3]),
  trControl = cntrl,
  tuneGrid = grid,
  method = "xgbTree",
  # metric = "ROC", 
  metric = "Kappa"
)

train.xgb



###############################################################################################################
## xgboost model (k-fold cross validation)
set.seed(1234)
folds <- createMultiFolds(y=data.xgb$outcome_h,k=5,times=20)   

c_train <- 0
acc_train <- 0
se_train <- 0
spe_train <- 0
ppv_train <- 0
npv_train <- 0
# brier_train <- 0

c_test <- 0
acc_test <- 0
se_test <- 0
spe_test <- 0
ppv_test <- 0
npv_test <- 0
# brier_test <- 0

for (i in 1:100){
  data.train.xg <- data.xgb[folds[[i]],]
  data.test.xg <- data.xgb[-folds[[i]],]
  
  ## data preprocessing
  
  # data.train.xg1<-data.matrix(data.train.xg[,c("Gender","Age", "BMI","AST","ALB", "NT_proBNP")])  ## stepwise feature
  
  # data.train.xg1<-data.matrix(data.train.xg[,c("NT_proBNP", "ALB","DD","AST","FBG","TT","HR","UA","PTINR","GLU")])  ##top10
  # 
  # data.train.xg1<-data.matrix(data.train.xg[,c("NT_proBNP", "ALB","DD","AST","FBG","TT","HR","UA","PTINR","GLU","CREA","Age", "CHA2DS2_VACS","TSH","Ever.smoking")])   ##top15
  
  # data.train.xg1<-data.matrix(data.train.xg[,c("NT_proBNP", "ALB","DD","AST","FBG","TT","HR","UA","PTINR","GLU","CREA","Age", "CHA2DS2_VACS","TSH","Ever.smoking",
  #                                           "LDH","Statins","BMI","Ccr","CK")])   ##top20
  
  # data.train.xg1<-data.matrix(data.train.xg[,c("NT_proBNP", "ALB","DD","AST","FBG","TT","HR","UA","PTINR","GLU","CREA","Age", "CHA2DS2_VACS","TSH","Ever.smoking",
  #                                              "LDH","Statins","BMI","Ccr","CK","Diuretics","LAD","DBP","LVESD","SBP","LVEF","LVEDD","Heparin","Hypertension","TnI")])   ###top30
  
  data.train.xg1<-data.matrix(data.train.xg[,c(6:7,10:23,27:71)]) ## all features
  
  data.train.xg2<-Matrix(data.train.xg1,sparse = T)
  data.train.xg_y<-data.train.xg$outcome_h
  traindata.xg<-list(data=data.train.xg2,label=data.train.xg_y)
  dtrain<-xgb.DMatrix(data = traindata.xg$data,label=traindata.xg$label)

  # data.test.xg1<-data.matrix(data.test.xg[,c("Gender","Age", "BMI","AST","ALB", "NT_proBNP")])  ## stepwise feature
  
  # data.test.xg1<-data.matrix(data.test.xg[,c("NT_proBNP", "ALB","DD","AST","FBG","TT","HR","UA","PTINR","GLU")])  ## top10
  # 
  # data.test.xg1<-data.matrix(data.test.xg[,c("NT_proBNP", "ALB","DD","AST","FBG","TT","HR","UA","PTINR","GLU","CREA","Age", "CHA2DS2_VACS","TSH","Ever.smoking")])   ## top15
  
  # data.test.xg1<-data.matrix(data.test.xg[,c("NT_proBNP", "ALB","DD","AST","FBG","TT","HR","UA","PTINR","GLU","CREA","Age", "CHA2DS2_VACS","TSH","Ever.smoking",
  #                                            "LDH","Statins","BMI","Ccr","CK")])  ## top20
  # 
  # data.test.xg1<-data.matrix(data.test.xg[,c("NT_proBNP", "ALB","DD","AST","FBG","TT","HR","UA","PTINR","GLU","CREA","Age", "CHA2DS2_VACS","TSH","Ever.smoking",
  #                                            "LDH","Statins","BMI","Ccr","CK","Diuretics","LAD","DBP","LVESD","SBP","LVEF","LVEDD","Heparin","Hypertension","TnI")])   ## top30
  
  data.test.xg1<-data.matrix(data.test.xg[,c(6:7,10:23,27:71)])   ## all features
  
  data.test.xg2<-Matrix(data.test.xg1,sparse = T)
  data.test.xg_y<-data.test.xg$outcome_h
  testdata.xg<-list(data=data.test.xg2,label=data.test.xg_y)
  dtest<-xgb.DMatrix(data = testdata.xg$data,label=testdata.xg$label)
  
  model_xgb_jiye <-xgboost(data = dtrain,objective="binary:logistic", booster="gbtree",nrounds = 300, max_depth = 3, eta = 0.01, gamma =
                             0.01, colsample_bytree = 0.5, min_child_weight = 1 , subsample = 0.7)
  
  # impor <- importance_jiye<-xgb.importance(data.train.xg2@Dimnames[[2]],model = model_xgb_jiye)
  # xgb.ggplot.importance(importance_jiye,rel_to_first = TRUE,n_clusters = 3, measure = "Cover")
  # xgb.ggplot.importance(importance_jiye,rel_to_first = TRUE,n_clusters = 3, measure = "Gain",top_n = 10)
  # xgb.ggplot.shap.summary(data = traindata.xg$data, model = model_xgb_jiye,target_class = 1,top_n = 10)
  
  ## predict
  pre.xgb.train <-predict(model_xgb_jiye,dtrain)
  pre.xgb.test <-predict(model_xgb_jiye,newdata = dtest)
  
  roc.xgb.train <- roc(response = data.train.xg_y, predictor = pre.xgb.train,ci=TRUE,print.auc = TRUE)
  roc.xgb.test <- roc(response = data.test.xg_y, predictor = pre.xgb.test,ci=TRUE,print.auc = TRUE)
  
  cm.xgb_train <- caret::confusionMatrix(as.factor(as.numeric(pre.xgb.train>0.034 )), as.factor(data.train.xg_y), positive = "1")
  cm.xgb_test <- caret::confusionMatrix(as.factor(as.numeric(pre.xgb.test>0.034)), as.factor(data.test.xg_y), positive = "1") 
  
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
# round(CI(brier_train,ci=0.95),3)

round(CI(c_test,ci=0.95),3)
round(CI(acc_test,ci=0.95),3)
round(CI(se_test,ci=0.95),3)
round(CI(spe_test,ci=0.95),3)
round(CI(ppv_test,ci=0.95),3)
round(CI(npv_test,ci=0.95),3)
# round(CI(brier_test,ci=0.95),3)



#######################################################################################################
## xgboost model with the top 10 features
set.seed(13)
index = sample(nrow(data.xgb),round(0.7*nrow(data.xgb)), )
data.train.xg = data.xgb[index,]
data.test.xg = data.xgb[-index,]

## data preprocessing
data.train.xg1<-data.matrix(data.train.xg[,c("NT_proBNP", "ALB","DD","AST","FBG","TT","HR","UA","PTINR","GLU")])
data.train.xg2<-Matrix(data.train.xg1,sparse = T)
data.train.xg_y<-data.train.xg$outcome_h
traindata.xg<-list(data=data.train.xg2,label=data.train.xg_y)
dtrain<-xgb.DMatrix(data = traindata.xg$data,label=traindata.xg$label)

data.test.xg1<-data.matrix(data.test.xg[,c("NT_proBNP", "ALB","DD","AST","FBG","TT","HR","UA","PTINR","GLU")])
data.test.xg2<-Matrix(data.test.xg1,sparse = T)
data.test.xg_y<-data.test.xg$outcome_h
testdata.xg<-list(data=data.test.xg2,label=data.test.xg_y)
dtest<-xgb.DMatrix(data = testdata.xg$data,label=testdata.xg$label)

## train the model
model_xgb5<-xgboost(data = dtrain,objective="binary:logistic", booster="gbtree",nrounds = 300, max_depth = 3, eta = 0.01, gamma = 0.01, colsample_bytree =
                      0.5, min_child_weight = 1 , subsample = 0.7)


importance5<-xgb.importance(data.train.xg2@Dimnames[[2]],model = model_xgb5)
# xgb.ggplot.importance(importance5,rel_to_first = TRUE)
# xgb.plot.importance(importance5,rel_to_first = TRUE,xlab = "Relative importance")

## predict
pre_xgb_train5<-predict(model_xgb5,dtrain)
pre_xgb_test5<-predict(model_xgb5,newdata = dtest)

## model evaluation
xgb.cf_train5<-caret::confusionMatrix(as.factor(as.numeric(pre_xgb_train5>0.08)),as.factor(data.train.xg_y), positive = "1")
xgb.cf_train5
roc_xgb_train5 <- roc(response = data.train.xg_y, predictor = pre_xgb_train5,ci=TRUE,print.auc = TRUE,levels = c(0,1), direction = "<")
roc_xgb_train5
# plot(roc_xgb_train5,ci=TRUE,print.auc = TRUE,lty=2)

xgb.cf_test5<-caret::confusionMatrix(as.factor(as.numeric(pre_xgb_test5>0.037)),as.factor(data.test.xg_y), positive = "1")
xgb.cf_test5
roc_xgb_test5 <- roc(response = data.test.xg_y, predictor = pre_xgb_test5,smooth=TRUE,ci=TRUE,print.auc = TRUE,levels = c(0,1), direction = "<")
roc_xgb_test5
# plot(roc_xgb_test5,ci=TRUE,print.auc = TRUE,lty=2)



## ROC curves of four models
par(mfrow=c(1,1))
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


