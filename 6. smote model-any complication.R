###########先进行数据均衡再拆分训练集和验证集（复合结局）
packageurl  <-  "https://cran.r-project.org/src/contrib/Archive/DMwR/DMwR_0.4.1.tar.gz"
install.packages("fastshap")
library(DMwR)
library(caret)
library(Rmisc)
library(MASS)
library(pROC)
library(Matrix)
library(grid)
library(xgboost)
library(e1071)
library(randomForest)
library(gbm)
library(terra)
library(modEvA)
library(OptimalCutpoints)
library(fastshap)
library(ggplot2)
library(kernlab)



setwd("/Users/weiyuna/Desktop/工作/合作项目/上海胸科/20211103导出数据/房颤/补充结局-不包含瓣膜手术")
# data<-read.csv("data_imputation_full.csv")
data<-subset(data,select = c(patient_id,outcome:Prior.CABG))

factor.name<-c("outcome","outcome_h","outcome_b", "AF_category", "Gender","LAD_factor","HF","UA_factor","CREA_factor","CK_factor","Aspirin","Clopidogrel","Other.antiplatelet.agents",
               "Antiplatelet.agents","Warfarin","Dabigatran","Rivaroxaban","Heparin", "Anticoagulants","Statins",
               "ACEI_ARB","β.blocker","Diuretics","CCB","Antihypertensive.agents","Ever.smoking","Ever.drinking","Angina","Heart.failure",
               "Stroke","PAD","COPD","Hypertension","Diabetes","Hyperlipidemia","MI","CHD", "CKD","Prior.RFA","Prior.PCI","Prior.CABG")

for (i in factor.name) {
  data[,i]<-as.factor(data[,i])
}

data_smo1_pre<-subset(data,select = c(outcome,AF_category:Age,Weight:LAD,LVEF:UA,TT:CREA,Ccr:CK,NT_proBNP:Prior.CABG))
# data_smo1<-SMOTE(outcome ~.,data=data_smo1_pre,perc.over = 5227,perc.under = 102,k=5,seed=1234)  ##  (1:1)
data_smo1<-SMOTE(outcome ~.,data=data_smo1_pre,perc.over = 964.5,perc.under = 552.3,k=5,seed=1234) ##  (1:5)
# write.csv(data_smo1,file = "data_smo1.csv")

### logistic regression
set.seed(1234)
index = sample(nrow(data_smo1),round(0.7*nrow(data_smo1)), )
data_smo1_train = data_smo1[index,]
data_smo1_test = data_smo1[-index,]

fit_smo1_log <- glm(outcome~.,data =data_smo1_train,family = binomial) # full model

## feature importance
logImp1 <- data.frame(varImp(fit_smo1_log,scale = FALSE))
head(logImp1)
logImp1 <- logImp1[order(logImp1$Overall,decreasing = TRUE),]
plot(logImp1)


# training set (0.928)
data_smo1_train$pred <- predict(fit_smo1_log, data_smo1_train,type="response")
roc_log1_train <- roc(data_smo1_train$outcome, data_smo1_train$pred,ci=TRUE,print.auc = TRUE) #levels = c(0,1), direction = "<")
roc_log1_train
plot(roc_log1_train,ci=TRUE,print.auc = TRUE,lty=2)

pre_cutoff<-optimal.cutpoints(X = "pred",status = "outcome", tag.healthy = 0, methods = "Youden",
                              data = data_smo1_train,ci.fit = TRUE,conf.level = 0.95)
summary(pre_cutoff)
caret::confusionMatrix(as.factor(as.numeric(data_smo1_train$pred>0.1583747 )), as.factor(data_smo1_train$outcome), positive = "1")

# test set (0.9178)
data_smo1_test$pred <- predict(fit_smo1_log, newdata = data_smo1_test, type="response")
roc_log1_test <- roc(data_smo1_test$outcome, data_smo1_test$pred,ci=TRUE,print.auc = TRUE) #levels = c(0,1), direction = "<")
roc_log1_test
plot(roc_log1_test,ci=TRUE,print.auc = TRUE,lty=2)
caret::confusionMatrix(as.factor(as.numeric(data_smo1_test$pred>0.1583747)), as.factor(data_smo1_test$outcome), positive = "1") 

### brier score (0.080)
brier <- function(df){
  n = nrow(df)
  df$delta_p_sq <- (df$pred-(as.numeric(df$outcome)-1))^2
  # brier <- (sum(df$delta_p_sq))/n
  brier <- mean(df$delta_p_sq)
  return(brier)
}  

brier(data_smo1_test)


### PR Curve
AUC(model = fit_smo1_log, obs = data_smo1_test$outcome, pred = data_smo1_test$pred, simplif = TRUE, interval = 0.005,
    FPR.limits = c(0, 1), curve = "PR", method = "trapezoid", plot = TRUE, diag = TRUE, 
    diag.col = "black", diag.lty = 2, curve.col = "steelblue", curve.lty = 1, curve.lwd = 2, plot.values = TRUE, 
    plot.digits = 3, plot.preds = FALSE, grid = FALSE,
    xlab = "Recall" , ylab = "Precision")


## calibration plot
data_log1_cal<-subset(data_smo1_test,select = c(outcome,pred))
summary(data_log1_cal)

data_log1_cal$outcome<-as.factor(data_log1_cal$outcome)
summary(data_log1_cal)

d1<-data_log1_cal[which(data_log1_cal$pred >=0 & data_log1_cal$pred <0.1),]
d2<-data_log1_cal[which(data_log1_cal$pred >=0.1 & data_log1_cal$pred <0.2),]
d3<-data_log1_cal[which(data_log1_cal$pred >=0.2 & data_log1_cal$pred <0.3),]
d4<-data_log1_cal[which(data_log1_cal$pred >=0.3 & data_log1_cal$pred <0.4),]
d5<-data_log1_cal[which(data_log1_cal$pred >=0.4 & data_log1_cal$pred <0.5),]
d6<-data_log1_cal[which(data_log1_cal$pred >=0.5 & data_log1_cal$pred <0.6),]
d7<-data_log1_cal[which(data_log1_cal$pred >=0.6 & data_log1_cal$pred <0.7),]
d8<-data_log1_cal[which(data_log1_cal$pred >=0.7 & data_log1_cal$pred <0.8),]
d9<-data_log1_cal[which(data_log1_cal$pred >=0.8 & data_log1_cal$pred <0.9),]
d10<-data_log1_cal[which(data_log1_cal$pred >=0.9 & data_log1_cal$pred <=1),]

t.test(d1$pred)
t.test(d2$pred)
t.test(d3$pred)
t.test(d4$pred)
t.test(d5$pred)
t.test(d6$pred)
t.test(d7$pred)
t.test(d8$pred)
t.test(d9$pred)
t.test(d10$pred)

summary(d1$outcome)

prop.test(nrow(d1[d1$outcome=="1",]),nrow(d1),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d2[d2$outcome=="1",]),nrow(d2),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d3[d3$outcome=="1",]),nrow(d3),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d4[d4$outcome=="1",]),nrow(d4),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d5[d5$outcome=="1",]),nrow(d5),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d6[d6$outcome=="1",]),nrow(d6),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d7[d7$outcome=="1",]),nrow(d7),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d8[d8$outcome=="1",]),nrow(d8),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d9[d9$outcome=="1",]),nrow(d9),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d10[d10$outcome=="1",]),nrow(d10),p = NULL, alternative = "two.sided",correct = TRUE)




###############################################################################################################
## logistic regression (k-fold cross validation)
set.seed(1234)   
folds <- createMultiFolds(y=data_smo1$outcome,k=5,times=10)   

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
  
  data.train <- data_smo1[folds[[i]],]
  data.test <- data_smo1[-folds[[i]],]

  Outcome <- "outcome"
  
  # CandidateVariables <- c("Statins","AF_category","GLU","UA","Prior.PCI","Hypertension","CHA2DS2_VACS","Weight","Gender","DD") # top 10
  
  # CandidateVariables <- c("Statins","AF_category","GLU","UA","Prior.PCI","Hypertension","CHA2DS2_VACS","Weight","Gender","DD",
  #                         "DBP","Age","Ever.drinking","Ccr","HR","Ever.smoking","Diabetes","β.blocker","CK","CCB") # top 20
  
  # CandidateVariables <- c("Statins","AF_category","GLU","UA","Prior.PCI","Hypertension","CHA2DS2_VACS","Weight","Gender","DD",
  #                         "DBP","Age","Ever.drinking","Ccr","HR","Ever.smoking","Diabetes","β.blocker","CK","CCB",
  #                         "CKD","LVESD","ALB","PAD","LAD","Antiplatelet.agents","LVEF","Stroke","SBP","LVEDD")  # top30
  
  CandidateVariables <- c("Statins","AF_category","GLU","UA","Prior.PCI","Hypertension","CHA2DS2_VACS","Weight","Gender","DD",
                          "DBP","Age","Ever.drinking","Ccr","HR","Ever.smoking","Diabetes","β.blocker","CK","CCB",
                          "CKD","LVESD","ALB","PAD","LAD","Antiplatelet.agents","LVEF","Stroke","SBP","LVEDD",
                          "ACEI_ARB","PTINR","AST","FBG","HAS_BLED","TT","CREA","Aspirin","TSH","Hyperlipidemia")  # top40
  
  Formula <- formula(paste(paste(Outcome,"~", collapse=" "),
                           paste(CandidateVariables, collapse=" + ")))

  model.log <- glm(Formula, data= data.train,family=binomial)
  # model.log <- glm(outcome~., data= data.train,family=binomial)
  
  data.train$p_prediction <- predict(model.log, data.train, type="response")
  data.test$p_prediction <- predict(model.log, data.test, type="response")
  pre_cutoff<-optimal.cutpoints(X = "p_prediction",status = "outcome", tag.healthy = 0, methods = "Youden",
                                data = data.train,ci.fit = TRUE,conf.level = 0.95)
  
  roc.train <- roc(data.train$outcome, data.train$p_prediction)
  roc.test <- roc(data.test$outcome, data.test$p_prediction)
  
  cm_train <- caret::confusionMatrix(as.factor(as.numeric(data.train$p_prediction>pre_cutoff$Youden$Global$optimal.cutoff$cutoff  )), as.factor(data.train$outcome), positive = "1")
  cm_test <- caret::confusionMatrix(as.factor(as.numeric(data.test$p_prediction>pre_cutoff$Youden$Global$optimal.cutoff$cutoff )), as.factor(data.test$outcome), positive = "1") 
  
  c_train[i] <- roc.train$auc
  acc_train[i] <- cm_train$overall
  se_train[i] <- cm_train$byClass[1]
  spe_train[i] <- cm_train$byClass[2]
  ppv_train[i] <- cm_train$byClass[3]
  npv_train[i] <- cm_train$byClass[4]
  cutoff[i] <- pre_cutoff$Youden$Global$optimal.cutoff$cutoff
  
  c_test[i] <- roc.test$auc
  acc_test[i] <- cm_test$overall
  se_test[i] <- cm_test$byClass[1]
  spe_test[i] <- cm_test$byClass[2]
  ppv_test[i] <- cm_test$byClass[3]
  npv_test[i] <- cm_test$byClass[4]
  
  brier_train[i] <- mean((data.train$p_prediction-(as.numeric(data.train$outcome)-1))^2)
  brier_test[i] <- mean((data.test$p_prediction-(as.numeric(data.test$outcome)-1))^2)
  
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
round(CI(cutoff,ci=0.95),3)




## svm model
set.seed(1234)
index = sample(nrow(data_smo1),round(0.7*nrow(data_smo1)), )
data_svm1_train = data_smo1[index,]
data_svm1_test = data_smo1[-index,]

## hyper-parameter selection for linear kernel

# data_svm1_train$outcome <- ifelse(data_svm1_train$outcome=="1","yes","no")
# data_svm1_test$outcome <- ifelse(data_svm1_test$outcome=="1","yes","no")
# data_svm1_train$outcome <- ifelse(data_svm1_train$outcome=="yes","1","0")
# data_svm1_test$outcome <- ifelse(data_svm1_test$outcome=="yes","1","0")

grid = expand.grid(
  # cost = c(0.001, 0.01, 0.1, 1,10)
  cost=0.01
)


cntrl = trainControl(
  method = "cv",
  number = 5
  # ,classProbs = TRUE,
  # summaryFunction = twoClassSummary
)

fit_smo1_svm = train(
  outcome ~ .,
  data = data_svm1_train,
  trControl = cntrl,
  tuneGrid = grid,
  method = "svmLinear2",
  # metric = "ROC"
  metric = "Kappa",
  probability = TRUE
)

## feature importance
svmImp1 <- varImp(fit_smo1_svm,scale = FALSE)
plot(svmImp1,top=40)
svmImp1 <- svmImp1$importance


# training set-full model (0.9224)
data_svm1_train$pred <- predict(fit_smo1_svm,data_svm1_train,type="prob")[,2]
roc_svm1_train <- roc(data_svm1_train$outcome, data_svm1_train$pred,ci=TRUE,print.auc = TRUE) #levels = c(0,1), direction = "<")
roc_svm1_train
plot(roc_svm1_train,ci=TRUE,print.auc = TRUE,lty=2)

pre_cutoff<-optimal.cutpoints(X = "pred",status = "outcome", tag.healthy = 0, methods = "Youden",
                              data = data_svm1_train,ci.fit = TRUE,conf.level = 0.95)
summary(pre_cutoff)
caret::confusionMatrix(as.factor(as.numeric(data_svm1_train$pred>0.1491121 )), as.factor(data_svm1_train$outcome), positive = "1")

# test set-full model (0.9082)
data_svm1_test$pred <- predict(fit_smo1_svm, newdata = data_svm1_test, type="prob")[,2]
roc_svm1_test <- roc(data_svm1_test$outcome, data_svm1_test$pred,ci=TRUE,print.auc = TRUE) #levels = c(0,1), direction = "<")
roc_svm1_test
plot(roc_svm1_test,ci=TRUE,print.auc = TRUE,lty=2)
caret::confusionMatrix(as.factor(as.numeric(data_svm1_test$pred>0.1491121)), as.factor(data_svm1_test$outcome), positive = "1") 

## brier score (0.07364839)
brier(data_svm1_test)


## calibration plot
data_svm1_cal<-subset(data_svm1_test,select = c(outcome,pred))
summary(data_svm1_cal)

data_svm1_cal$outcome<-as.factor(data_svm1_cal$outcome)
summary(data_svm1_cal)

d1<-data_svm1_cal[which(data_svm1_cal$pred >=0 & data_svm1_cal$pred <0.1),]
d2<-data_svm1_cal[which(data_svm1_cal$pred >=0.1 & data_svm1_cal$pred <0.2),]
d3<-data_svm1_cal[which(data_svm1_cal$pred >=0.2 & data_svm1_cal$pred <0.3),]
d4<-data_svm1_cal[which(data_svm1_cal$pred >=0.3 & data_svm1_cal$pred <0.4),]
d5<-data_svm1_cal[which(data_svm1_cal$pred >=0.4 & data_svm1_cal$pred <0.5),]
d6<-data_svm1_cal[which(data_svm1_cal$pred >=0.5 & data_svm1_cal$pred <0.6),]
d7<-data_svm1_cal[which(data_svm1_cal$pred >=0.6 & data_svm1_cal$pred <0.7),]
d8<-data_svm1_cal[which(data_svm1_cal$pred >=0.7 & data_svm1_cal$pred <0.8),]
d9<-data_svm1_cal[which(data_svm1_cal$pred >=0.8 & data_svm1_cal$pred <0.9),]
d10<-data_svm1_cal[which(data_svm1_cal$pred >=0.9 & data_svm1_cal$pred <=1),]

t.test(d1$pred)
t.test(d2$pred)
t.test(d3$pred)
t.test(d4$pred)
t.test(d5$pred)
t.test(d6$pred)
t.test(d7$pred)
t.test(d8$pred)
t.test(d9$pred)
t.test(d10$pred)

summary(d1$outcome)

prop.test(nrow(d1[d1$outcome=="1",]),nrow(d1),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d2[d2$outcome=="1",]),nrow(d2),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d3[d3$outcome=="1",]),nrow(d3),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d4[d4$outcome=="1",]),nrow(d4),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d5[d5$outcome=="1",]),nrow(d5),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d6[d6$outcome=="1",]),nrow(d6),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d7[d7$outcome=="1",]),nrow(d7),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d8[d8$outcome=="1",]),nrow(d8),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d9[d9$outcome=="1",]),nrow(d9),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d10[d10$outcome=="1",]),nrow(d10),p = NULL, alternative = "two.sided",correct = TRUE)



#### svm model k-fold cross validation
set.seed(1234)   
folds <- createMultiFolds(y=data_smo1$outcome,k=5,times=10)   

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
  
  data.train <- data_smo1[folds[[i]],]
  data.test <- data_smo1[-folds[[i]],]
  
  # Outcome <- "outcome"
  
  # CandidateVariables <- c("DBP","AST","TT","NT_proBNP","TnI","Statins","HR","Ccr","UA","BMI") # top 10
  
  # CandidateVariables <- c("DBP","AST","TT","NT_proBNP","TnI","Statins","HR","Ccr","UA","BMI",
  #                         "SBP","Age","AF_category","CREA","CHA2DS2_VACS","Stroke","CK","LDH","HAS_BLED","Weight") # top 20
  
  # CandidateVariables <- c("DBP","AST","TT","NT_proBNP","TnI","Statins","HR","Ccr","UA","BMI",
  #                         "SBP","Age","AF_category","CREA","CHA2DS2_VACS","Stroke","CK","LDH","HAS_BLED","Weight","Antiplatelet.agents","Clopidogrel",
  #                         "PAD","Heparin","Antihypertensive.agents","LVESD","Rivaroxaban","LVEDD",
  #                         "CHD","Hypertension")  # top30

  # CandidateVariables <- c("DBP","AST","TT","NT_proBNP","TnI","Statins","HR","Ccr","UA","BMI",
  #                         "SBP","Age","AF_category","CREA","CHA2DS2_VACS","Stroke","CK","LDH","HAS_BLED","Weight","Antiplatelet.agents","Clopidogrel",
  #                         "PAD","Heparin","Antihypertensive.agents","LVESD","Rivaroxaban","LVEDD",
  #                         "CHD","Hypertension","GLU","Warfarin","CKD","HF",
  #                         "Hyperlipidemia","Prior.PCI","CCB","Diuretics","PTINR","Prior.RFA")  # top40
  # 
  # Formula <- formula(paste(paste(Outcome,"~", collapse=" "),
  #                          paste(CandidateVariables, collapse=" + ")))
  
  grid = expand.grid(cost=0.01)
  
  cntrl = trainControl(method = "cv",number = 5)
  
  # model.svm = train(Formula,data = data.train,trControl = cntrl,tuneGrid = grid,method = "svmLinear2",
  #                   metric = "Kappa",probability = TRUE)
  
  model.svm = train(outcome~.,data = data.train,trControl = cntrl,tuneGrid = grid,method = "svmLinear2",
                    metric = "Kappa",probability = TRUE)  ## all feature
  
  data.train$p_prediction <- predict(model.svm, data.train, type="prob")[,2]
  data.test$p_prediction <- predict(model.svm, data.test, type="prob")[,2]
  pre_cutoff<-optimal.cutpoints(X = "p_prediction",status = "outcome", tag.healthy = 0, methods = "Youden",
                                data = data.train,ci.fit = TRUE,conf.level = 0.95)
  
  roc.train <- roc(data.train$outcome, data.train$p_prediction)
  roc.test <- roc(data.test$outcome, data.test$p_prediction)
  
  cm_train <- caret::confusionMatrix(as.factor(as.numeric(data.train$p_prediction>pre_cutoff$Youden$Global$optimal.cutoff$cutoff  )), as.factor(data.train$outcome), positive = "1")
  cm_test <- caret::confusionMatrix(as.factor(as.numeric(data.test$p_prediction>pre_cutoff$Youden$Global$optimal.cutoff$cutoff )), as.factor(data.test$outcome), positive = "1") 
  
  c_train[i] <- roc.train$auc
  acc_train[i] <- cm_train$overall
  se_train[i] <- cm_train$byClass[1]
  spe_train[i] <- cm_train$byClass[2]
  ppv_train[i] <- cm_train$byClass[3]
  npv_train[i] <- cm_train$byClass[4]
  cutoff[i] <- pre_cutoff$Youden$Global$optimal.cutoff$cutoff
  
  c_test[i] <- roc.test$auc
  acc_test[i] <- cm_test$overall
  se_test[i] <- cm_test$byClass[1]
  spe_test[i] <- cm_test$byClass[2]
  ppv_test[i] <- cm_test$byClass[3]
  npv_test[i] <- cm_test$byClass[4]
  
  brier_train[i] <- mean((data.train$p_prediction-(as.numeric(data.train$outcome)-1))^2)
  brier_test[i] <- mean((data.test$p_prediction-(as.numeric(data.test$outcome)-1))^2)
  
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





#### random forest model
set.seed(1234)
index = sample(nrow(data_smo1),round(0.7*nrow(data_smo1)), )
data_rf1_train = data_smo1[index,]
data_rf1_test = data_smo1[-index,]

# fit_smo1_rf <- randomForest(outcome~DD+HR+FBG+Ccr+NT_proBNP+Age+ALB+CK+TSH+CREA,data = data_rf1_train,importance = TRUE) # raw data feature selection top10

fit_smo1_rf <- randomForest(outcome~.,data = data_rf1_train,importance = TRUE)


## feature importance
importance_smo_rf1 <- data.frame(fit_smo1_rf$importance)
head(importance_smo_rf1)
importance_smo_rf1 <- importance_smo_rf1[order(importance_smo_rf1$MeanDecreaseAccuracy,decreasing = TRUE),]



# training set (1)
data_rf1_train$pred <- predict(fit_smo1_rf, data_rf1_train,type="prob")[,2]
roc_rf1_train <- roc(data_rf1_train$outcome, data_rf1_train$pred,ci=TRUE,print.auc = TRUE) #levels = c(0,1), direction = "<")
roc_rf1_train
# plot(roc_rf1_train,ci=TRUE,print.auc = TRUE,lty=2)
pre_cutoff<-optimal.cutpoints(X = "pred",status = "outcome", tag.healthy = 0, methods = "Youden",
                              data = data_rf1_train,ci.fit = TRUE,conf.level = 0.95)
summary(pre_cutoff)
caret::confusionMatrix(as.factor(as.numeric(data_rf1_train$pred>0.652)), as.factor(data_rf1_train$outcome), positive = "1")

# test set (0.9964)
data_rf1_test$pred <- predict(fit_smo1_rf, newdata = data_rf1_test,type="prob")[,2]
roc_rf1_test <- roc(data_rf1_test$outcome, as.numeric(data_rf1_test$pred),ci=TRUE,print.auc = TRUE) #levels = c(0,1), direction = "<")
roc_rf1_test
# plot(roc_rf1_test,ci=TRUE,print.auc = TRUE,lty=2)
caret::confusionMatrix(as.factor(as.numeric(data_rf1_test$pred>0.652)), as.factor(data_rf1_test$outcome), positive = "1") 

## brier score (0.03604603)
brier(data_rf1_test)


## calibration plot
data_rf1_cal<-subset(data_rf1_test,select = c(outcome,pred))
summary(data_rf1_cal)
data_rf1_cal$outcome<-as.factor(data_rf1_cal$outcome)

d1<-data_rf1_cal[which(data_rf1_cal$pred >=0 & data_rf1_cal$pred <0.1),]
d2<-data_rf1_cal[which(data_rf1_cal$pred >=0.1 & data_rf1_cal$pred <0.2),]
d3<-data_rf1_cal[which(data_rf1_cal$pred >=0.2 & data_rf1_cal$pred <0.3),]
d4<-data_rf1_cal[which(data_rf1_cal$pred >=0.3 & data_rf1_cal$pred <0.4),]
d5<-data_rf1_cal[which(data_rf1_cal$pred >=0.4 & data_rf1_cal$pred <0.5),]
d6<-data_rf1_cal[which(data_rf1_cal$pred >=0.5 & data_rf1_cal$pred <0.6),]
d7<-data_rf1_cal[which(data_rf1_cal$pred >=0.6 & data_rf1_cal$pred <0.7),]
d8<-data_rf1_cal[which(data_rf1_cal$pred >=0.7 & data_rf1_cal$pred <0.8),]
d9<-data_rf1_cal[which(data_rf1_cal$pred >=0.8 & data_rf1_cal$pred <0.9),]
d10<-data_rf1_cal[which(data_rf1_cal$pred >=0.9 & data_rf1_cal$pred <=1),]

t.test(d1$pred)
t.test(d2$pred)
t.test(d3$pred)
t.test(d4$pred)
t.test(d5$pred)
t.test(d6$pred)
t.test(d7$pred)
t.test(d8$pred)
t.test(d9$pred)
t.test(d10$pred)

summary(d1$outcome)

prop.test(nrow(d1[d1$outcome=="1",]),nrow(d1),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d2[d2$outcome=="1",]),nrow(d2),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d3[d3$outcome=="1",]),nrow(d3),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d4[d4$outcome=="1",]),nrow(d4),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d5[d5$outcome=="1",]),nrow(d5),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d6[d6$outcome=="1",]),nrow(d6),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d7[d7$outcome=="1",]),nrow(d7),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d8[d8$outcome=="1",]),nrow(d8),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d9[d9$outcome=="1",]),nrow(d9),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d10[d10$outcome=="1",]),nrow(d10),p = NULL, alternative = "two.sided",correct = TRUE)




###############################################################################################################
## random forest (k-fold cross validation)
set.seed(1234)   
folds <- createMultiFolds(y=data_smo1$outcome,k=5,times=10)   

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
  
  data.train <- data_smo1[folds[[i]],]
  data.test <- data_smo1[-folds[[i]],]
  
  # Outcome <- "outcome"
  
  # CandidateVariables <- c("TnI","CHA2DS2_VACS","HAS_BLED","Ccr", "CK","Age","Statins","NT_proBNP", "UA","CREA")  # top10
  
  # CandidateVariables <- c("TnI","CHA2DS2_VACS","HAS_BLED","Ccr", "CK","Age","Statins","NT_proBNP", "UA","CREA",
  #                         "HR","GLU","LVESD","DD","LVEDD","Weight","Prior.PCI","DBP","ALB","LVEF") # top20
  # 
  # CandidateVariables <- c("TnI","CHA2DS2_VACS","HAS_BLED","Ccr", "CK","Age","Statins","NT_proBNP", "UA","CREA",
  #                         "HR","GLU","LVESD","DD","LVEDD","Weight","Prior.PCI","DBP","ALB","LVEF",
  #                         "TT","BMI","SBP","LDH","LAD","AF_category","FBG","AST","TSH","PTINR")  # top30
  # 
  # CandidateVariables <- c("TnI","CHA2DS2_VACS","HAS_BLED","Ccr", "CK","Age","Statins","NT_proBNP", "UA","CREA",
  #                         "HR","GLU","LVESD","DD","LVEDD","Weight","Prior.PCI","DBP","ALB","LVEF",
  #                         "TT","BMI","SBP","LDH","LAD","AF_category","FBG","AST","TSH","PTINR",
  #                         "PAD","CCB","Antihypertensive.agents","Hypertension","CHD","CKD","Gender",
  #                         "β.blocker","Diuretics","Heparin")  # top40
  
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
  
  cutoff[i] <- pre_cutoff$Youden$Global$optimal.cutoff$cutoff
  c_test[i] <- roc.test$auc
  acc_test[i] <- cm_test$overall
  se_test[i] <- cm_test$byClass[1]
  spe_test[i] <- cm_test$byClass[2]
  ppv_test[i] <- cm_test$byClass[3]
  npv_test[i] <- cm_test$byClass[4]
  brier_train[i] <- mean((data.train$p_prediction-(as.numeric(data.train$outcome)-1))^2)
  brier_test[i] <- mean((data.test$p_prediction-(as.numeric(data.test$outcome)-1))^2)
  
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
round(CI(cutoff,ci=0.95),3)



############# gbm model
set.seed(1234)
index = sample(nrow(data_smo1),round(0.7*nrow(data_smo1)), )
data_gbm1_train = data_smo1[index,]
data_gbm1_test = data_smo1[-index,]

## hyper-parameter selection
data_gbm1_train$outcome <- ifelse(data_gbm1_train$outcome == "1","yes","no")
data_gbm1_test$outcome <- ifelse(data_gbm1_test$outcome == "1","yes","no")
data_gbm1_train$outcome <- ifelse(data_gbm1_train$outcome == "yes","1","0")
data_gbm1_test$outcome <- ifelse(data_gbm1_test$outcome == "yes","1","0")

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
  metric = "ROC"
  # metric = "Kappa"
)

train.gbm1

### model fit
grid = expand.grid(
  interaction.depth = 9, 
  n.trees = 500, 
  shrinkage = 0.2,
  n.minobsinnode = 20
)


cntrl = trainControl(
  method = "cv",
  number = 5
)

fit_gbm1 = train(
  outcome ~ .,
  data = data_gbm1_train,
  # trControl = cntrl,
  tuneGrid = grid,
  method = "gbm",
  # metric = "ROC",
  metric = "Kappa"
)

## feature importance
gbmImp1 <-varImp(fit_gbm1,scale = FALSE)
plot(gbmImp1, top = 40)
gbmImp1 <- gbmImp1$importance

# training set (1)
data_gbm1_train$pred <- predict(fit_gbm1, data_gbm1_train,type = "prob")[,2]
roc_gbm1_train <- roc(data_gbm1_train$outcome, data_gbm1_train$pred,ci=TRUE,print.auc = TRUE,levels = c(0,1), direction = "<")
roc_gbm1_train
# plot(roc_gbm1_train,ci=TRUE,print.auc = TRUE,lty=2)

pre_cutoff<-optimal.cutpoints(X = "pred",status = "outcome", tag.healthy = 0, methods = "Youden",
                              data = data_gbm1_train,ci.fit = TRUE,conf.level = 0.95)
summary(pre_cutoff)

caret::confusionMatrix(as.factor(as.numeric(data_gbm1_train$pred>0.9645113)), as.factor(data_gbm1_train$outcome), positive = "1")

# test set (0.997)
data_gbm1_test$pred <- predict(fit_gbm1, newdata = data_gbm1_test,type = "prob")[,2]
roc_gbm1_test <- roc(data_gbm1_test$outcome, data_gbm1_test$pred,ci=TRUE,print.auc = TRUE) #levels = c(0,1), direction = "<")
roc_gbm1_test
plot(roc_gbm1_test,ci=TRUE,print.auc = TRUE,lty=2)

caret::confusionMatrix(as.factor(as.numeric(data_gbm1_test$pred>0.9645113)), as.factor(data_gbm1_test$outcome), positive = "1") 

## brier score (0.01845647)
brier(data_gbm1_test)

## calibration plot
data_gbm1_cal<-subset(data_gbm1_test,select = c(outcome,pred))
summary(data_gbm1_cal)
data_gbm1_cal$outcome<-as.factor(data_gbm1_cal$outcome)

d1<-data_gbm1_cal[which(data_gbm1_cal$pred >=0 & data_gbm1_cal$pred <0.1),]
d2<-data_gbm1_cal[which(data_gbm1_cal$pred >=0.1 & data_gbm1_cal$pred <0.2),]
d3<-data_gbm1_cal[which(data_gbm1_cal$pred >=0.2 & data_gbm1_cal$pred <0.3),]
d4<-data_gbm1_cal[which(data_gbm1_cal$pred >=0.3 & data_gbm1_cal$pred <0.4),]
d5<-data_gbm1_cal[which(data_gbm1_cal$pred >=0.4 & data_gbm1_cal$pred <0.5),]
d6<-data_gbm1_cal[which(data_gbm1_cal$pred >=0.5 & data_gbm1_cal$pred <0.6),]
d7<-data_gbm1_cal[which(data_gbm1_cal$pred >=0.6 & data_gbm1_cal$pred <0.7),]
d8<-data_gbm1_cal[which(data_gbm1_cal$pred >=0.7 & data_gbm1_cal$pred <0.8),]
d9<-data_gbm1_cal[which(data_gbm1_cal$pred >=0.8 & data_gbm1_cal$pred <0.9),]
d10<-data_gbm1_cal[which(data_gbm1_cal$pred >=0.9 & data_gbm1_cal$pred <=1),]

t.test(d1$pred)
t.test(d2$pred)
t.test(d3$pred)
t.test(d4$pred)
t.test(d5$pred)
t.test(d6$pred)
t.test(d7$pred)
t.test(d8$pred)
t.test(d9$pred)
t.test(d10$pred)

summary(d1$outcome)

prop.test(nrow(d1[d1$outcome=="1",]),nrow(d1),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d2[d2$outcome=="1",]),nrow(d2),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d3[d3$outcome=="1",]),nrow(d3),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d4[d4$outcome=="1",]),nrow(d4),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d5[d5$outcome=="1",]),nrow(d5),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d6[d6$outcome=="1",]),nrow(d6),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d7[d7$outcome=="1",]),nrow(d7),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d8[d8$outcome=="1",]),nrow(d8),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d9[d9$outcome=="1",]),nrow(d9),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d10[d10$outcome=="1",]),nrow(d10),p = NULL, alternative = "two.sided",correct = TRUE)


###############################################################################################################
## gbm model (k-fold cross validation)
set.seed(1234)   
folds <- createMultiFolds(y=data_smo1$outcome,k=5,times=10)   

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
  
  data.train <- data_smo1[folds[[i]],]
  data.test <- data_smo1[-folds[[i]],]
  
  grid = expand.grid(
    interaction.depth = 9, 
    n.trees = 500, 
    shrinkage = 0.2,
    n.minobsinnode = 20
  )
  
  
  cntrl = trainControl(
    method = "cv",
    number = 5
  )
  
  model.gbm = train(
    # outcome ~TnI+CHA2DS2_VACS+CK+Ccr+Statins+ALB+HAS_BLED+GLU+LVEDD+HR, # top 10
    # outcome ~TnI+CHA2DS2_VACS+CK+Ccr+Statins+ALB+HAS_BLED+GLU+LVEDD+HR+UA+CREA+DD+Prior.PCI+Age+TT+LVEF+NT_proBNP+DBP+LAD, # top20
    # outcome ~TnI+CHA2DS2_VACS+CK+Ccr+Statins+ALB+HAS_BLED+GLU+LVEDD+HR+UA+CREA+DD+Prior.PCI+Age+TT+LVEF+NT_proBNP+DBP+LAD+
    #   BMI+LDH+AST+AF_category+Hypertension+LVESD+TSH+Weight+SBP+FBG, # top30
    # outcome ~TnI+CHA2DS2_VACS+CK+Ccr+Statins+ALB+HAS_BLED+GLU+LVEDD+HR+UA+CREA+DD+Prior.PCI+Age+TT+LVEF+NT_proBNP+DBP+LAD+
    #   BMI+LDH+AST+AF_category+Hypertension+LVESD+TSH+Weight+SBP+FBG+PTINR+β.blocker+CKD+Gender+Prior.RFA+Ever.drinking+
    #   Ever.smoking+PAD+Diabetes, # top 40
    outcome ~ .,
    data = data.train,
    trControl = cntrl,
    tuneGrid = grid,
    method = "gbm",
    # metric = "ROC",
    metric = "Kappa"
  )
  
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
  
  cutoff[i] <- pre_cutoff$Youden$Global$optimal.cutoff$cutoff
  c_test[i] <- roc.test$auc
  acc_test[i] <- cm_test$overall
  se_test[i] <- cm_test$byClass[1]
  spe_test[i] <- cm_test$byClass[2]
  ppv_test[i] <- cm_test$byClass[3]
  npv_test[i] <- cm_test$byClass[4]
  brier_train[i] <- mean((data.train$p_prediction-(as.numeric(data.train$outcome)-1))^2)
  brier_test[i] <- mean((data.test$p_prediction-(as.numeric(data.test$outcome)-1))^2)
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
round(CI(cutoff,ci=0.95),3)





### xgboost model
data_smo1_xgb<-read.csv("data_smo1.csv")
data_smo1_xgb<-subset(data_smo1_xgb,select = c(outcome:LAD,LVEF:UA,TT:CREA,Ccr:CK,NT_proBNP:Prior.CABG))
names(data_smo1_xgb)

data_smo1_xgb$Persistent_AF<-ifelse(data_smo1_xgb$AF_category=="2",1,0)
data_smo1_xgb$Chronic_AF<-ifelse(data_smo1_xgb$AF_category=="3",1,0)
names(data_smo1_xgb)
str(data_smo1_xgb)

### hyper-parameter selection
set.seed(1234)
index = sample(nrow(data_smo1_xgb),round(0.7*nrow(data_smo1_xgb)), )
data.train.xg = data_smo1_xgb[index,]
data.test.xg = data_smo1_xgb[-index,]
summary(data.train.xg)
summary(data.test.xg)

data.train.xg$outcome <- ifelse(data.train.xg$outcome == "1","yes","no")
data.test.xg$outcome <- ifelse(data.test.xg$outcome == "1","yes","no")

grid = expand.grid(
  nrounds = c(100,200,300),
  colsample_bytree = c(0.5, 0.8, 1),
  min_child_weight = c(1, 3, 5, 7),
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
train.xgb1 = train(
  x = data.train.xg[,c(3:64)],
  y = as.factor(data.train.xg[,1]),
  trControl = cntrl,
  tuneGrid = grid,
  method = "xgbTree",
  metric = "ROC"
  # metric = "Kappa"
)

train.xgb1

#######################################################################################################
### random split
set.seed(1234)
index = sample(nrow(data_smo1_xgb),round(0.7*nrow(data_smo1_xgb)), )
data.train.xg = data_smo1_xgb[index,]
data.test.xg = data_smo1_xgb[-index,]

## data preprocessing
data.train.xg1<-data.matrix(data.train.xg[,c(3:64)])
data.train.xg2<-Matrix(data.train.xg1,sparse = T)
data.train.xg_y1<-data.train.xg$outcome
traindata.xg<-list(data=data.train.xg2,label=data.train.xg_y1)
dtrain<-xgb.DMatrix(data = traindata.xg$data,label=traindata.xg$label)

data.test.xg1<-data.matrix(data.test.xg[,c(3:64)])
data.test.xg2<-Matrix(data.test.xg1,sparse = T)
data.test.xg_y1<-data.test.xg$outcome
testdata.xg<-list(data=data.test.xg2,label=data.test.xg_y1)
dtest<-xgb.DMatrix(data = testdata.xg$data,label=testdata.xg$label)

fit_xgb1 <-xgboost(data = dtrain, objective="binary:logistic", booster="gbtree",nrounds = 300, 
                         max_depth = 10, eta = 0.1, gamma = 0.1, colsample_bytree = 0.5, min_child_weight = 1, subsample = 0.5)

## feature importance
importance.xgb1<-xgb.importance(data.train.xg2@Dimnames[[2]],model = fit_xgb1)
# xgb.ggplot.importance(importance.xgb1,rel_to_first = TRUE,n_clusters = 3, measure = "Cover")
# xgb.ggplot.importance(importance.xgb1,rel_to_first = TRUE,n_clusters = 3, measure = "Gain")
xgb.plot.importance(importance.xgb1,rel_to_first = TRUE,xlab = "Relative importance")
# xgb.ggplot.shap.summary(data = traindata.xg$data, model = fit_xgb1,target_class = 1,top_n = 20)


## predict
pre.xgb.train1<-predict(fit_xgb1,dtrain)
pre.xgb.test1<-predict(fit_xgb1,newdata = dtest)

pre_cutoff<-optimal.cutpoints(X = "pre.xgb.train1",status = "data.train.xg_y1", tag.healthy = 0, methods = "Youden",
                              data = data.frame(cbind(pre.xgb.train1,data.train.xg_y1)),ci.fit = TRUE,conf.level = 0.95)
summary(pre_cutoff)

## model evaluation (1)
xgb.cf.train1<-caret::confusionMatrix(as.factor(as.numeric(pre.xgb.train1>0.7489846)),as.factor(data.train.xg_y1), positive = "1")
xgb.cf.train1
roc_xgb1_train <- roc(response = data.train.xg_y1, predictor = pre.xgb.train1,ci=TRUE,print.auc = TRUE)
roc_xgb1_train
# plot(roc_xgb1_train,ci=TRUE,print.auc = TRUE,lty=2)

## test set (0.9894)
xgb.cf.test1<-caret::confusionMatrix(as.factor(as.numeric(pre.xgb.test1>0.7489846)),as.factor(data.test.xg_y1), positive = "1")
xgb.cf.test1
roc_xgb1_test <- roc(response = data.test.xg_y1, predictor = pre.xgb.test1,ci=TRUE,print.auc = TRUE)
roc_xgb1_test
# plot(roc_xgb1_test,ci=TRUE,print.auc = TRUE,lty=2)

# brier score(0.006)
val_xgb1 <- val.prob(pre.xgb.test1,data.test.xg_y1,cex = 0.8)

## calibration plot
data_xgb1_cal<-data.frame(cbind(pre.xgb.test1,data.test.xg_y1))
summary(data_xgb1_cal)
data_xgb1_cal$data.test.xg_y1<-as.factor(data_xgb1_cal$data.test.xg_y1)

d1<-data_xgb1_cal[which(data_xgb1_cal$pre.xgb.test1 >=0 & data_xgb1_cal$pre.xgb.test1 <0.1),]
d2<-data_xgb1_cal[which(data_xgb1_cal$pre.xgb.test1 >=0.1 & data_xgb1_cal$pre.xgb.test1 <0.2),]
d3<-data_xgb1_cal[which(data_xgb1_cal$pre.xgb.test1 >=0.2 & data_xgb1_cal$pre.xgb.test1 <0.3),]
d4<-data_xgb1_cal[which(data_xgb1_cal$pre.xgb.test1 >=0.3 & data_xgb1_cal$pre.xgb.test1 <0.4),]
d5<-data_xgb1_cal[which(data_xgb1_cal$pre.xgb.test1 >=0.4 & data_xgb1_cal$pre.xgb.test1 <0.5),]
d6<-data_xgb1_cal[which(data_xgb1_cal$pre.xgb.test1 >=0.5 & data_xgb1_cal$pre.xgb.test1 <0.6),]
d7<-data_xgb1_cal[which(data_xgb1_cal$pre.xgb.test1 >=0.6 & data_xgb1_cal$pre.xgb.test1 <0.7),]
d8<-data_xgb1_cal[which(data_xgb1_cal$pre.xgb.test1 >=0.7 & data_xgb1_cal$pre.xgb.test1 <0.8),]
d9<-data_xgb1_cal[which(data_xgb1_cal$pre.xgb.test1 >=0.8 & data_xgb1_cal$pre.xgb.test1 <0.9),]
d10<-data_xgb1_cal[which(data_xgb1_cal$pre.xgb.test1 >=0.9 & data_xgb1_cal$pre.xgb.test1 <=1),]

t.test(d1$pre.xgb.test1)
t.test(d2$pre.xgb.test1)
t.test(d3$pre.xgb.test1)
t.test(d4$pre.xgb.test1)
t.test(d5$pre.xgb.test1)
t.test(d6$pre.xgb.test1)
t.test(d7$pre.xgb.test1)
t.test(d8$pre.xgb.test1)
t.test(d9$pre.xgb.test1)
t.test(d10$pre.xgb.test1)

summary(d1$data.test.xg_y1)

prop.test(nrow(d1[d1$data.test.xg_y1=="1",]),nrow(d1),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d2[d2$data.test.xg_y1=="1",]),nrow(d2),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d3[d3$data.test.xg_y1=="1",]),nrow(d3),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d4[d4$data.test.xg_y1=="1",]),nrow(d4),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d5[d5$data.test.xg_y1=="1",]),nrow(d5),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d6[d6$data.test.xg_y1=="1",]),nrow(d6),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d7[d7$data.test.xg_y1=="1",]),nrow(d7),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d8[d8$data.test.xg_y1=="1",]),nrow(d8),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d9[d9$data.test.xg_y1=="1",]),nrow(d9),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d10[d10$data.test.xg_y1=="1",]),nrow(d10),p = NULL, alternative = "two.sided",correct = TRUE)



###############################################################################################################
## xgboost model (k-fold cross validation)
set.seed(1234)
folds <- createMultiFolds(y=data_smo1_xgb$outcome,k=5,times=10)   

c_train <- 0
acc_train <- 0
se_train <- 0
spe_train <- 0
ppv_train <- 0
npv_train <- 0
brier_train <- 0

cutoff <- 0
c_test <- 0
acc_test <- 0
se_test <- 0
spe_test <- 0
ppv_test <- 0
npv_test <- 0
brier_test <- 0

for (i in 1:50){
  data.train.xg <- data_smo1_xgb[folds[[i]],]
  data.test.xg <- data_smo1_xgb[-folds[[i]],]
  
  ## data preprocessing
  # data.train.xg1<-data.matrix(data.train.xg[,c("CHA2DS2_VACS","TnI","HAS_BLED","Ccr","LVESD","CK","ALB","DD","HR","UA")])  ##top10
  
  # data.train.xg1<-data.matrix(data.train.xg[,c("CHA2DS2_VACS","TnI","HAS_BLED","Ccr","LVESD","CK","ALB","DD","HR","UA",
  #                                              "Age","CREA","AST","NT_proBNP","PTINR","TT","LDH","LVEDD","GLU","DBP")])  ##top20
  # 
  # data.train.xg1<-data.matrix(data.train.xg[,c("CHA2DS2_VACS","TnI","HAS_BLED","Ccr","LVESD","CK","ALB","DD","HR","UA",
  #                                              "Age","CREA","AST","NT_proBNP","PTINR","TT","LDH","LVEDD","GLU","DBP",
  #                                              "LVEF","FBG","SBP","TSH","Weight","LAD","BMI","Prior.PCI","Statins","Gender")])  ##top30
  # 
  # data.train.xg1<-data.matrix(data.train.xg[,c("CHA2DS2_VACS","TnI","HAS_BLED","Ccr","LVESD","CK","ALB","DD","HR","UA",
  #                                              "Age","CREA","AST","NT_proBNP","PTINR","TT","LDH","LVEDD","GLU","DBP",
  #                                              "LVEF","FBG","SBP","TSH","Weight","LAD","BMI","Prior.PCI","Statins","Gender",
  #                                              "Chronic_AF","CHD","PAD","Antihypertensive.agents","Hypertension","CCB","CKD",
  #                                              "Rivaroxaban","ACEI_ARB","β.blocker")])  ##top40
  
  data.train.xg1<-data.matrix(data.train.xg[,c(3:64)]) # all feature
  
  data.train.xg2<-Matrix(data.train.xg1,sparse = T)
  data.train.xg_y<-data.train.xg$outcome
  traindata.xg<-list(data=data.train.xg2,label=data.train.xg_y)
  dtrain<-xgb.DMatrix(data = traindata.xg$data,label=traindata.xg$label)
  
  # data.test.xg1<-data.matrix(data.test.xg[,c("CHA2DS2_VACS","TnI","HAS_BLED","Ccr","LVESD","CK","ALB","DD","HR","UA")])  ##top10
  
  # data.test.xg1<-data.matrix(data.test.xg[,c("CHA2DS2_VACS","TnI","HAS_BLED","Ccr","LVESD","CK","ALB","DD","HR","UA",
  #                                            "Age","CREA","AST","NT_proBNP","PTINR","TT","LDH","LVEDD","GLU","DBP")])  ##top20
  # 
  # data.test.xg1<-data.matrix(data.test.xg[,c("CHA2DS2_VACS","TnI","HAS_BLED","Ccr","LVESD","CK","ALB","DD","HR","UA",
  #                                            "Age","CREA","AST","NT_proBNP","PTINR","TT","LDH","LVEDD","GLU","DBP",
  #                                            "LVEF","FBG","SBP","TSH","Weight","LAD","BMI","Prior.PCI","Statins","Gender")])  ##top30
  # 
  # data.test.xg1<-data.matrix(data.test.xg[,c("CHA2DS2_VACS","TnI","HAS_BLED","Ccr","LVESD","CK","ALB","DD","HR","UA",
  #                                            "Age","CREA","AST","NT_proBNP","PTINR","TT","LDH","LVEDD","GLU","DBP",
  #                                            "LVEF","FBG","SBP","TSH","Weight","LAD","BMI","Prior.PCI","Statins","Gender",
  #                                            "Chronic_AF","CHD","PAD","Antihypertensive.agents","Hypertension","CCB","CKD",
  #                                            "Rivaroxaban","ACEI_ARB","β.blocker")])  ##top40
  
  data.test.xg1<-data.matrix(data.test.xg[,c(3:64)]) # all feature
  
  data.test.xg2<-Matrix(data.test.xg1,sparse = T)
  data.test.xg_y<-data.test.xg$outcome
  testdata.xg<-list(data=data.test.xg2,label=data.test.xg_y)
  dtest<-xgb.DMatrix(data = testdata.xg$data,label=testdata.xg$label)
  
  model_xgb_fuhe <-xgboost(data = dtrain,objective="binary:logistic", booster="gbtree",nrounds = 300, max_depth = 10, eta = 0.1, gamma =
                             0.1, colsample_bytree = 0.5, min_child_weight = 1 , subsample = 0.5)
  
  # importance_fuhe<-xgb.importance(data.train.xg2@Dimnames[[2]],model = model_xgb_fuhe)
  # xgb.ggplot.importance(importance_fuhe,rel_to_first = TRUE,n_clusters = 3, measure = "Cover")
  # xgb.ggplot.importance(importance_fuhe,rel_to_first = TRUE,n_clusters = 3, measure = "Gain",top_n = 20)
  # xgb.ggplot.shap.summary(data = traindata.xg$data, model = model_xgb_fuhe,target_class = 1,top_n = 20)
  
  ## predict
  pre.xgb.train <-predict(model_xgb_fuhe,dtrain)
  pre.xgb.test <-predict(model_xgb_fuhe,newdata = dtest)
  
  pre_cutoff<-optimal.cutpoints(X = "pre.xgb.train",status = "data.train.xg_y", tag.healthy = 0, methods = "Youden",
                                data = data.frame(cbind(pre.xgb.train,data.train.xg_y)),ci.fit = TRUE,conf.level = 0.95)
  
  roc.xgb.train <- roc(response = data.train.xg_y, predictor = pre.xgb.train,ci=TRUE,print.auc = TRUE)
  roc.xgb.test <- roc(response = data.test.xg_y, predictor = pre.xgb.test,ci=TRUE,print.auc = TRUE)
  
  cm.xgb_train <- caret::confusionMatrix(as.factor(as.numeric(pre.xgb.train>pre_cutoff$Youden$Global$optimal.cutoff$cutoff )), as.factor(data.train.xg_y), positive = "1")
  cm.xgb_test <- caret::confusionMatrix(as.factor(as.numeric(pre.xgb.test>pre_cutoff$Youden$Global$optimal.cutoff$cutoff)), as.factor(data.test.xg_y), positive = "1") 
  
  c_train[i] <- roc.xgb.train$auc
  acc_train[i] <- cm.xgb_train$overall
  se_train[i] <- cm.xgb_train$byClass[1]
  spe_train[i] <- cm.xgb_train$byClass[2]
  ppv_train[i] <- cm.xgb_train$byClass[3]
  npv_train[i] <- cm.xgb_train$byClass[4]
  
  cutoff[i] <- pre_cutoff$Youden$Global$optimal.cutoff$cutoff
  c_test[i] <- roc.xgb.test$auc
  acc_test[i] <- cm.xgb_test$overall
  se_test[i] <- cm.xgb_test$byClass[1]
  spe_test[i] <- cm.xgb_test$byClass[2]
  ppv_test[i] <- cm.xgb_test$byClass[3]
  npv_test[i] <- cm.xgb_test$byClass[4]
  
  brier_train[i] <- mean((pre.xgb.train-as.numeric(data.train.xg_y))^2)
  brier_test[i] <- mean((pre.xgb.test-as.numeric(data.test.xg_y))^2)
  
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
round(CI(cutoff,ci=0.95),3)



## ROC curves of four models
par(mfrow=c(1,1))
plot(roc_log1_test,ci=TRUE,lty=2,col="#1B1919FF",legacy.axes=T)
plot(roc_svm1_test,add=TRUE,ci=TRUE,lty=2,col="#3B4992FF",legacy.axes=T)
plot(roc_rf1_test,add=TRUE,ci=TRUE,lty=2,col="#008280FF")
plot(roc_gbm1_test,add=TRUE,ci=TRUE,lty=2,col="#631879FF")
plot(roc_xgb1_test,add=TRUE,ci=TRUE,lty=2,col="#A20056FF")
abline(h = seq(0,1,0.1), v = seq(0,1,0.1), col = "lightgrey", lty = 1,lwd = 0.4)
legend("bottomright",legend = c("Any complication AUC (95% CI)",
                                "LR: 0.917 (0.913-0.921)",
                                "SVM: 0.908 (0.904-0.913)",
                                "RF: 0.996 (0.996-0.997)",
                                "GBM: 0.995 (0.994-0.996)",
                                "XGBoost: 0.992 (0.991-0.993)"),
       col = c("white","#1B1919FF","#3B4992FF","#008280FF","#631879FF","#A20056FF"),lty = 2,lwd = 1,cex = 0.8, text.font = 2,box.lwd = "none")
# legend("topright",legend = "A", text.font = 2,box.lwd = "none")

