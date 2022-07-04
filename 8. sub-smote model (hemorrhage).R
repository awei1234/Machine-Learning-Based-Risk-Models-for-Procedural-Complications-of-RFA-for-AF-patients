setwd("/Users/weiyuna/Desktop/工作/合作项目/上海胸科/20211103导出数据/房颤/补充结局-不包含瓣膜手术")
install.packages("devtools")
library(devtools)
install_github("cran/DMwR",ref="master")
install.packages("caret")
install.packages("Rmisc")
install.packages("MASS")
install.packages("pROC")
install.packages("Matrix")
install.packages("grid")
install.packages("xgboost")
install.packages("e1071")
install.packages("gbm")
install.packages("terra")
install.packages("modEvA")
install.packages("OptimalCutpoints")
install.packages("randomForest")

library(DMwR)
library(caret)
library(Rmisc)
library(MASS)
library(pROC)
library(Matrix)
library(grid)
library(xgboost)
library(e1071)
library(gbm)
library(terra)
library(modEvA)
library(OptimalCutpoints)
library(randomForest)

data<-read.csv("data_imputation_full.csv")
data<-subset(data,select = c(patient_id,outcome:Prior.CABG))

factor.name<-c("outcome","outcome_h","outcome_b", "AF_category", "Gender","LAD_factor","HF","UA_factor","CREA_factor","CK_factor","Aspirin","Clopidogrel","Other.antiplatelet.agents",
               "Antiplatelet.agents","Warfarin","Dabigatran","Rivaroxaban","Heparin", "Anticoagulants","Statins",
               "ACEI_ARB","β.blocker","Diuretics","CCB","Antihypertensive.agents","Ever.smoking","Ever.drinking","Angina","Heart.failure",
               "Stroke","PAD","COPD","Hypertension","Diabetes","Hyperlipidemia","MI","CHD", "CKD","Prior.RFA","Prior.PCI","Prior.CABG")

for (i in factor.name) {
  data[,i]<-as.factor(data[,i])
}

data_smo3_pre<-subset(data,select = c(outcome_b,AF_category:Age,Weight:LAD,LVEF:UA,TT:CREA,Ccr:CK,NT_proBNP:Prior.CABG))
data_smo3<-SMOTE(outcome_b ~.,data=data_smo3_pre,perc.over = 2374,perc.under = 520.7,k=5,seed=1234) #### (1:5)
# write.csv(data_smo3,file = "data_smo3.csv")


## logistic regression
set.seed(1234)
index = sample(nrow(data_smo3),round(0.7*nrow(data_smo3)), )
data_smo3_train = data_smo3[index,]
data_smo3_test = data_smo3[-index,]

fit_smo3_log <- glm(outcome_b ~ ., data=data_smo3_train,family=binomial)


## feature importance
logImp3 <- varImp(fit_smo3_log,scale = FALSE)
plot(logImp3,top=40)

# training set (0.9829)
data_smo3_train$pred <- predict(fit_smo3_log, data_smo3_train,type="response")
roc_log3_train <- roc(data_smo3_train$outcome_b, data_smo3_train$pred,ci=TRUE,print.auc = TRUE) #levels = c(0,1), direction = "<")
roc_log3_train
# plot(roc_log3_train,ci=TRUE,print.auc = TRUE,lty=2)

pre_cutoff<-optimal.cutpoints(X = "pred",status = "outcome_b", tag.healthy = 0, methods = "Youden",
                              data = data_smo3_train,ci.fit = TRUE,conf.level = 0.95)
summary(pre_cutoff)

caret::confusionMatrix(as.factor(as.numeric(data_smo3_train$pred>0.25212363 )), as.factor(data_smo3_train$outcome_b), positive = "1")

# test set (0.9674)
data_smo3_test$pred <- predict(fit_smo3_log, newdata = data_smo3_test, type="response")
roc_log3_test <- roc(data_smo3_test$outcome_b, data_smo3_test$pred,ci=TRUE,print.auc = TRUE) #levels = c(0,1), direction = "<")
roc_log3_test
plot(roc_log3_test,ci=TRUE,print.auc = TRUE,lty=2)
caret::confusionMatrix(as.factor(as.numeric(data_smo3_test$pred>0.25212363)), as.factor(data_smo3_test$outcome_b), positive = "1") 

### brier score
brier(data_smo3_test)

### PR Curve
AUC(model = fit_smo3_log, obs = data_smo3_test$outcome_b, pred = data_smo3_test$pred, simplif = TRUE, interval = 0.005,
    FPR.limits = c(0, 1), curve = "PR", method = "trapezoid", plot = TRUE, diag = TRUE, 
    diag.col = "black", diag.lty = 2, curve.col = "blue", curve.lty = 1, curve.lwd = 2, plot.values = TRUE, 
    plot.digits = 3, plot.preds = FALSE, grid = FALSE,
    xlab = "Recall" , ylab = "Precision")


## calibration plot
data_log3_cal<-subset(data_smo3_test,select = c(outcome_b,pred))
data_log3_cal$outcome_b<-as.factor(data_log3_cal$outcome_b)
summary(data_log3_cal)

d1<-data_log3_cal[which(data_log3_cal$pred >=0 & data_log3_cal$pred <0.1),]
d2<-data_log3_cal[which(data_log3_cal$pred >=0.1 & data_log3_cal$pred <0.2),]
d3<-data_log3_cal[which(data_log3_cal$pred >=0.2 & data_log3_cal$pred <0.3),]
d4<-data_log3_cal[which(data_log3_cal$pred >=0.3 & data_log3_cal$pred <0.4),]
d5<-data_log3_cal[which(data_log3_cal$pred >=0.4 & data_log3_cal$pred <0.5),]
d6<-data_log3_cal[which(data_log3_cal$pred >=0.5 & data_log3_cal$pred <0.6),]
d7<-data_log3_cal[which(data_log3_cal$pred >=0.6 & data_log3_cal$pred <0.7),]
d8<-data_log3_cal[which(data_log3_cal$pred >=0.7 & data_log3_cal$pred <0.8),]
d9<-data_log3_cal[which(data_log3_cal$pred >=0.8 & data_log3_cal$pred <0.9),]
d10<-data_log3_cal[which(data_log3_cal$pred >=0.9 & data_log3_cal$pred <=1),]

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

summary(d8$outcome_b)

prop.test(nrow(d1[d1$outcome_b=="1",]),nrow(d1),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d2[d2$outcome_b=="1",]),nrow(d2),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d3[d3$outcome_b=="1",]),nrow(d3),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d4[d4$outcome_b=="1",]),nrow(d4),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d5[d5$outcome_b=="1",]),nrow(d5),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d6[d6$outcome_b=="1",]),nrow(d6),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d7[d7$outcome_b=="1",]),nrow(d7),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d8[d8$outcome_b=="1",]),nrow(d8),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d9[d9$outcome_b=="1",]),nrow(d9),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d10[d10$outcome_b=="1",]),nrow(d10),p = NULL, alternative = "two.sided",correct = TRUE)





###############################################################################################################
## k-fold cross validation
set.seed(1234)   
folds <- createMultiFolds(y=data_smo3$outcome_b,k=5,times=10)   

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
  
  data.train <- data_smo3[folds[[i]],]
  data.test <- data_smo3[-folds[[i]],]
  
  # Outcome <- "outcome_b"
  
  # CandidateVariables <- c("DD","DBP","AF_category","UA","Statins","TnI","β.blocker","Age","CKD","Weight","Ccr") ## top10
  
  # CandidateVariables <- c("DD","DBP","AF_category","UA","Statins","TnI","β.blocker","Age","CKD","Weight","Ccr",
  #                         "CREA","ACEI_ARB","BMI","Heparin","SBP","PAD","ALB","LAD","CHD","AST") # top20
  
  # CandidateVariables <- c("DD","DBP","AF_category","UA","Statins","TnI","β.blocker","Age","CKD","Weight","Ccr",
  #                         "CREA","ACEI_ARB","BMI","Heparin","SBP","PAD","ALB","LAD","CHD","AST",
  #                         "Aspirin","Prior.PCI","LVEDD","CK","Clopidogrel","LVESD","Antihypertensive.agents","CHA2DS2_VACS","NT_proBNP","Gender") # top30
  
  # CandidateVariables <- c("DD","DBP","AF_category","UA","Statins","TnI","β.blocker","Age","CKD","Weight","Ccr",
  #                         "CREA","ACEI_ARB","BMI","Heparin","SBP","PAD","ALB","LAD","CHD","AST",
  #                         "Aspirin","Prior.PCI","LVEDD","CK","Clopidogrel","LVESD","Antihypertensive.agents","CHA2DS2_VACS","NT_proBNP","Gender",
  #                         "Antiplatelet.agents","Diabetes","Anticoagulants","HR","TT","HAS_BLED","GLU","LDH","CCB","Stroke") # top40

  # Formula <- formula(paste(paste(Outcome,"~", collapse=" "),
  #                          paste(CandidateVariables, collapse=" + ")))
  
  # model.log <- glm(Formula, data= data.train,family=binomial)
  model.log <- glm(outcome_b ~., data= data.train,family=binomial)
  
  data.train$p_prediction <- predict(model.log, data.train, type="response")
  data.test$p_prediction <- predict(model.log, data.test, type="response")
  
  pre_cutoff<-optimal.cutpoints(X = "p_prediction",status = "outcome_b", tag.healthy = 0, methods = "Youden",
                                data = data.train,ci.fit = TRUE,conf.level = 0.95)
  
  roc.train <- roc(data.train$outcome_b, data.train$p_prediction)
  roc.test <- roc(data.test$outcome_b, data.test$p_prediction)
  
  cm_train <- caret::confusionMatrix(as.factor(as.numeric(data.train$p_prediction>pre_cutoff$Youden$Global$optimal.cutoff$cutoff)), as.factor(data.train$outcome_b), positive = "1")
  cm_test <- caret::confusionMatrix(as.factor(as.numeric(data.test$p_prediction>pre_cutoff$Youden$Global$optimal.cutoff$cutoff)), as.factor(data.test$outcome_b), positive = "1") 
  
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
  
  brier_train[i] <- mean((data.train$p_prediction-(as.numeric(data.train$outcome_b)-1))^2)
  brier_test[i] <- mean((data.test$p_prediction-(as.numeric(data.test$outcome_b)-1))^2)
  
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
index = sample(nrow(data_smo3),round(0.7*nrow(data_smo3)), )
data_svm3_train = data_smo3[index,]
data_svm3_test = data_smo3[-index,]

## hyper-parameter selection

# data_svm3_train$outcome_b <- ifelse(data_svm3_train$outcome_b=="1","yes","no")
# data_svm3_test$outcome_b <- ifelse(data_svm3_test$outcome_b=="1","yes","no")
# data_svm3_train$outcome_b <- ifelse(data_svm3_train$outcome_b=="yes","1","0")
# data_svm3_test$outcome_b <- ifelse(data_svm3_test$outcome_b=="yes","1","0")

grid = expand.grid(
  # cost = c(0.001, 0.01, 0.1, 1,10)
  cost=1
)


cntrl = trainControl(
  method = "cv",
  number = 5
  # ,classProbs = TRUE,
  # summaryFunction = twoClassSummary
)

fit_smo3_svm = train(
  outcome_b ~ .,
  data = data_svm3_train,
  trControl = cntrl,
  tuneGrid = grid,
  method = "svmLinear2",
  # metric = "ROC"
  metric = "Kappa",
  probability = TRUE
)

## feature importance
svmImp3 <- varImp(fit_smo3_svm,scale = FALSE)
plot(svmImp,top=40)
svmImp3 <- svmImp3$importance

# training set-full model (0.9794)
data_svm3_train$pred <- predict(fit_smo3_svm,data_svm3_train,type="prob")[,2]
roc_svm3_train <- roc(data_svm3_train$outcome_b, data_svm3_train$pred,ci=TRUE,print.auc = TRUE) #levels = c(0,1), direction = "<")
roc_svm3_train
plot(roc_svm3_train,ci=TRUE,print.auc = TRUE,lty=2)

pre_cutoff<-optimal.cutpoints(X = "pred",status = "outcome_b", tag.healthy = 0, methods = "Youden",
                              data = data_svm3_train,ci.fit = TRUE,conf.level = 0.95)
summary(pre_cutoff)
caret::confusionMatrix(as.factor(as.numeric(data_svm3_train$pred>0.19812603 )), as.factor(data_svm3_train$outcome_b), positive = "1")

# test set-full model (0.9619)
data_svm3_test$pred <- predict(fit_smo3_svm, newdata = data_svm3_test, type="prob")[,2]
roc_svm3_test <- roc(data_svm3_test$outcome_b, data_svm3_test$pred,ci=TRUE,print.auc = TRUE) #levels = c(0,1), direction = "<")
roc_svm3_test
plot(roc_svm3_test,ci=TRUE,print.auc = TRUE,lty=2)
caret::confusionMatrix(as.factor(as.numeric(data_svm3_test$pred>0.19812603)), as.factor(data_svm3_test$outcome_b), positive = "1") 

## brier score ()
brier(data_svm3_test)

## calibration plot
data_svm3_cal<-subset(data_svm3_test,select = c(outcome_b,pred))
summary(data_svm3_cal)

data_svm3_cal$outcome_b<-as.factor(data_svm3_cal$outcome_b)
summary(data_svm3_cal)

d1<-data_svm3_cal[which(data_svm3_cal$pred >=0 & data_svm3_cal$pred <0.1),]
d2<-data_svm3_cal[which(data_svm3_cal$pred >=0.1 & data_svm3_cal$pred <0.2),]
d3<-data_svm3_cal[which(data_svm3_cal$pred >=0.2 & data_svm3_cal$pred <0.3),]
d4<-data_svm3_cal[which(data_svm3_cal$pred >=0.3 & data_svm3_cal$pred <0.4),]
d5<-data_svm3_cal[which(data_svm3_cal$pred >=0.4 & data_svm3_cal$pred <0.5),]
d6<-data_svm3_cal[which(data_svm3_cal$pred >=0.5 & data_svm3_cal$pred <0.6),]
d7<-data_svm3_cal[which(data_svm3_cal$pred >=0.6 & data_svm3_cal$pred <0.7),]
d8<-data_svm3_cal[which(data_svm3_cal$pred >=0.7 & data_svm3_cal$pred <0.8),]
d9<-data_svm3_cal[which(data_svm3_cal$pred >=0.8 & data_svm3_cal$pred <0.9),]
d10<-data_svm3_cal[which(data_svm3_cal$pred >=0.9 & data_svm3_cal$pred <=1),]

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

summary(d1$outcome_b)

prop.test(nrow(d1[d1$outcome_b=="1",]),nrow(d1),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d2[d2$outcome_b=="1",]),nrow(d2),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d3[d3$outcome_b=="1",]),nrow(d3),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d4[d4$outcome_b=="1",]),nrow(d4),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d5[d5$outcome_b=="1",]),nrow(d5),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d6[d6$outcome_b=="1",]),nrow(d6),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d7[d7$outcome_b=="1",]),nrow(d7),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d8[d8$outcome_b=="1",]),nrow(d8),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d9[d9$outcome_b=="1",]),nrow(d9),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d10[d10$outcome_b=="1",]),nrow(d10),p = NULL, alternative = "two.sided",correct = TRUE)


#### svm model k-fold cross validation
set.seed(1234)   
folds <- createMultiFolds(y=data_smo3$outcome_b,k=5,times=2)   

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


for (i in 1:10){
  
  data.train <- data_smo3[folds[[i]],]
  data.test <- data_smo3[-folds[[i]],]
  
  # Outcome <- "outcome_b"
  
  # CandidateVariables <- c("DD","DBP","NT_proBNP","TnI","LVEDD","UA","CHA2DS2_VACS","Ccr","AST","HAS_BLED") # top 10
  
  # CandidateVariables <- c("DD","DBP","NT_proBNP","TnI","LVEDD","UA","CHA2DS2_VACS","Ccr","AST","HAS_BLED",
  #                         "LVESD","Age","CREA","Statins","TT","PTINR","β.blocker","AF_category","Weight","Stroke") # top 20
  # 
  # CandidateVariables <- c("DD","DBP","NT_proBNP","TnI","LVEDD","UA","CHA2DS2_VACS","Ccr","AST","HAS_BLED",
  #                         "LVESD","Age","CREA","Statins","TT","PTINR","β.blocker","AF_category","Weight","Stroke",
  #                         "HR","Hypertension","CKD","SBP","Diuretics","LAD","Antiplatelet.agents","Clopidogrel","LDH","CHD")  # top30
  # 
  # CandidateVariables <- c("DD","DBP","NT_proBNP","TnI","LVEDD","UA","CHA2DS2_VACS","Ccr","AST","HAS_BLED",
  #                         "LVESD","Age","CREA","Statins","TT","PTINR","β.blocker","AF_category","Weight","Stroke",
  #                         "HR","Hypertension","CKD","SBP","Diuretics","LAD","Antiplatelet.agents","Clopidogrel","LDH","CHD",
  #                         "CCB","LVEF","CK","TSH","Rivaroxaban","ACEI_ARB","Heparin","Antihypertensive.agents","ALB","Diabetes")  # top40

  # Formula <- formula(paste(paste(Outcome,"~", collapse=" "),
  #                          paste(CandidateVariables, collapse=" + ")))
  
  grid = expand.grid(cost=1)
  
  cntrl = trainControl(method = "cv",number = 5)
  
  # model.svm = train(Formula,data = data.train,trControl = cntrl,tuneGrid = grid,method = "svmLinear2",
  #                   metric = "Kappa",probability = TRUE)
  
  model.svm = train(outcome_b~.,data = data.train,trControl = cntrl,tuneGrid = grid,method = "svmLinear2",
                    metric = "Kappa",probability = TRUE)  ## all feature
  
  data.train$p_prediction <- predict(model.svm, data.train, type="prob")[,2]
  data.test$p_prediction <- predict(model.svm, data.test, type="prob")[,2]
  pre_cutoff<-optimal.cutpoints(X = "p_prediction",status = "outcome_b", tag.healthy = 0, methods = "Youden",
                                data = data.train,ci.fit = TRUE,conf.level = 0.95)
  
  roc.train <- roc(data.train$outcome_b, data.train$p_prediction)
  roc.test <- roc(data.test$outcome_b, data.test$p_prediction)
  
  cm_train <- caret::confusionMatrix(as.factor(as.numeric(data.train$p_prediction>pre_cutoff$Youden$Global$optimal.cutoff$cutoff  )), as.factor(data.train$outcome_b), positive = "1")
  cm_test <- caret::confusionMatrix(as.factor(as.numeric(data.test$p_prediction>pre_cutoff$Youden$Global$optimal.cutoff$cutoff )), as.factor(data.test$outcome_b), positive = "1") 
  
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
  
  brier_train[i] <- mean((data.train$p_prediction-(as.numeric(data.train$outcome_b)-1))^2)
  brier_test[i] <- mean((data.test$p_prediction-(as.numeric(data.test$outcome_b)-1))^2)
  
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




#### random forest model
set.seed(1234)
index = sample(nrow(data_smo3),round(0.7*nrow(data_smo3)), )
data_rf3_train = data_smo3[index,]
data_rf3_test = data_smo3[-index,]

fit_smo_rf3 <- randomForest(outcome_b~.,data = data_rf3_train,importance = TRUE)

### feature importance 
importance_smo_rf3 <- data.frame(fit_smo_rf3$importance)
importance_smo_rf3 <- importance_smo_rf3[order(importance_smo_rf3$MeanDecreaseAccuracy,decreasing = TRUE),]
head(importance_smo_rf3)

# training set (1)
data_rf3_train$pred <- predict(fit_smo_rf3, data_rf3_train,type="prob")[,2]
roc_rf3_train <- roc(data_rf3_train$outcome_b, as.numeric(data_rf3_train$pred),ci=TRUE,print.auc = TRUE) #levels = c(0,1), direction = "<")
roc_rf3_train
# plot(roc_rf3_train,ci=TRUE,print.auc = TRUE,lty=2)

pre_cutoff<-optimal.cutpoints(X = "pred",status = "outcome_b", tag.healthy = 0, methods = "Youden",
                              data = data_rf3_train,ci.fit = TRUE,conf.level = 0.95)
summary(pre_cutoff)

caret::confusionMatrix(as.factor(as.numeric(data_rf3_train$pred>0.7)), as.factor(data_rf3_train$outcome_b), positive = "1")

# test set (0.9986)
data_rf3_test$pred <- predict(fit_smo_rf3, newdata = data_rf3_test,type="prob")[,2]
roc_rf3_test <- roc(data_rf3_test$outcome_b, as.numeric(data_rf3_test$pred),ci=TRUE,print.auc = TRUE) #levels = c(0,1), direction = "<")
roc_rf3_test
# plot(roc_rf3_test,ci=TRUE,print.auc = TRUE,lty=2)
caret::confusionMatrix(as.factor(as.numeric(data_rf3_test$pred>0.7)), as.factor(data_rf3_test$outcome_b), positive = "1") 



## calibration plot
data_rf3_cal<-subset(data_rf3_test,select = c(outcome_b,pred))
summary(data_rf3_cal)
data_rf3_cal$outcome_b<-as.factor(data_rf3_cal$outcome_b)

d1<-data_rf3_cal[which(data_rf3_cal$pred >=0 & data_rf3_cal$pred <0.1),]
d2<-data_rf3_cal[which(data_rf3_cal$pred >=0.1 & data_rf3_cal$pred <0.2),]
d3<-data_rf3_cal[which(data_rf3_cal$pred >=0.2 & data_rf3_cal$pred <0.3),]
d4<-data_rf3_cal[which(data_rf3_cal$pred >=0.3 & data_rf3_cal$pred <0.4),]
d5<-data_rf3_cal[which(data_rf3_cal$pred >=0.4 & data_rf3_cal$pred <0.5),]
d6<-data_rf3_cal[which(data_rf3_cal$pred >=0.5 & data_rf3_cal$pred <0.6),]
d7<-data_rf3_cal[which(data_rf3_cal$pred >=0.6 & data_rf3_cal$pred <0.7),]
d8<-data_rf3_cal[which(data_rf3_cal$pred >=0.7 & data_rf3_cal$pred <0.8),]
d9<-data_rf3_cal[which(data_rf3_cal$pred >=0.8 & data_rf3_cal$pred <0.9),]
d10<-data_rf3_cal[which(data_rf3_cal$pred >=0.9 & data_rf3_cal$pred <=1),]

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

summary(d1$outcome_b)

prop.test(nrow(d1[d1$outcome_b=="1",]),nrow(d1),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d2[d2$outcome_b=="1",]),nrow(d2),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d3[d3$outcome_b=="1",]),nrow(d3),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d4[d4$outcome_b=="1",]),nrow(d4),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d5[d5$outcome_b=="1",]),nrow(d5),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d6[d6$outcome_b=="1",]),nrow(d6),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d7[d7$outcome_b=="1",]),nrow(d7),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d8[d8$outcome_b=="1",]),nrow(d8),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d9[d9$outcome_b=="1",]),nrow(d9),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d10[d10$outcome_b=="1",]),nrow(d10),p = NULL, alternative = "two.sided",correct = TRUE)



###############################################################################################################
## random forest (k-fold cross validation)
set.seed(1234)   
folds <- createMultiFolds(y=data_smo3$outcome_b,k=5,times=10)   

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
  
  data.train <- data_smo3[folds[[i]],]
  data.test <- data_smo3[-folds[[i]],]
  
  # Outcome <- "outcome_b"
  
  # CandidateVariables <- c("DD","DBP","TnI","HAS_BLED","CHA2DS2_VACS","UA","LVEDD","Ccr","NT_proBNP","LVESD")  # top10
  
  # CandidateVariables <- c("DD","DBP","TnI","HAS_BLED","CHA2DS2_VACS","UA","LVEDD","Ccr","NT_proBNP","LVESD",
  #                         "AST","ALB","LAD","CREA","PTINR","Age","TT","Weight","CK","FBG")  # top20
  
  # CandidateVariables <- c("DD","DBP","TnI","HAS_BLED","CHA2DS2_VACS","UA","LVEDD","Ccr","NT_proBNP","LVESD",
  #                         "AST","ALB","LAD","CREA","PTINR","Age","TT","Weight","CK","FBG",
  #                         "SBP","LVEF","TSH","BMI","LDH","HR","GLU","β.blocker","CKD","Antihypertensive.agents")  # top30
  
  # CandidateVariables <- c("DD","DBP","TnI","HAS_BLED","CHA2DS2_VACS","UA","LVEDD","Ccr","NT_proBNP","LVESD",
  #                         "AST","ALB","LAD","CREA","PTINR","Age","TT","Weight","CK","FBG",
  #                         "SBP","LVEF","TSH","BMI","LDH","HR","GLU","β.blocker","CKD","Antihypertensive.agents",
  #                         "AF_category","Statins","Heparin","Antiplatelet.agents","Clopidogrel","Stroke",
  #                         "Anticoagulants","Hypertension","CHD","ACEI_ARB")  # top40
  
  # Formula <- formula(paste(paste(Outcome,"~", collapse=" "),
  #                          paste(CandidateVariables, collapse=" + ")))
  
  # model.rf <- randomForest(Formula, data= data.train,importance = TRUE)
  model.rf <- randomForest(outcome_b~., data= data.train,importance = TRUE)
  
  data.train$p_prediction <- predict(model.rf, data.train, type="prob")[,2]
  data.test$p_prediction <- predict(model.rf, data.test, type="prob")[,2]
  
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
  
  cutoff[i] <- pre_cutoff$Youden$Global$optimal.cutoff$cutoff
  c_test[i] <- roc.test$auc
  acc_test[i] <- cm_test$overall
  se_test[i] <- cm_test$byClass[1]
  spe_test[i] <- cm_test$byClass[2]
  ppv_test[i] <- cm_test$byClass[3]
  npv_test[i] <- cm_test$byClass[4]
  brier_train[i] <- mean((data.train$p_prediction-(as.numeric(data.train$outcome_b)-1))^2)
  brier_test[i] <- mean((data.test$p_prediction-(as.numeric(data.test$outcome_b)-1))^2)
  
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
index = sample(nrow(data_smo3),round(0.7*nrow(data_smo3)), )
data_gbm3_train = data_smo3[index,]
data_gbm3_test = data_smo3[-index,]

grid = expand.grid(interaction.depth = 9, n.trees = 350, shrinkage = 0.2,n.minobsinnode = 20)
cntrl = trainControl(method = "cv",number = 5)

fit_gbm3 = train(
  outcome_b ~.,data=data_gbm3_train,trControl = cntrl,tuneGrid = grid,method = "gbm",metric = "Kappa")

## feature importance 
gbmImp3 <- varImp(fit_gbm3,scale = FALSE)
gbmImp3 <- gbmImp3$importance

# training set (1)
data_gbm3_train$pred <- predict(fit_gbm3, data_gbm3_train,type = "prob")[,2]
roc_gbm3_train <- roc(data_gbm3_train$outcome_b, data_gbm3_train$pred,ci=TRUE,print.auc = TRUE,levels = c(0,1), direction = "<")
roc_gbm3_train
# plot(roc_gbm3_train,ci=TRUE,print.auc = TRUE,lty=2)

pre_cutoff<-optimal.cutpoints(X = "pred",status = "outcome_b", tag.healthy = 0, methods = "Youden",
                              data = data_gbm3_train,ci.fit = TRUE,conf.level = 0.95)
summary(pre_cutoff)

caret::confusionMatrix(as.factor(as.numeric(data_gbm3_train$pred>0.9999835)), as.factor(data_gbm3_train$outcome_b), positive = "1")

# test set (0.9992)
data_gbm3_test$pred <- predict(fit_gbm3, newdata = data_gbm3_test,type = "prob")[,2]
roc_gbm3_test <- roc(data_gbm3_test$outcome_b, data_gbm3_test$pred,ci=TRUE,print.auc = TRUE) #levels = c(0,1), direction = "<")
roc_gbm3_test
plot(roc_gbm3_test,ci=TRUE,print.auc = TRUE,lty=2)

caret::confusionMatrix(as.factor(as.numeric(data_gbm3_test$pred>0.9999835)), as.factor(data_gbm3_test$outcome_b), positive = "1") 


## calibration plot
data_gbm3_cal<-subset(data_gbm3_test,select = c(outcome_b,pred))
summary(data_gbm3_cal)
data_gbm3_cal$outcome_b<-as.factor(data_gbm3_cal$outcome_b)

d1<-data_gbm3_cal[which(data_gbm3_cal$pred >=0 & data_gbm3_cal$pred <0.1),]
d2<-data_gbm3_cal[which(data_gbm3_cal$pred >=0.1 & data_gbm3_cal$pred <0.2),]
d3<-data_gbm3_cal[which(data_gbm3_cal$pred >=0.2 & data_gbm3_cal$pred <0.3),]
d4<-data_gbm3_cal[which(data_gbm3_cal$pred >=0.3 & data_gbm3_cal$pred <0.4),]
d5<-data_gbm3_cal[which(data_gbm3_cal$pred >=0.4 & data_gbm3_cal$pred <0.5),]
d6<-data_gbm3_cal[which(data_gbm3_cal$pred >=0.5 & data_gbm3_cal$pred <0.6),]
d7<-data_gbm3_cal[which(data_gbm3_cal$pred >=0.6 & data_gbm3_cal$pred <0.7),]
d8<-data_gbm3_cal[which(data_gbm3_cal$pred >=0.7 & data_gbm3_cal$pred <0.8),]
d9<-data_gbm3_cal[which(data_gbm3_cal$pred >=0.8 & data_gbm3_cal$pred <0.9),]
d10<-data_gbm3_cal[which(data_gbm3_cal$pred >=0.9 & data_gbm3_cal$pred <=1),]

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

summary(d1$outcome_b)

prop.test(nrow(d1[d1$outcome_b=="1",]),nrow(d1),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d2[d2$outcome_b=="1",]),nrow(d2),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d3[d3$outcome_b=="1",]),nrow(d3),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d4[d4$outcome_b=="1",]),nrow(d4),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d5[d5$outcome_b=="1",]),nrow(d5),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d6[d6$outcome_b=="1",]),nrow(d6),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d7[d7$outcome_b=="1",]),nrow(d7),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d8[d8$outcome_b=="1",]),nrow(d8),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d9[d9$outcome_b=="1",]),nrow(d9),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d10[d10$outcome_b=="1",]),nrow(d10),p = NULL, alternative = "two.sided",correct = TRUE)



###############################################################################################################
## gbm model (k-fold cross validation)
set.seed(1234)   
folds <- createMultiFolds(y=data_smo3$outcome_b,k=5,times=2)   

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


for (i in 1:10){
  
  data.train <- data_smo3[folds[[i]],]
  data.test <- data_smo3[-folds[[i]],]
  
  grid = expand.grid(interaction.depth = 9, n.trees =350 , shrinkage =0.2 ,n.minobsinnode = 20)
  
  cntrl = trainControl(method = "cv",number = 5)
  
  model.gbm = train(
    # outcome_b ~ DD+TnI+CHA2DS2_VACS+ALB+DBP+Ccr+NT_proBNP+UA+LAD+LVESD, # top10
    
    # outcome_b ~ DD+TnI+CHA2DS2_VACS+ALB+DBP+Ccr+NT_proBNP+UA+LAD+LVESD+
    #   HAS_BLED+LVEDD+AST+TT+CK+CKD+LVEF+PTINR+SBP+HR, # top20
    
    # outcome_b ~ DD+TnI+CHA2DS2_VACS+ALB+DBP+Ccr+NT_proBNP+UA+LAD+LVESD+
    #   HAS_BLED+LVEDD+AST+TT+CK+CKD+LVEF+PTINR+SBP+HR+
    #   Age+FBG+LDH+BMI+CREA+Weight+ACEI_ARB+GLU+Anticoagulants+TSH, # top30
    
    # outcome_b ~ DD+TnI+CHA2DS2_VACS+ALB+DBP+Ccr+NT_proBNP+UA+LAD+LVESD+
    #   HAS_BLED+LVEDD+AST+TT+CK+CKD+LVEF+PTINR+SBP+HR+
    #   Age+FBG+LDH+BMI+CREA+Weight+ACEI_ARB+GLU+Anticoagulants+TSH+
    #   β.blocker+Heparin+Diuretics+Gender+Statins+AF_category+Antiplatelet.agents+Prior.PCI+Antihypertensive.agents+Aspirin, # top40
    
    outcome_b ~ ., # all
    
    data = data.train,trControl = cntrl,tuneGrid = grid,method = "gbm",metric = "Kappa")
  
  
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
  
  cutoff[i] <- pre_cutoff$Youden$Global$optimal.cutoff$cutoff
  c_test[i] <- roc.test$auc
  acc_test[i] <- cm_test$overall
  se_test[i] <- cm_test$byClass[1]
  spe_test[i] <- cm_test$byClass[2]
  ppv_test[i] <- cm_test$byClass[3]
  npv_test[i] <- cm_test$byClass[4]
  brier_train[i] <- mean((data.train$p_prediction-(as.numeric(data.train$outcome_b)-1))^2)
  brier_test[i] <- mean((data.test$p_prediction-(as.numeric(data.test$outcome_b)-1))^2)
  
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
setwd("/Users/weiyuna/Desktop/工作/合作项目/上海胸科/20211103导出数据/房颤/补充结局-不包含瓣膜手术")
# setwd("C:/Users/90604/Desktop/WYN")
data_smo3_xgb<-read.csv("data_smo3.csv")
data_smo3_xgb<-subset(data_smo3_xgb,select = c(outcome_b:LAD,LVEF:UA,TT:CREA,Ccr:CK,NT_proBNP:Prior.CABG))
names(data_smo3_xgb)

data_smo3_xgb$Persistent_AF<-ifelse(data_smo3_xgb$AF_category=="2",1,0)
data_smo3_xgb$Chronic_AF<-ifelse(data_smo3_xgb$AF_category=="3",1,0)
names(data_smo3_xgb)
str(data_smo3_xgb)

### hyper-parameter selection
set.seed(1234)
index = sample(nrow(data_smo3_xgb),round(0.7*nrow(data_smo3_xgb)), )
data.train.xg = data_smo3_xgb[index,]
data.test.xg = data_smo3_xgb[-index,]
summary(data.train.xg)
summary(data.test.xg)

data.train.xg$outcome_b <- ifelse(data.train.xg$outcome_b== "1","yes","no")
data.test.xg$outcome_b<- ifelse(data.test.xg$outcome_b == "1","yes","no")

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
train.xgb3 = train(
  x = data.train.xg[,c(3:64)],
  y = as.factor(data.train.xg[, 1]),
  trControl = cntrl,
  tuneGrid = grid,
  method = "xgbTree",
  metric = "ROC"
  # metric = "Kappa"
)

train.xgb3


#######################################################################################################
### random split
set.seed(1234)
index = sample(nrow(data_smo3_xgb),round(0.7*nrow(data_smo3_xgb)), )
data.train.xg = data_smo3_xgb[index,]
data.test.xg = data_smo3_xgb[-index,]

## data preprocessing
data.train.xg1<-data.matrix(data.train.xg[,c(3:64)])
data.train.xg2<-Matrix(data.train.xg1,sparse = T)
data.train.xg_y3<-data.train.xg$outcome_b
traindata.xg<-list(data=data.train.xg2,label=data.train.xg_y3)
dtrain<-xgb.DMatrix(data = traindata.xg$data,label=traindata.xg$label)

data.test.xg1<-data.matrix(data.test.xg[,c(3:64)])
data.test.xg2<-Matrix(data.test.xg1,sparse = T)
data.test.xg_y3<-data.test.xg$outcome_b
testdata.xg<-list(data=data.test.xg2,label=data.test.xg_y3)
dtest<-xgb.DMatrix(data = testdata.xg$data,label=testdata.xg$label)


fit_xgb3 <-xgboost(data = dtrain, objective="binary:logistic", booster="gbtree",nrounds = 100, max_depth = 7, eta = 0.5, 
                      gamma =0.5, colsample_bytree = 0.5, min_child_weight = 1 , subsample = 0.7)

### feature importance
importance.xgb3<-xgb.importance(data.train.xg2@Dimnames[[2]],model = fit_xgb3)
# xgb.ggplot.importance(importance.xgb3,rel_to_first = TRUE,n_clusters = 3, measure = "Cover")
# xgb.ggplot.importance(importance.xgb3,rel_to_first = TRUE,n_clusters = 3, measure = "Gain")
# xgb.plot.importance(importance.xgb3,rel_to_first = TRUE,xlab = "Relative importance")
# xgb.ggplot.shap.summary(data = traindata.xg$data, model = fit_xgb3,target_class = 1,top_n = 20)

## predict
pre.xgb.train3<-predict(fit_xgb3,dtrain)
pre.xgb.test3<-predict(fit_xgb3,newdata = dtest)

pre_cutoff<-optimal.cutpoints(X = "pre.xgb.train3",status = "data.train.xg_y3", tag.healthy = 0, methods = "Youden",
                              data = data.frame(cbind(pre.xgb.train3,data.train.xg_y3)),ci.fit = TRUE,conf.level = 0.95)
summary(pre_cutoff)

## model evaluation
# training set (1)
xgb.cf.train3<-caret::confusionMatrix(as.factor(as.numeric(pre.xgb.train3>0.6589385)),as.factor(data.train.xg_y3), positive = "1")
xgb.cf.train3
roc_xgb3_train <- roc(response = data.train.xg_y3, predictor = pre.xgb.train3,ci=TRUE,print.auc = TRUE)
roc_xgb3_train
# plot(roc_xgb3_train,ci=TRUE,print.auc = TRUE,lty=2)

## test set (0.9977)
xgb.cf.test3<-caret::confusionMatrix(as.factor(as.numeric(pre.xgb.test3>0.6589385)),as.factor(data.test.xg_y3), positive = "1")
xgb.cf.test3
roc_xgb3_test <- roc(response = data.test.xg_y3, predictor = pre.xgb.test3,ci=TRUE,print.auc = TRUE)
roc_xgb3_test
# plot(roc_xgb3_test,ci=TRUE,print.auc = TRUE,lty=2)

# brier score()
val_xgb3 <- val.prob(pre.xgb.test3,data.test.xg_y3,cex = 0.8)

## calibration plot
data_xgb3_cal<-data.frame(cbind(pre.xgb.test3,data.test.xg_y3))
summary(data_xgb3_cal)
data_xgb3_cal$data.test.xg_y3<-as.factor(data_xgb3_cal$data.test.xg_y3)

d1<-data_xgb3_cal[which(data_xgb3_cal$pre.xgb.test3 >=0 & data_xgb3_cal$pre.xgb.test3 <0.1),]
d2<-data_xgb3_cal[which(data_xgb3_cal$pre.xgb.test3 >=0.1 & data_xgb3_cal$pre.xgb.test3 <0.2),]
d3<-data_xgb3_cal[which(data_xgb3_cal$pre.xgb.test3 >=0.2 & data_xgb3_cal$pre.xgb.test3 <0.3),]
d4<-data_xgb3_cal[which(data_xgb3_cal$pre.xgb.test3 >=0.3 & data_xgb3_cal$pre.xgb.test3 <0.4),]
d5<-data_xgb3_cal[which(data_xgb3_cal$pre.xgb.test3 >=0.4 & data_xgb3_cal$pre.xgb.test3 <0.5),]
d6<-data_xgb3_cal[which(data_xgb3_cal$pre.xgb.test3 >=0.5 & data_xgb3_cal$pre.xgb.test3 <0.6),]
d7<-data_xgb3_cal[which(data_xgb3_cal$pre.xgb.test3 >=0.6 & data_xgb3_cal$pre.xgb.test3 <0.7),]
d8<-data_xgb3_cal[which(data_xgb3_cal$pre.xgb.test3 >=0.7 & data_xgb3_cal$pre.xgb.test3 <0.8),]
d9<-data_xgb3_cal[which(data_xgb3_cal$pre.xgb.test3 >=0.8 & data_xgb3_cal$pre.xgb.test3 <0.9),]
d10<-data_xgb3_cal[which(data_xgb3_cal$pre.xgb.test3 >=0.9 & data_xgb3_cal$pre.xgb.test3 <=1),]

t.test(d1$pre.xgb.test3)
t.test(d2$pre.xgb.test3)
t.test(d3$pre.xgb.test3)
t.test(d4$pre.xgb.test3)
t.test(d5$pre.xgb.test3)
t.test(d6$pre.xgb.test3)
t.test(d7$pre.xgb.test3)
t.test(d8$pre.xgb.test3)
t.test(d9$pre.xgb.test3)
t.test(d10$pre.xgb.test3)

summary(d1$data.test.xg_y3)

prop.test(nrow(d1[d1$data.test.xg_y3=="1",]),nrow(d1),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d2[d2$data.test.xg_y3=="1",]),nrow(d2),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d3[d3$data.test.xg_y3=="1",]),nrow(d3),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d4[d4$data.test.xg_y3=="1",]),nrow(d4),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d5[d5$data.test.xg_y3=="1",]),nrow(d5),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d6[d6$data.test.xg_y3=="1",]),nrow(d6),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d7[d7$data.test.xg_y3=="1",]),nrow(d7),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d8[d8$data.test.xg_y3=="1",]),nrow(d8),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d9[d9$data.test.xg_y3=="1",]),nrow(d9),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d10[d10$data.test.xg_y3=="1",]),nrow(d10),p = NULL, alternative = "two.sided",correct = TRUE)


###############################################################################################################
### xgboost model (k-fold cross validation)
set.seed(1234)
folds <- createMultiFolds(y=data_smo3_xgb$outcome_b,k=5,times=2)   

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

for (i in 1:10){
  data.train.xg <- data_smo3_xgb[folds[[i]],]
  data.test.xg <- data_smo3_xgb[-folds[[i]],]
  
  
  ## data preprocessing
  # data.train.xg1<-data.matrix(data.train.xg[,c("TnI","HAS_BLED","DBP","UA","PTINR","DD","Ccr","CHA2DS2_VACS","LVEDD","ALB")])  ##top10
  
  # data.train.xg1<-data.matrix(data.train.xg[,c("TnI","HAS_BLED","DBP","UA","PTINR","DD","Ccr","CHA2DS2_VACS","LVEDD","ALB",
  #                                              "NT_proBNP","LVEF","AST","LDH","LAD","FBG","Age","LVESD","CREA","HR")])  ##top20
  
  # data.train.xg1<-data.matrix(data.train.xg[,c("TnI","HAS_BLED","DBP","UA","PTINR","DD","Ccr","CHA2DS2_VACS","LVEDD","ALB",
  #                                              "NT_proBNP","LVEF","AST","LDH","LAD","FBG","Age","LVESD","CREA","HR",
  #                                              "Dabigatran","TSH","BMI","TT","CK","Persistent_AF","Antiplatelet.agents","Weight","GLU","ACEI_ARB")])  ##top30
 
  # data.train.xg1<-data.matrix(data.train.xg[,c("TnI","HAS_BLED","DBP","UA","PTINR","DD","Ccr","CHA2DS2_VACS","LVEDD","ALB",
  #                                              "NT_proBNP","LVEF","AST","LDH","LAD","FBG","Age","LVESD","CREA","HR",
  #                                              "Dabigatran","TSH","BMI","TT","CK","Persistent_AF","Antiplatelet.agents","Weight","GLU","ACEI_ARB",
  #                                              "CKD","Statins","Anticoagulants","SBP","β.blocker","CCB","Hypertension")])   ## top40
  
  data.train.xg1<-data.matrix(data.train.xg[,c(3:64)]) ## all feature
  
  data.train.xg2<-Matrix(data.train.xg1,sparse = T)
  data.train.xg_y<-data.train.xg$outcome_b
  traindata.xg<-list(data=data.train.xg2,label=data.train.xg_y)
  dtrain<-xgb.DMatrix(data = traindata.xg$data,label=traindata.xg$label)
  
  # data.test.xg1<-data.matrix(data.test.xg[,c("TnI","HAS_BLED","DBP","UA","PTINR","DD","Ccr","CHA2DS2_VACS","LVEDD","ALB")])  ##top10
  
  # data.test.xg1<-data.matrix(data.test.xg[,c("TnI","HAS_BLED","DBP","UA","PTINR","DD","Ccr","CHA2DS2_VACS","LVEDD","ALB",
  #                                            "NT_proBNP","LVEF","AST","LDH","LAD","FBG","Age","LVESD","CREA","HR")])  ##top20
  
  # data.test.xg1<-data.matrix(data.test.xg[,c("TnI","HAS_BLED","DBP","UA","PTINR","DD","Ccr","CHA2DS2_VACS","LVEDD","ALB",
  #                                            "NT_proBNP","LVEF","AST","LDH","LAD","FBG","Age","LVESD","CREA","HR",
  #                                            "Dabigatran","TSH","BMI","TT","CK","Persistent_AF","Antiplatelet.agents","Weight","GLU","ACEI_ARB")])  ##top30
 
  # data.test.xg1<-data.matrix(data.test.xg[,c("TnI","HAS_BLED","DBP","UA","PTINR","DD","Ccr","CHA2DS2_VACS","LVEDD","ALB",
  #                                            "NT_proBNP","LVEF","AST","LDH","LAD","FBG","Age","LVESD","CREA","HR",
  #                                            "Dabigatran","TSH","BMI","TT","CK","Persistent_AF","Antiplatelet.agents","Weight","GLU","ACEI_ARB",
  #                                            "CKD","Statins","Anticoagulants","SBP","β.blocker","CCB","Hypertension")])  ##top40
  
  data.test.xg1<-data.matrix(data.test.xg[,c(3:64)])  ## all feature
  
  data.test.xg2<-Matrix(data.test.xg1,sparse = T)
  data.test.xg_y<-data.test.xg$outcome_b
  testdata.xg<-list(data=data.test.xg2,label=data.test.xg_y)
  dtest<-xgb.DMatrix(data = testdata.xg$data,label=testdata.xg$label)
  
  model_xgb_chuxue <-xgboost(data = dtrain,objective="binary:logistic", booster="gbtree",nrounds = 100, max_depth = 7, eta = 0.5, 
                             gamma =0.5, colsample_bytree = 0.5, min_child_weight = 1 , subsample = 0.7)
  
  ## feature importance
  # importance_chuxue<-xgb.importance(data.train.xg2@Dimnames[[2]],model = model_xgb_chuxue)
  # xgb.ggplot.importance(importance_chuxue,rel_to_first = TRUE,n_clusters = 3, measure = "Cover")
  # xgb.ggplot.importance(importance_chuxue,rel_to_first = TRUE,n_clusters = 3, measure = "Gain",top_n = 20)
  # xgb.ggplot.shap.summary(data = traindata.xg$data, model = model_xgb_chuxue,target_class = 1,top_n = 20)
  
  ## predict
  pre.xgb.train <-predict(model_xgb_chuxue,dtrain)
  pre.xgb.test <-predict(model_xgb_chuxue,newdata = dtest)
  
  pre_cutoff<-optimal.cutpoints(X = "pre.xgb.train",status = "data.train.xg_y", tag.healthy = 0, methods = "Youden",
                                data = data.frame(cbind(pre.xgb.train,data.train.xg_y)),ci.fit = TRUE,conf.level = 0.95)
  
  roc.xgb.train <- roc(response = data.train.xg_y, predictor = pre.xgb.train,ci=TRUE,print.auc = TRUE)
  roc.xgb.test <- roc(response = data.test.xg_y, predictor = pre.xgb.test,ci=TRUE,print.auc = TRUE)
  
  cm.xgb_train <- caret::confusionMatrix(as.factor(as.numeric(pre.xgb.train>pre_cutoff$Youden$Global$optimal.cutoff$cutoff)), as.factor(data.train.xg_y), positive = "1")
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



### ROC curve of four models
par(mfrow=c(1,1))
plot(roc_log3_test,ci=TRUE,lty=2,col="#1B1919FF",legacy.axes=T)
plot(roc_svm3_test,add=TRUE,ci=TRUE,lty=2,col="#3B4992FF",legacy.axes=T)
plot(roc_rf3_test,add=TRUE,ci=TRUE,lty=2,col="#008280FF")
plot(roc_gbm3_test,add=TRUE,ci=TRUE,lty=2,col="#631879FF")
plot(roc_xgb3_test,add=TRUE,ci=TRUE,lty=2,col="#A20056FF")
abline(h = seq(0,1,0.1), v = seq(0,1,0.1), col = "lightgrey", lty = 1,lwd = 0.4)
legend("bottomright",legend = c("Hemorrhage AUC (95% CI)",
                                "LR: 0.976 (0.974-0.977)",
                                "SVM: 0.974 (0.972-0.977)",
                                "RF: 1.000 (1.000-1.000)",
                                "GBM: 1.000 (1.000-1.000)",
                                "XGBoost: 0.998 (0.997-0.999)"),
       col = c("white","#1B1919FF","#3B4992FF","#008280FF","#631879FF","#A20056FF"),lty = 2,lwd = 1,cex = 0.8, text.font = 2,box.lwd = "none")
# legend("topright",legend = "C", text.font = 2,box.lwd = "none")



## combined roc
par(mfrow=c(1,3))

## any complication
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

## cardiac effusion
plot(roc_log2_test,ci=TRUE,lty=2,col="#1B1919FF",legacy.axes=T)
plot(roc_svm2_test,add=TRUE,ci=TRUE,lty=2,col="#3B4992FF",legacy.axes=T)
plot(roc_rf2_test,add=TRUE,ci=TRUE,lty=2,col="#008280FF")
plot(roc_gbm2_test,add=TRUE,ci=TRUE,lty=2,col="#631879FF")
plot(roc_xgb2_test,add=TRUE,ci=TRUE,lty=2,col="#A20056FF")
abline(h = seq(0,1,0.1), v = seq(0,1,0.1), col = "lightgrey", lty = 1,lwd = 0.4)
legend("bottomright",legend = c("Cardiac effusion AUC (95% CI)",
                                "LR: 0.956 (0.955-0.958)",
                                "SVM: 0.953 (0.947-0.959)",
                                "RF: 1.000 (1.000-1.000)",
                                "GBM: 1.000 (1.000-1.000)",
                                "XGBoost: 0.999 (0.998-0.999)"),
       col = c("white","#1B1919FF","#3B4992FF","#008280FF","#631879FF","#A20056FF"),lty = 2,lwd = 1,cex = 0.8, text.font = 2,box.lwd = "none")
# legend("topleft",legend = "B", text.font = 2,box.lwd = "none")

## hemorrhage
plot(roc_log3_test,ci=TRUE,lty=2,col="#1B1919FF",legacy.axes=T)
plot(roc_svm3_test,add=TRUE,ci=TRUE,lty=2,col="#3B4992FF",legacy.axes=T)
plot(roc_rf3_test,add=TRUE,ci=TRUE,lty=2,col="#008280FF")
plot(roc_gbm3_test,add=TRUE,ci=TRUE,lty=2,col="#631879FF")
plot(roc_xgb3_test,add=TRUE,ci=TRUE,lty=2,col="#A20056FF")
abline(h = seq(0,1,0.1), v = seq(0,1,0.1), col = "lightgrey", lty = 1,lwd = 0.4)
legend("bottomright",legend = c("Hemorrhage AUC (95% CI)",
                                "LR: 0.976 (0.974-0.977)",
                                "SVM: 0.974 (0.972-0.977)",
                                "RF: 1.000 (1.000-1.000)",
                                "GBM: 1.000 (1.000-1.000)",
                                "XGBoost: 0.998 (0.997-0.999)"),
       col = c("white","#1B1919FF","#3B4992FF","#008280FF","#631879FF","#A20056FF"),lty = 2,lwd = 1,cex = 0.8, text.font = 2,box.lwd = "none")
# legend("topright",legend = "C", text.font = 2,box.lwd = "none")


