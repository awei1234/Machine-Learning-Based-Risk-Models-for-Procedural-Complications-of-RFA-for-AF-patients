setwd("/Users/weiyuna/Desktop/工作/合作项目/上海胸科/20211103导出数据/房颤/补充结局-不包含瓣膜手术")
install.packages("devtools")
library(devtools)
install_github("cran/DMwR",ref = "master")
# install.packages("DMwR")
# install.packages("caret")
# install.packages("Rmisc")
# install.packages("MASS")
# install.packages("pROC")
# install.packages("Matrix")
# install.packages("grid")
# install.packages("xgboost")
# install.packages("e1071")
# install.packages("gbm")
# install.packages("terra")
# install.packages("modEvA")
# install.packages("OptimalCutpoints")
# install.packages("randomForest")
# install.packages("ggplot2")
# install.packages("lattice")

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
library(ggplot2)
library(lattice)


data<-read.csv("data_imputation_full.csv")
data<-subset(data,select = c(patient_id,outcome:Prior.CABG))

factor.name<-c("outcome","outcome_h","outcome_b", "AF_category", "Gender","LAD_factor","HF","UA_factor","CREA_factor","CK_factor","Aspirin","Clopidogrel","Other.antiplatelet.agents",
               "Antiplatelet.agents","Warfarin","Dabigatran","Rivaroxaban","Heparin", "Anticoagulants","Statins",
               "ACEI_ARB","β.blocker","Diuretics","CCB","Antihypertensive.agents","Ever.smoking","Ever.drinking","Angina","Heart.failure",
               "Stroke","PAD","COPD","Hypertension","Diabetes","Hyperlipidemia","MI","CHD", "CKD","Prior.RFA","Prior.PCI","Prior.CABG")

for (i in factor.name) {
  data[,i]<-as.factor(data[,i])
}

data_smo2_pre<-subset(data,select = c(outcome_h,AF_category:Age,Weight:LAD,LVEF:UA,TT:CREA,Ccr:CK,NT_proBNP:Prior.CABG))
data_smo2<-SMOTE(outcome_h ~.,data=data_smo2_pre,perc.over = 2282,perc.under = 522,k=5,seed=1234) #### (1:5)

# write.csv(data_smo2,file = "data_smo2.csv")



## logistic regression
set.seed(1234)
index = sample(nrow(data_smo2),round(0.7*nrow(data_smo2)), )
data_smo2_train = data_smo2[index,]
data_smo2_test = data_smo2[-index,]

fit_smo2_log <- glm(outcome_h ~ ., data=data_smo2_train,family=binomial) # full model

## feature importance
logImp2 <- varImp(fit_smo2_log,scale=FALSE)


# training set (0.9631)
data_smo2_train$pred <- predict(fit_smo2_log, data_smo2_train,type="response")
roc_log2_train <- roc(data_smo2_train$outcome_h, data_smo2_train$pred,ci=TRUE,print.auc = TRUE) #levels = c(0,1), direction = "<")
roc_log2_train
plot(roc_log2_train,ci=TRUE,print.auc = TRUE,lty=2)

pre_cutoff<-optimal.cutpoints(X = "pred",status = "outcome_h", tag.healthy = 0, methods = "Youden",
                              data = data_smo2_train,ci.fit = TRUE,conf.level = 0.95)
summary(pre_cutoff)

caret::confusionMatrix(as.factor(as.numeric(data_smo2_train$pred>0.1561087 )), as.factor(data_smo2_train$outcome_h), positive = "1")

# test set (0.96)
data_smo2_test$pred <- predict(fit_smo2_log, newdata = data_smo2_test, type="response")
roc_log2_test <- roc(data_smo2_test$outcome_h, data_smo2_test$pred,ci=TRUE,print.auc = TRUE) #levels = c(0,1), direction = "<")
roc_log2_test
plot(roc_log2_test,ci=TRUE,print.auc = TRUE,lty=2)
caret::confusionMatrix(as.factor(as.numeric(data_smo2_test$pred>0.1561087)), as.factor(data_smo2_test$outcome_h), positive = "1") 

### brier score (0.05072842)
brier(data_smo2_test)

### PR Curve
AUC(model = fit_smo2_log, obs = data_smo2_test$outcome_h, pred = data_smo2_test$pred, simplif = TRUE, interval = 0.005,
    FPR.limits = c(0, 1), curve = "PR", method = "trapezoid", plot = TRUE, diag = TRUE, 
    diag.col = "black", diag.lty = 2, curve.col = "blue", curve.lty = 1, curve.lwd = 2, plot.values = TRUE, 
    plot.digits = 3, plot.preds = FALSE, grid = FALSE,
    xlab = "Recall" , ylab = "Precision")

## calibration plot
data_log2_cal<-subset(data_smo2_test,select = c(outcome_h,pred))
summary(data_log2_cal)

data_log2_cal$outcome_h<-as.factor(data_log2_cal$outcome_h)
summary(data_log2_cal)

d1<-data_log2_cal[which(data_log2_cal$pred >=0 & data_log2_cal$pred <0.1),]
d2<-data_log2_cal[which(data_log2_cal$pred >=0.1 & data_log2_cal$pred <0.2),]
d3<-data_log2_cal[which(data_log2_cal$pred >=0.2 & data_log2_cal$pred <0.3),]
d4<-data_log2_cal[which(data_log2_cal$pred >=0.3 & data_log2_cal$pred <0.4),]
d5<-data_log2_cal[which(data_log2_cal$pred >=0.4 & data_log2_cal$pred <0.5),]
d6<-data_log2_cal[which(data_log2_cal$pred >=0.5 & data_log2_cal$pred <0.6),]
d7<-data_log2_cal[which(data_log2_cal$pred >=0.6 & data_log2_cal$pred <0.7),]
d8<-data_log2_cal[which(data_log2_cal$pred >=0.7 & data_log2_cal$pred <0.8),]
d9<-data_log2_cal[which(data_log2_cal$pred >=0.8 & data_log2_cal$pred <0.9),]
d10<-data_log2_cal[which(data_log2_cal$pred >=0.9 & data_log2_cal$pred <=1),]

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

summary(d8$outcome_h)

prop.test(nrow(d1[d1$outcome_h=="1",]),nrow(d1),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d2[d2$outcome_h=="1",]),nrow(d2),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d3[d3$outcome_h=="1",]),nrow(d3),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d4[d4$outcome_h=="1",]),nrow(d4),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d5[d5$outcome_h=="1",]),nrow(d5),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d6[d6$outcome_h=="1",]),nrow(d6),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d7[d7$outcome_h=="1",]),nrow(d7),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d8[d8$outcome_h=="1",]),nrow(d8),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d9[d9$outcome_h=="1",]),nrow(d9),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d10[d10$outcome_h=="1",]),nrow(d10),p = NULL, alternative = "two.sided",correct = TRUE)



###############################################################################################################
## logistic regression (k-fold cross validation)
set.seed(1234)   
folds <- createMultiFolds(y=data_smo2$outcome_h,k=5,times=10)   

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
  
  data.train <- data_smo2[folds[[i]],]
  data.test <- data_smo2[-folds[[i]],]
  
  # Outcome <- "outcome_h"
  
  # CandidateVariables <- c("Hyperlipidemia","ALB","AF_category","Statins","GLU","Ever.smoking","NT_proBNP","AST","Prior.RFA","HR") # top10
  
  # CandidateVariables <- c("Hyperlipidemia","ALB","AF_category","Statins","GLU","Ever.smoking","NT_proBNP","AST","Prior.RFA","HR",
  #                         "Age","Diabetes","Warfarin","Anticoagulants","UA","LVESD","LVEDD","HF","FBG","Ccr") # top20
  # 
  # CandidateVariables <- c("Hyperlipidemia","ALB","AF_category","Statins","GLU","Ever.smoking","NT_proBNP","AST","Prior.RFA","HR",
  #                         "Age","Diabetes","Warfarin","Anticoagulants","UA","LVESD","LVEDD","HF","FBG","Ccr",
  #                         "LAD","CHA2DS2_VACS","Prior.PCI","BMI","DD","ACEI_ARB","CREA","CK","CCB","Weight") # top30
  # 
  # CandidateVariables <- c("Hyperlipidemia","ALB","AF_category","Statins","GLU","Ever.smoking","NT_proBNP","AST","Prior.RFA","HR",
  #                         "Age","Diabetes","Warfarin","Anticoagulants","UA","LVESD","LVEDD","HF","FBG","Ccr",
  #                         "LAD","CHA2DS2_VACS","Prior.PCI","BMI","DD","ACEI_ARB","CREA","CK","CCB","Weight",
  #                         "LDH","LVEF","DBP","CHD","PAD","Gender","SBP","PTINR","β.blocker","Heparin") # top40
  
  
  # Formula <- formula(paste(paste(Outcome,"~", collapse=" "),
  #                          paste(CandidateVariables, collapse=" + ")))
  
  # model.log <- glm(Formula, data= data.train,family=binomial)
  model.log <- glm(outcome_h~., data= data.train,family=binomial)
  
  data.train$p_prediction <- predict(model.log, data.train, type="response")
  data.test$p_prediction <- predict(model.log, data.test, type="response")
  
  pre_cutoff<-optimal.cutpoints(X = "p_prediction",status = "outcome_h", tag.healthy = 0, methods = "Youden",
                                data = data.train,ci.fit = TRUE,conf.level = 0.95)
  
  roc.train <- roc(data.train$outcome_h, data.train$p_prediction)
  roc.test <- roc(data.test$outcome_h, data.test$p_prediction)
  
  cm_train <- caret::confusionMatrix(as.factor(as.numeric(data.train$p_prediction>pre_cutoff$Youden$Global$optimal.cutoff$cutoff)), as.factor(data.train$outcome_h), positive = "1")
  cm_test <- caret::confusionMatrix(as.factor(as.numeric(data.test$p_prediction>pre_cutoff$Youden$Global$optimal.cutoff$cutoff)), as.factor(data.test$outcome_h), positive = "1") 
  
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
  
  brier_train[i] <- mean((data.train$p_prediction-(as.numeric(data.train$outcome_h)-1))^2)
  brier_test[i] <- mean((data.test$p_prediction-(as.numeric(data.test$outcome_h)-1))^2)
  
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
index = sample(nrow(data_smo2),round(0.7*nrow(data_smo2)), )
data_svm2_train = data_smo2[index,]
data_svm2_test = data_smo2[-index,]

## hyper-parameter selection

# data_svm2_train$outcome_h <- ifelse(data_svm2_train$outcome_h=="1","yes","no")
# data_svm2_test$outcome_h <- ifelse(data_svm2_test$outcome_h=="1","yes","no")
# data_svm2_train$outcome_h <- ifelse(data_svm2_train$outcome_h=="yes","1","0")
# data_svm2_test$outcome_h <- ifelse(data_svm2_test$outcome_h=="yes","1","0")

grid = expand.grid(
  # cost = c(0.001, 0.01, 0.1, 1,10)
  cost=10
)


cntrl = trainControl(
  method = "cv",
  number = 5
  # ,classProbs = TRUE,
  # summaryFunction = twoClassSummary
)

fit_smo2_svm = train(
  outcome_h ~ .,
  data = data_svm2_train,
  trControl = cntrl,
  tuneGrid = grid,
  method = "svmLinear2",
  # metric = "ROC"
  metric = "Kappa",
  probability = TRUE
)


## feature importance
svmImp2 <- varImp(fit_smo2_svm,scale = FALSE)
plot(svmImp2,top=40)
svmImp2 <- svmImp2$importance
svmImp2 <- svmImp2[order(svmImp2$X0,decreasing=TRUE),]

# training set-full model (0.9586)
data_svm2_train$pred <- predict(fit_smo2_svm,data_svm2_train,type="prob")[,2]
roc_svm2_train <- roc(data_svm2_train$outcome_h, data_svm2_train$pred,ci=TRUE,print.auc = TRUE) #levels = c(0,1), direction = "<")
roc_svm2_train
plot(roc_svm2_train,ci=TRUE,print.auc = TRUE,lty=2)

pre_cutoff<-optimal.cutpoints(X = "pred",status = "outcome_h", tag.healthy = 0, methods = "Youden",
                              data = data_svm2_train,ci.fit = TRUE,conf.level = 0.95)
summary(pre_cutoff)
caret::confusionMatrix(as.factor(as.numeric(data_svm2_train$pred>0.1888098 )), as.factor(data_svm2_train$outcome_h), positive = "1")

# test set-full model (0.9585)
data_svm2_test$pred <- predict(fit_smo2_svm, newdata = data_svm2_test, type="prob")[,2]
roc_svm2_test <- roc(data_svm2_test$outcome_h, data_svm2_test$pred,ci=TRUE,print.auc = TRUE) #levels = c(0,1), direction = "<")
roc_svm2_test
plot(roc_svm2_test,ci=TRUE,print.auc = TRUE,lty=2)
caret::confusionMatrix(as.factor(as.numeric(data_svm2_test$pred>0.1888098)), as.factor(data_svm2_test$outcome_h), positive = "1") 

## brier score (0.05307284)
brier(data_svm2_test)


## calibration plot
data_svm2_cal<-subset(data_svm2_test,select = c(outcome_h,pred))
summary(data_svm2_cal)

data_svm2_cal$outcome_h<-as.factor(data_svm2_cal$outcome_h)
summary(data_svm2_cal)

d1<-data_svm2_cal[which(data_svm2_cal$pred >=0 & data_svm2_cal$pred <0.1),]
d2<-data_svm2_cal[which(data_svm2_cal$pred >=0.1 & data_svm2_cal$pred <0.2),]
d3<-data_svm2_cal[which(data_svm2_cal$pred >=0.2 & data_svm2_cal$pred <0.3),]
d4<-data_svm2_cal[which(data_svm2_cal$pred >=0.3 & data_svm2_cal$pred <0.4),]
d5<-data_svm2_cal[which(data_svm2_cal$pred >=0.4 & data_svm2_cal$pred <0.5),]
d6<-data_svm2_cal[which(data_svm2_cal$pred >=0.5 & data_svm2_cal$pred <0.6),]
d7<-data_svm2_cal[which(data_svm2_cal$pred >=0.6 & data_svm2_cal$pred <0.7),]
d8<-data_svm2_cal[which(data_svm2_cal$pred >=0.7 & data_svm2_cal$pred <0.8),]
d9<-data_svm2_cal[which(data_svm2_cal$pred >=0.8 & data_svm2_cal$pred <0.9),]
d10<-data_svm2_cal[which(data_svm2_cal$pred >=0.9 & data_svm2_cal$pred <=1),]

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

summary(d1$outcome_h)

prop.test(nrow(d1[d1$outcome_h=="1",]),nrow(d1),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d2[d2$outcome_h=="1",]),nrow(d2),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d3[d3$outcome_h=="1",]),nrow(d3),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d4[d4$outcome_h=="1",]),nrow(d4),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d5[d5$outcome_h=="1",]),nrow(d5),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d6[d6$outcome_h=="1",]),nrow(d6),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d7[d7$outcome_h=="1",]),nrow(d7),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d8[d8$outcome_h=="1",]),nrow(d8),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d9[d9$outcome_h=="1",]),nrow(d9),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d10[d10$outcome_h=="1",]),nrow(d10),p = NULL, alternative = "two.sided",correct = TRUE)




#### svm model k-fold cross validation
set.seed(1234)   
folds <- createMultiFolds(y=data_smo2$outcome_h,k=5,times=2)   

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
  
  data.train <- data_smo2[folds[[i]],]
  data.test <- data_smo2[-folds[[i]],]
  
  # Outcome <- "outcome_h"
  
  # CandidateVariables <- c("AST","NT_proBNP","TT","TnI","GLU","ALB","FBG","CK","Age","CHA2DS2_VACS") # top 10
  
  # CandidateVariables <- c("AST","NT_proBNP","TT","TnI","GLU","ALB","FBG","CK","Age","CHA2DS2_VACS",
  #                         "LDH","BMI","HR","DBP","AF_category","Statins","DD","HAS_BLED","Ever.smoking","LVESD") # top 20
  
  # CandidateVariables <- c("AST","NT_proBNP","TT","TnI","GLU","ALB","FBG","CK","Age","CHA2DS2_VACS",
  #                         "LDH","BMI","HR","DBP","AF_category","Statins","DD","HAS_BLED","Ever.smoking","LVESD",
  #                         "β.blocker","CCB","Antihypertensive.agents","TSH","Warfarin","Hyperlipidemia","UA","Ccr","HF","Diuretics")  # top30
  
  # CandidateVariables <- c("AST","NT_proBNP","TT","TnI","GLU","ALB","FBG","CK","Age","CHA2DS2_VACS",
  #                         "LDH","BMI","HR","DBP","AF_category","Statins","DD","HAS_BLED","Ever.smoking","LVESD",
  #                         "β.blocker","CCB","Antihypertensive.agents","TSH","Warfarin","Hyperlipidemia","UA","Ccr","HF","Diuretics",
  #                         "Weight","LAD","LVEDD","Hypertension","Ever.drinking","Heparin","Prior.PCI","Rivaroxaban","ACEI_ARB","Dabigatran")  # top40

  # Formula <- formula(paste(paste(Outcome,"~", collapse=" "),
  #                          paste(CandidateVariables, collapse=" + ")))
  
  grid = expand.grid(cost=10)
  
  cntrl = trainControl(method = "cv",number = 5)
  
  # model.svm = train(Formula,data = data.train,trControl = cntrl,tuneGrid = grid,method = "svmLinear2",
  #                   metric = "Kappa",probability = TRUE)
  
  model.svm = train(outcome_h~.,data = data.train,trControl = cntrl,tuneGrid = grid,method = "svmLinear2",
                    metric = "Kappa",probability = TRUE)  ## all feature
  
  data.train$p_prediction <- predict(model.svm, data.train, type="prob")[,2]
  data.test$p_prediction <- predict(model.svm, data.test, type="prob")[,2]
  pre_cutoff<-optimal.cutpoints(X = "p_prediction",status = "outcome_h", tag.healthy = 0, methods = "Youden",
                                data = data.train,ci.fit = TRUE,conf.level = 0.95)
  
  roc.train <- roc(data.train$outcome_h, data.train$p_prediction)
  roc.test <- roc(data.test$outcome_h, data.test$p_prediction)
  
  cm_train <- caret::confusionMatrix(as.factor(as.numeric(data.train$p_prediction>pre_cutoff$Youden$Global$optimal.cutoff$cutoff  )), as.factor(data.train$outcome_h), positive = "1")
  cm_test <- caret::confusionMatrix(as.factor(as.numeric(data.test$p_prediction>pre_cutoff$Youden$Global$optimal.cutoff$cutoff )), as.factor(data.test$outcome_h), positive = "1") 
  
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
  
  brier_train[i] <- mean((data.train$p_prediction-(as.numeric(data.train$outcome_h)-1))^2)
  brier_test[i] <- mean((data.test$p_prediction-(as.numeric(data.test$outcome_h)-1))^2)
  
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




#### random forest
set.seed(1234)
index = sample(nrow(data_smo2),round(0.7*nrow(data_smo2)), )
data_rf2_train = data_smo2[index,]
data_rf2_test = data_smo2[-index,]

fit_smo_rf2 <- randomForest(outcome_h~.,data = data_rf2_train,importance = TRUE)
fit_smo_rf2

# training set (1)
data_rf2_train$pred <- predict(fit_smo_rf2, data_rf2_train,type="prob")[,2]
roc_rf2_train <- roc(data_rf2_train$outcome_h, as.numeric(data_rf2_train$pred),ci=TRUE,print.auc = TRUE) #levels = c(0,1), direction = "<")
roc_rf2_train
# plot(roc_rf2_train,ci=TRUE,print.auc = TRUE,lty=2)

pre_cutoff<-optimal.cutpoints(X = "pred",status = "outcome_h", tag.healthy = 0, methods = "Youden",
                              data = data_rf2_train,ci.fit = TRUE,conf.level = 0.95)
summary(pre_cutoff)

caret::confusionMatrix(as.factor(as.numeric(data_rf2_train$pred>0.65)), as.factor(data_rf2_train$outcome_h), positive = "1")

# test set (0.9996)
data_rf2_test$pred <- predict(fit_smo_rf2, newdata = data_rf2_test,type="prob")[,2]
roc_rf2_test <- roc(data_rf2_test$outcome_h, as.numeric(data_rf2_test$pred),ci=TRUE,print.auc = TRUE) #levels = c(0,1), direction = "<")
roc_rf2_test
# plot(roc_rf2_test,ci=TRUE,print.auc = TRUE,lty=2)
caret::confusionMatrix(as.factor(as.numeric(data_rf2_test$pred>0.65)), as.factor(data_rf2_test$outcome_h), positive = "1") 

### feature importance
importance_smo_rf2 <- data.frame(fit_smo_rf2$importance)
head(importance_smo_rf2)
importance_smo_rf2 <- importance_smo_rf2[order(importance_smo_rf2$MeanDecreaseAccuracy,decreasing = TRUE),]


## calibration plot
data_rf2_cal<-subset(data_rf2_test,select = c(outcome_h,pred))
summary(data_rf2_cal)
data_rf2_cal$outcome_h<-as.factor(data_rf2_cal$outcome_h)

d1<-data_rf2_cal[which(data_rf2_cal$pred >=0 & data_rf2_cal$pred <0.1),]
d2<-data_rf2_cal[which(data_rf2_cal$pred >=0.1 & data_rf2_cal$pred <0.2),]
d3<-data_rf2_cal[which(data_rf2_cal$pred >=0.2 & data_rf2_cal$pred <0.3),]
d4<-data_rf2_cal[which(data_rf2_cal$pred >=0.3 & data_rf2_cal$pred <0.4),]
d5<-data_rf2_cal[which(data_rf2_cal$pred >=0.4 & data_rf2_cal$pred <0.5),]
d6<-data_rf2_cal[which(data_rf2_cal$pred >=0.5 & data_rf2_cal$pred <0.6),]
d7<-data_rf2_cal[which(data_rf2_cal$pred >=0.6 & data_rf2_cal$pred <0.7),]
d8<-data_rf2_cal[which(data_rf2_cal$pred >=0.7 & data_rf2_cal$pred <0.8),]
d9<-data_rf2_cal[which(data_rf2_cal$pred >=0.8 & data_rf2_cal$pred <0.9),]
d10<-data_rf2_cal[which(data_rf2_cal$pred >=0.9 & data_rf2_cal$pred <=1),]

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

summary(d1$outcome_h)

prop.test(nrow(d1[d1$outcome_h=="1",]),nrow(d1),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d2[d2$outcome_h=="1",]),nrow(d2),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d3[d3$outcome_h=="1",]),nrow(d3),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d4[d4$outcome_h=="1",]),nrow(d4),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d5[d5$outcome_h=="1",]),nrow(d5),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d6[d6$outcome_h=="1",]),nrow(d6),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d7[d7$outcome_h=="1",]),nrow(d7),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d8[d8$outcome_h=="1",]),nrow(d8),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d9[d9$outcome_h=="1",]),nrow(d9),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d10[d10$outcome_h=="1",]),nrow(d10),p = NULL, alternative = "two.sided",correct = TRUE)



###############################################################################################################
## random forest (k-fold cross validation)
set.seed(1234)   
folds <- createMultiFolds(y=data_smo2$outcome_h,k=5,times=10)   

c_train <- 0
acc_train <- 0
se_train <- 0
spe_train <- 0
ppv_train <- 0
npv_train <- 0
cutoff <- 0

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
  
  data.train <- data_smo2[folds[[i]],]
  data.test <- data_smo2[-folds[[i]],]
  
  # Outcome <- "outcome_h"
  
  # CandidateVariables <- c("AST","TnI","NT_proBNP","HAS_BLED", "TT","CHA2DS2_VACS","GLU","ALB", "HR","BMI")  # top10
  
  # CandidateVariables <- c("AST","TnI","NT_proBNP","HAS_BLED", "TT","CHA2DS2_VACS","GLU","ALB", "HR","BMI",
  #                         "FBG","Age","LVESD","LVEDD","CK","UA","LVEF","Weight","LDH","DBP")  # top20
  
  # CandidateVariables <- c("AST","TnI","NT_proBNP","HAS_BLED", "TT","CHA2DS2_VACS","GLU","ALB", "HR","BMI",
  #                         "FBG","Age","LVESD","LVEDD","CK","UA","LVEF","Weight","LDH","DBP",
  #                         "Ccr","SBP","DD","PTINR","LAD","TSH","CREA","Hyperlipidemia","Ever.smoking","Statins")  # top30
  
  # CandidateVariables <- c("AST","TnI","NT_proBNP","HAS_BLED", "TT","CHA2DS2_VACS","GLU","ALB", "HR","BMI",
  #                         "FBG","Age","LVESD","LVEDD","CK","UA","LVEF","Weight","LDH","DBP",
  #                         "Ccr","SBP","DD","PTINR","LAD","TSH","CREA","Hyperlipidemia","Ever.smoking","Statins",
  #                         "AF_category","Hypertension","Warfarin","HF","Prior.PCI","Diuretics","Heparin","Antihypertensive.agents","Gender","CCB")  # top40
  
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
  
  cutoff[i] <- pre_cutoff$Youden$Global$optimal.cutoff$cutoff
  c_test[i] <- roc.test$auc
  acc_test[i] <- cm_test$overall
  se_test[i] <- cm_test$byClass[1]
  spe_test[i] <- cm_test$byClass[2]
  ppv_test[i] <- cm_test$byClass[3]
  npv_test[i] <- cm_test$byClass[4]
  brier_train[i] <- mean((data.train$p_prediction-(as.numeric(data.train$outcome_h)-1))^2)
  brier_test[i] <- mean((data.test$p_prediction-(as.numeric(data.test$outcome_h)-1))^2)
  
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



############# gbm model
set.seed(1234)
index = sample(nrow(data_smo2),round(0.7*nrow(data_smo2)), )
data_gbm2_train = data_smo2[index,]
data_gbm2_test = data_smo2[-index,]

## hyper-parameter selection
data_gbm2_train$outcome_h <- ifelse(data_gbm2_train$outcome_h == "1","yes","no")
data_gbm2_test$outcome_h <- ifelse(data_gbm2_test$outcome_h == "1","yes","no")
data_gbm2_train$outcome_h <- ifelse(data_gbm2_train$outcome_h == "yes","1","0")
data_gbm2_test$outcome_h <- ifelse(data_gbm2_test$outcome_h == "yes","1","0")

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
train.gbm2 = train(
  outcome_h ~.,
  data = data_gbm2_train,
  trControl = cntrl,
  tuneGrid = grid,
  method = "gbm",
  metric = "ROC"
  # metric = "Kappa"
)

train.gbm2

## feature importance
gbmImp2 <-varImp(train.gbm2,n.trees = 500, interaction.depth =9, shrinkage = 0.2,
                 n.minobsinnode = 20)
plot(gbmImp2, top = 40)
gbmImp2 <- gbmImp2$importance
# gbmImp2 <- gbmImp2[order(gbmImp2$Overall,decreasing = TRUE),]


### model fit
grid = expand.grid(interaction.depth = 9, n.trees = 500, shrinkage = 0.2,n.minobsinnode = 20)
cntrl = trainControl(method = "cv",number = 5)

fit_gbm2 = train(
  outcome_h ~ . ,
  data = data_gbm2_train,
  trControl = cntrl,
  tuneGrid = grid,
  method = "gbm",
  metric = "Kappa"
)

# training set (1)
data_gbm2_train$pred <- predict(fit_gbm2, data_gbm2_train,type = "prob")[,2]
roc_gbm2_train <- roc(data_gbm2_train$outcome_h, data_gbm2_train$pred,ci=TRUE,print.auc = TRUE,levels = c(0,1), direction = "<")
roc_gbm2_train
# plot(roc_gbm2_train,ci=TRUE,print.auc = TRUE,lty=2)

pre_cutoff<-optimal.cutpoints(X = "pred",status = "outcome_h", tag.healthy = 0, methods = "Youden",
                              data = data_gbm2_train,ci.fit = TRUE,conf.level = 0.95)
summary(pre_cutoff)

caret::confusionMatrix(as.factor(as.numeric(data_gbm2_train$pred>0.9999994)), as.factor(data_gbm2_train$outcome_h), positive = "1")

# test set (0.9997)
data_gbm2_test$pred <- predict(fit_gbm2, newdata = data_gbm2_test,type = "prob")[,2]
roc_gbm2_test <- roc(data_gbm2_test$outcome_h, data_gbm2_test$pred,ci=TRUE,print.auc = TRUE) #levels = c(0,1), direction = "<")
roc_gbm2_test
# plot(roc_gbm2_test,ci=TRUE,print.auc = TRUE,lty=2)

caret::confusionMatrix(as.factor(as.numeric(data_gbm2_test$pred>0.9999994)), as.factor(data_gbm2_test$outcome_h), positive = "1") 


## calibration plot
data_gbm2_cal<-subset(data_gbm2_test,select = c(outcome_h,pred))
summary(data_gbm2_cal)
data_gbm2_cal$outcome_h<-as.factor(data_gbm2_cal$outcome_h)

d1<-data_gbm2_cal[which(data_gbm2_cal$pred >=0 & data_gbm2_cal$pred <0.1),]
d2<-data_gbm2_cal[which(data_gbm2_cal$pred >=0.1 & data_gbm2_cal$pred <0.2),]
d3<-data_gbm2_cal[which(data_gbm2_cal$pred >=0.2 & data_gbm2_cal$pred <0.3),]
d4<-data_gbm2_cal[which(data_gbm2_cal$pred >=0.3 & data_gbm2_cal$pred <0.4),]
d5<-data_gbm2_cal[which(data_gbm2_cal$pred >=0.4 & data_gbm2_cal$pred <0.5),]
d6<-data_gbm2_cal[which(data_gbm2_cal$pred >=0.5 & data_gbm2_cal$pred <0.6),]
d7<-data_gbm2_cal[which(data_gbm2_cal$pred >=0.6 & data_gbm2_cal$pred <0.7),]
d8<-data_gbm2_cal[which(data_gbm2_cal$pred >=0.7 & data_gbm2_cal$pred <0.8),]
d9<-data_gbm2_cal[which(data_gbm2_cal$pred >=0.8 & data_gbm2_cal$pred <0.9),]
d10<-data_gbm2_cal[which(data_gbm2_cal$pred >=0.9 & data_gbm2_cal$pred <=1),]

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

summary(d1$outcome_h)

prop.test(nrow(d1[d1$outcome_h=="1",]),nrow(d1),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d2[d2$outcome_h=="1",]),nrow(d2),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d3[d3$outcome_h=="1",]),nrow(d3),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d4[d4$outcome_h=="1",]),nrow(d4),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d5[d5$outcome_h=="1",]),nrow(d5),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d6[d6$outcome_h=="1",]),nrow(d6),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d7[d7$outcome_h=="1",]),nrow(d7),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d8[d8$outcome_h=="1",]),nrow(d8),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d9[d9$outcome_h=="1",]),nrow(d9),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d10[d10$outcome_h=="1",]),nrow(d10),p = NULL, alternative = "two.sided",correct = TRUE)



###############################################################################################################
## gbm model (k-fold cross validation)
set.seed(1234)   
folds <- createMultiFolds(y=data_smo2$outcome_h,k=5,times=2)   

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
  
  data.train <- data_smo2[folds[[i]],]
  data.test <- data_smo2[-folds[[i]],]
  
  grid = expand.grid(interaction.depth = 9, n.trees = 500, shrinkage = 0.2,n.minobsinnode = 20)
  
  cntrl = trainControl(method = "cv",number = 5)
  
  model.gbm = train(
    # outcome_h ~ TnI+AST+GLU+NT_proBNP+TT+ALB+FBG+HAS_BLED+SBP+HR, # top10
    
    # outcome_h ~ TnI+AST+GLU+NT_proBNP+TT+ALB+FBG+HAS_BLED+SBP+HR
    # +LVEDD+CHA2DS2_VACS+Hyperlipidemia+Age+CK+LVEF+BMI+LVESD+UA+DD, # top20
    
    # outcome_h ~ TnI+AST+GLU+NT_proBNP+TT+ALB+FBG+HAS_BLED+SBP+HR
    # +LVEDD+CHA2DS2_VACS+Hyperlipidemia+Age+CK+LVEF+BMI+LVESD+UA+DD+
    #   LDH+PTINR+Weight+TSH+AF_category+Ever.smoking+Ccr+Prior.RFA+Gender+Statins, # top30
    
    # outcome_h ~ TnI+AST+GLU+NT_proBNP+TT+ALB+FBG+HAS_BLED+SBP+HR
    # +LVEDD+CHA2DS2_VACS+Hyperlipidemia+Age+CK+LVEF+BMI+LVESD+UA+DD+
    #   LDH+PTINR+Weight+TSH+AF_category+Ever.smoking+Ccr+Prior.RFA+Gender+Statins+
    #   CREA+DBP+CCB+LAD+Diuretics+CHD+HF+Anticoagulants+Prior.PCI+ACEI_ARB, # top40
    
    outcome_h ~ .,    # all feature
    data = data.train,
    trControl = cntrl,
    tuneGrid = grid,
    method = "gbm",
    metric = "Kappa")
  
  data.train$p_prediction <- predict(model.gbm, data.train, type="prob")[,2]
  data.test$p_prediction <- predict(model.gbm, data.test, type="prob")[,2]
  
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
  
  cutoff[i] <- pre_cutoff$Youden$Global$optimal.cutoff$cutoff
  c_test[i] <- roc.test$auc
  acc_test[i] <- cm_test$overall
  se_test[i] <- cm_test$byClass[1]
  spe_test[i] <- cm_test$byClass[2]
  ppv_test[i] <- cm_test$byClass[3]
  npv_test[i] <- cm_test$byClass[4]
  brier_train[i] <- mean((data.train$p_prediction-(as.numeric(data.train$outcome_h)-1))^2)
  brier_test[i] <- mean((data.test$p_prediction-(as.numeric(data.test$outcome_h)-1))^2)
  
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




### xgboost model
setwd("/Users/weiyuna/Desktop/工作/合作项目/上海胸科/20211103导出数据/房颤/补充结局-不包含瓣膜手术")
# setwd("/Users/air/Desktop/工作")
data_smo2_xgb<-read.csv("data_smo2.csv")
data_smo2_xgb<-subset(data_smo2_xgb,select = c(outcome_h:LAD,LVEF:UA,TT:CREA,Ccr:CK,NT_proBNP:Prior.CABG))
names(data_smo2_xgb)

data_smo2_xgb$Persistent_AF<-ifelse(data_smo2_xgb$AF_category=="2",1,0)
data_smo2_xgb$Chronic_AF<-ifelse(data_smo2_xgb$AF_category=="3",1,0)
names(data_smo2_xgb)
str(data_smo2_xgb)


### hyper-parameter selection
set.seed(1234)
index = sample(nrow(data_smo2_xgb),round(0.7*nrow(data_smo2_xgb)), )
data.train.xg = data_smo2_xgb[index,]
data.test.xg = data_smo2_xgb[-index,]
summary(data.train.xg)
summary(data.test.xg)

data.train.xg$outcome_h <- ifelse(data.train.xg$outcome_h == "1","yes","no")
data.test.xg$outcome_h <- ifelse(data.test.xg$outcome_h == "1","yes","no")

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
index = sample(nrow(data_smo2_xgb),round(0.7*nrow(data_smo2_xgb)), )
data.train.xg = data_smo2_xgb[index,]
data.test.xg = data_smo2_xgb[-index,]

## data preprocessing
data.train.xg1<-data.matrix(data.train.xg[,c(3:64)])
data.train.xg2<-Matrix(data.train.xg1,sparse = T)
data.train.xg_y2<-data.train.xg$outcome_h
traindata.xg<-list(data=data.train.xg2,label=data.train.xg_y2)
dtrain<-xgb.DMatrix(data = traindata.xg$data,label=traindata.xg$label)

data.test.xg1<-data.matrix(data.test.xg[,c(3:64)])
data.test.xg2<-Matrix(data.test.xg1,sparse = T)
data.test.xg_y2<-data.test.xg$outcome_h
testdata.xg<-list(data=data.test.xg2,label=data.test.xg_y2)
dtest<-xgb.DMatrix(data = testdata.xg$data,label=testdata.xg$label)

fit_xgb2 <-xgboost(data = dtrain, objective="binary:logistic", booster="gbtree",nrounds = 300, max_depth = 7, eta = 0.05, gamma = 0.1, colsample_bytree =
                     0.5, min_child_weight = 1 , subsample = 0.7)

### feature importance
importance.xgb2<-xgb.importance(data.train.xg2@Dimnames[[2]],model = fit_xgb2)
# xgb.ggplot.importance(importance.xgb2,rel_to_first = TRUE,n_clusters = 3, measure = "Cover")
# xgb.ggplot.importance(importance.xgb2,rel_to_first = TRUE,n_clusters = 3, measure = "Gain")
xgb.plot.importance(importance.xgb2,rel_to_first = TRUE,xlab = "Relative importance")
# xgb.ggplot.shap.summary(data = traindata.xg$data, model = fit_xgb1,target_class = 1,top_n = 20)


## predict
pre.xgb.train2<-predict(fit_xgb2,dtrain)
pre.xgb.test2<-predict(fit_xgb2,newdata = dtest)

pre_cutoff<-optimal.cutpoints(X = "pre.xgb.train2",status = "data.train.xg_y2", tag.healthy = 0, methods = "Youden",
                              data = data.frame(cbind(pre.xgb.train2,data.train.xg_y2)),ci.fit = TRUE,conf.level = 0.95)
summary(pre_cutoff)

## model evaluation
# training set (1)
xgb.cf.train2<-caret::confusionMatrix(as.factor(as.numeric(pre.xgb.train2>0.7321888)),as.factor(data.train.xg_y2), positive = "1")
xgb.cf.train2
roc_xgb2_train <- roc(response = data.train.xg_y2, predictor = pre.xgb.train2,ci=TRUE,print.auc = TRUE)
roc_xgb2_train
# plot(roc_xgb2_train,ci=TRUE,print.auc = TRUE,lty=2)

# test set (0.9984)
xgb.cf.test2<-caret::confusionMatrix(as.factor(as.numeric(pre.xgb.test2>0.7321888)),as.factor(data.test.xg_y2), positive = "1")
xgb.cf.test2
roc_xgb2_test <- roc(response = data.test.xg_y2, predictor = pre.xgb.test2,ci=TRUE,print.auc = TRUE)
roc_xgb2_test
# plot(roc_xgb2_test,ci=TRUE,print.auc = TRUE,lty=2)

# brier score(0.058)
val_xgb2 <- val.prob(pre.xgb.test2,data.test.xg_y2,cex = 0.8)

## calibration plot
data_xgb2_cal<-data.frame(cbind(pre.xgb.test2,data.test.xg_y2))
summary(data_xgb2_cal)
data_xgb2_cal$data.test.xg_y2<-as.factor(data_xgb2_cal$data.test.xg_y2)

d1<-data_xgb2_cal[which(data_xgb2_cal$pre.xgb.test2 >=0 & data_xgb2_cal$pre.xgb.test2 <0.1),]
d2<-data_xgb2_cal[which(data_xgb2_cal$pre.xgb.test2 >=0.1 & data_xgb2_cal$pre.xgb.test2 <0.2),]
d3<-data_xgb2_cal[which(data_xgb2_cal$pre.xgb.test2 >=0.2 & data_xgb2_cal$pre.xgb.test2 <0.3),]
d4<-data_xgb2_cal[which(data_xgb2_cal$pre.xgb.test2 >=0.3 & data_xgb2_cal$pre.xgb.test2 <0.4),]
d5<-data_xgb2_cal[which(data_xgb2_cal$pre.xgb.test2 >=0.4 & data_xgb2_cal$pre.xgb.test2 <0.5),]
d6<-data_xgb2_cal[which(data_xgb2_cal$pre.xgb.test2 >=0.5 & data_xgb2_cal$pre.xgb.test2 <0.6),]
d7<-data_xgb2_cal[which(data_xgb2_cal$pre.xgb.test2 >=0.6 & data_xgb2_cal$pre.xgb.test2 <0.7),]
d8<-data_xgb2_cal[which(data_xgb2_cal$pre.xgb.test2 >=0.7 & data_xgb2_cal$pre.xgb.test2 <0.8),]
d9<-data_xgb2_cal[which(data_xgb2_cal$pre.xgb.test2 >=0.8 & data_xgb2_cal$pre.xgb.test2 <0.9),]
d10<-data_xgb2_cal[which(data_xgb2_cal$pre.xgb.test2 >=0.9 & data_xgb2_cal$pre.xgb.test2 <=1),]

t.test(d1$pre.xgb.test2)
t.test(d2$pre.xgb.test2)
t.test(d3$pre.xgb.test2)
t.test(d4$pre.xgb.test2)
t.test(d5$pre.xgb.test2)
t.test(d6$pre.xgb.test2)
t.test(d7$pre.xgb.test2)
t.test(d8$pre.xgb.test2)
t.test(d9$pre.xgb.test2)
t.test(d10$pre.xgb.test2)

summary(d1$data.test.xg_y2)

prop.test(nrow(d1[d1$data.test.xg_y2=="1",]),nrow(d1),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d2[d2$data.test.xg_y2=="1",]),nrow(d2),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d3[d3$data.test.xg_y2=="1",]),nrow(d3),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d4[d4$data.test.xg_y2=="1",]),nrow(d4),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d5[d5$data.test.xg_y2=="1",]),nrow(d5),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d6[d6$data.test.xg_y2=="1",]),nrow(d6),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d7[d7$data.test.xg_y2=="1",]),nrow(d7),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d8[d8$data.test.xg_y2=="1",]),nrow(d8),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d9[d9$data.test.xg_y2=="1",]),nrow(d9),p = NULL, alternative = "two.sided",correct = TRUE)
prop.test(nrow(d10[d10$data.test.xg_y2=="1",]),nrow(d10),p = NULL, alternative = "two.sided",correct = TRUE)


###############################################################################################################
## xgboost model (k-fold cross validation)
set.seed(1234)
folds <- createMultiFolds(y=data_smo2_xgb$outcome_h,k=5,times=5)   

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

for (i in 1:25){
  data.train.xg <- data_smo2_xgb[folds[[i]],]
  data.test.xg <- data_smo2_xgb[-folds[[i]],]
  
  
  ## data preprocessing
  # data.train.xg1<-data.matrix(data.train.xg[,c("TnI", "AST","HAS_BLED","GLU","TT","NT_proBNP","ALB","BMI","LVESD","HR")])  ##top10
  
  # data.train.xg1<-data.matrix(data.train.xg[,c("TnI", "AST","HAS_BLED","GLU","TT","NT_proBNP","ALB","BMI","LVESD","HR",
  #                                              "CHA2DS2_VACS","CK","LVEDD","Age","LVEF","Hyperlipidemia","FBG","DD","UA","SBP")])  ##top20
  # 
  # data.train.xg1<-data.matrix(data.train.xg[,c("TnI", "AST","HAS_BLED","GLU","TT","NT_proBNP","ALB","BMI","LVESD","HR",
  #                                              "CHA2DS2_VACS","CK","LVEDD","Age","LVEF","Hyperlipidemia","FBG","DD","UA","SBP",
  #                                              "LDH","PTINR","TSH","Ccr","Weight","CREA","DBP","LAD","HF","Warfarin")])  ##top30
  # 
  # data.train.xg1<-data.matrix(data.train.xg[,c("TnI", "AST","HAS_BLED","GLU","TT","NT_proBNP","ALB","BMI","LVESD","HR",
  #                                              "CHA2DS2_VACS","CK","LVEDD","Age","LVEF","Hyperlipidemia","FBG","DD","UA","SBP",
  #                                              "LDH","PTINR","TSH","Ccr","Weight","CREA","DBP","LAD","HF","Warfarin",
  #                                              "Chronic_AF","Prior.PCI","β.blocker","Heparin","Anticoagulants","Ever.smoking","Prior.RFA","ACEI_ARB","PAD","Gender")])  ##top40
  # 
  data.train.xg1<-data.matrix(data.train.xg[,c(3:64)])  ## all feature
  
  data.train.xg2<-Matrix(data.train.xg1,sparse = T)
  data.train.xg_y<-data.train.xg$outcome_h
  traindata.xg<-list(data=data.train.xg2,label=data.train.xg_y)
  dtrain<-xgb.DMatrix(data = traindata.xg$data,label=traindata.xg$label)
  
  # data.test.xg1<-data.matrix(data.test.xg[,c("TnI", "AST","HAS_BLED","GLU","TT","NT_proBNP","ALB","BMI","LVESD","HR")])  ##top10
  
  # data.test.xg1<-data.matrix(data.test.xg[,c("TnI", "AST","HAS_BLED","GLU","TT","NT_proBNP","ALB","BMI","LVESD","HR",
  #                                            "CHA2DS2_VACS","CK","LVEDD","Age","LVEF","Hyperlipidemia","FBG","DD","UA","SBP")])  ##top20
  # 
  # data.test.xg1<-data.matrix(data.test.xg[,c("TnI", "AST","HAS_BLED","GLU","TT","NT_proBNP","ALB","BMI","LVESD","HR",
  #                                            "CHA2DS2_VACS","CK","LVEDD","Age","LVEF","Hyperlipidemia","FBG","DD","UA","SBP",
  #                                            "LDH","PTINR","TSH","Ccr","Weight","CREA","DBP","LAD","HF","Warfarin")])  ##top30
  # 
  # data.test.xg1<-data.matrix(data.test.xg[,c("TnI", "AST","HAS_BLED","GLU","TT","NT_proBNP","ALB","BMI","LVESD","HR",
  #                                            "CHA2DS2_VACS","CK","LVEDD","Age","LVEF","Hyperlipidemia","FBG","DD","UA","SBP",
  #                                            "LDH","PTINR","TSH","Ccr","Weight","CREA","DBP","LAD","HF","Warfarin",
  #                                            "Chronic_AF","Prior.PCI","β.blocker","Heparin","Anticoagulants","Ever.smoking","Prior.RFA","ACEI_ARB","PAD","Gender")])  ##top40
  
  data.test.xg1<-data.matrix(data.test.xg[,c(3:64)])  ## all feature
  
  data.test.xg2<-Matrix(data.test.xg1,sparse = T)
  data.test.xg_y<-data.test.xg$outcome_h
  testdata.xg<-list(data=data.test.xg2,label=data.test.xg_y)
  dtest<-xgb.DMatrix(data = testdata.xg$data,label=testdata.xg$label)
  
  model_xgb_jiye <-xgboost(data = dtrain,objective="binary:logistic", booster="gbtree",nrounds = 300, max_depth = 7, eta = 0.05, gamma = 0.1, colsample_bytree =
                             0.5, min_child_weight = 1 , subsample = 0.7)
  
  ### feature importance
  # importance_fuhe<-xgb.importance(data.train.xg2@Dimnames[[2]],model = model_xgb_jiye)
  # xgb.ggplot.importance(importance_fuhe,rel_to_first = TRUE,n_clusters = 3, measure = "Cover")
  # xgb.ggplot.importance(importance_fuhe,rel_to_first = TRUE,n_clusters = 3, measure = "Gain",top_n = 20)
  # xgb.ggplot.shap.summary(data = traindata.xg$data, model = model_xgb_jiye,target_class = 1,top_n = 20)
  
  ## predict
  pre.xgb.train <-predict(model_xgb_jiye,dtrain)
  pre.xgb.test <-predict(model_xgb_jiye,newdata = dtest)
  
  pre_cutoff<-optimal.cutpoints(X = "pre.xgb.train",status = "data.train.xg_y", tag.healthy = 0, methods = "Youden",
                                data = data.frame(cbind(pre.xgb.train,data.train.xg_y)),ci.fit = TRUE,conf.level = 0.95)
  summary(pre_cutoff)
  
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




## ROC curves of four models
par(mfrow=c(1,1))
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
# legend("topright",legend = "B", text.font = 2,box.lwd = "none")

