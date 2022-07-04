setwd("/Users/weiyuna/Desktop/工作/合作项目/上海胸科/20211103导出数据/房颤/补充结局-不包含瓣膜手术")
library(epiDisplay)
library(mice)

data<-read.csv("AF_new.csv")
data<-subset(data,select = c(patient_id,outcome:UA_factor,TT:Prior.CABG))
factor.name<-c("outcome","outcome_h","outcome_b", "AF_category", "Gender", "HF","UA_factor","CREA_factor","CK_factor","Aspirin","Clopidogrel","Other.antiplatelet.agents",
               "Antiplatelet.agents","Warfarin","Dabigatran","Rivaroxaban","Heparin", "Anticoagulants","Statins",
               "ACEI_ARB","β.blocker","Diuretics","CCB","Antihypertensive.agents","Ever.smoking","Ever.drinking","Angina","Heart.failure",
               "Stroke","PAD","COPD","Hypertension","Diabetes","Hyperlipidemia","MI","CHD", "CKD","Prior.RFA","Prior.PCI","Prior.CABG")

for (i in factor.name) {
  data[,i]<-as.factor(data[,i])
}

# result1<-tableStack(vars = AF_category:Prior.CABG,by= "none",dataFrame = data)
# write.csv(result1,file = "table1.csv")
# result2<-tableStack(vars = AF_category:Prior.CABG,by= outcome,dataFrame = data)
# write.csv(result2,file = "table2 (any complication).csv")
# result3<-tableStack(vars = AF_category:Prior.CABG,by= outcome_h,dataFrame = data)
# write.csv(result3,file = "table3 (cardiac effusion).csv")
# result4<-tableStack(vars = AF_category:Prior.CABG,by= outcome_b,dataFrame = data)
# write.csv(result4,file = "table4 (hemorrhage).csv")


#######################################################################################################
####data imputation
### remove outcome from the full dataset
data1<-subset(data,select=c(patient_id,AF_category:Prior.CABG))
names(data1)
data1$patient_id<-as.character(data1$patient_id)

### define column name of imputed data
varlist<-names(data1)
varlist

### define the imputation dataset
data1.impu = data1[varlist]
data1.impu
data1.impu<-mice(data1.impu,m=20,seed = 1234)
summary(data1.impu)

### save one of the imputated results 
data2<-complete(data1.impu,action = 10)
# write.csv(data2,file = "data_imputation.csv")

### combining the imputation data and outcomes as a whole
data_out<-subset(data,select = c(patient_id,outcome,outcome_h,outcome_b))
data_out$patient_id<-as.character(data_out$patient_id)
data3<-merge(data_out,data2,by="patient_id")
# write.csv(data3,file = "data_imputation_full.csv")

### manual process the LAD/UA/CREA/CK (the normal range is different between men and women)
data3<-read.csv("data_imputation_full.csv")
data3<-subset(data3,select = c(patient_id,outcome:Prior.CABG))

factor.name<-c("outcome","outcome_h","outcome_b", "AF_category", "Gender","LAD_factor","HF","UA_factor","CREA_factor","CK_factor","Aspirin","Clopidogrel","Other.antiplatelet.agents",
               "Antiplatelet.agents","Warfarin","Dabigatran","Rivaroxaban","Heparin", "Anticoagulants","Statins",
               "ACEI_ARB","β.blocker","Diuretics","CCB","Antihypertensive.agents","Ever.smoking","Ever.drinking","Angina","Heart.failure",
               "Stroke","PAD","COPD","Hypertension","Diabetes","Hyperlipidemia","MI","CHD", "CKD","Prior.RFA","Prior.PCI","Prior.CABG")

for (i in factor.name) {
  data3[,i]<-as.factor(data3[,i])
}

### data transformation 
data3$Age_factor<-cut(data3$Age, c(-1000,64.9,1000), labels=c("<65","≥65"))
data3$Age_factor<-factor(data3$Age_factor,levels = c("<65","≥65"),labels = c(1,2))

data3$BMI_factor<-cut(data3$BMI, c(-1000,24.999,1000), labels=c("<25", "≥25"))
data3$BMI_factor<-factor(data3$BMI_factor,levels = c("<25", "≥25"),labels = c(1,2))

data3$HR_factor<-cut(data3$HR, c(-1000,99.9,1000), labels=c("<100","≥100"))
data3$HR_factor<-factor(data3$HR_factor,levels = c("<100","≥100"),labels = c(1,2))

data3$DBP_factor<-cut(data3$DBP, c(-1000,89.9,1000), labels=c("<90","≥90"))
data3$DBP_factor<-factor(data3$DBP_factor,levels = c("<90","≥90"),labels = c(1,2))

data3$SBP_factor<-cut(data3$SBP, c(-1000,139.9,1000), labels=c("<140","≥140"))
data3$SBP_factor<-factor(data3$SBP_factor,levels = c("<140","≥140"),labels = c(1,2))

data3$LVESD_factor<-cut(data3$LVESD, c(-1000,40.001,1000), labels=c("≤40",">40"))
data3$LVESD_factor<-factor(data3$LVESD_factor,levels = c("≤40",">40"),labels = c(1,2))

data3$LVEDD_factor<-cut(data3$LVEDD, c(-1000,55.001,1000), labels=c("≤55",">55"))
data3$LVEDD_factor<-factor(data3$LVEDD_factor,levels = c("≤55",">55"),labels = c(1,2))

data3$LVEF_factor<-cut(data3$LVEF, c(-1000,49.999,1000), labels=c("<50","≥50"))
data3$LVEF_factor<-factor(data3$LVEF_factor,levels = c("≥50","<50"),labels = c(1,2))

data3$HAS_BLED_factor<-cut(data3$HAS_BLED, c(-100,1.9,1000), labels=c("<2","≥2"))
data3$HAS_BLED_factor<-factor(data3$HAS_BLED_factor,levels = c("<2","≥2"),labels = c(1,2))

data3$CHA2DS2_VACS_factor<-cut(data3$CHA2DS2_VACS, c(-100,1.9,1000), labels=c("<2","≥2"))
data3$CHA2DS2_VACS_factor<-factor(data3$CHA2DS2_VACS_factor,levels = c("<2","≥2"),labels = c(1,2))

data3$TSH_factor<-cut(data3$TSH, c(-1000,5.3301,1000), labels=c("≤5.33",">5.33"))
data3$TSH_factor<-factor(data3$TSH_factor,levels = c("≤5.33",">5.33"),labels = c(1,2))

data3$FBG_factor<-cut(data3$FBG, c(-1000,1.999,4.001,1000), labels=c("<2","2-4",">4"))
data3$FBG_factor<-factor(data3$FBG_factor,levels = c("<2","2-4",">4"),labels = c(2,1,3))

data3$TT_factor<-cut(data3$TT, c(-1000,24.01,10000),labels = c("≤24",">24"))
data3$TT_factor<-factor(data3$TT_factor,levels = c("≤24",">24"),labels = c(1,2))

data3$PTINR_factor<-cut(data3$PTINR, c(-1000,1.5001,1000), labels=c("≤1.5",">1.5"))
data3$PTINR_factor<-factor(data3$PTINR_factor,levels = c("≤1.5", ">1.5"),labels = c(1,2))

data3$Ccr_factor<-cut(data3$Ccr, c(-1000,120.001,1000), labels=c("≤120", ">120"))
data3$Ccr_factor<-factor(data3$Ccr_factor,levels = c("≤120", ">120"),labels = c(1,2))

data3$DD_factor<-cut(data3$DD, c(-1000,0.5501,100), labels=c("≤0.55",">0.55"))
data3$DD_factor<-factor(data3$DD_factor,levels = c("≤0.55",">0.55"),labels = c(1,2))

data3$TnI_factor<-cut(data3$TnI, c(-1000,0.04001,1000), labels=c("≤0.04",">0.04"))
data3$TnI_factor<-factor(data3$TnI_factor,levels = c("≤0.04",">0.04"),labels = c(1,2))

data3$LDH_factor<-cut(data3$LDH, c(-1000,250.01,2000), labels=c("≤250",">250"))
data3$LDH_factor<-factor(data3$LDH_factor,levels = c("≤250",">250"),labels = c(1,2))

data3$AST_factor<-cut(data3$AST, c(-1000,40.01,1000), labels=c("≤40",">40"))
data3$AST_factor<-factor(data3$AST_factor,levels = c("≤40",">40"),labels = c(1,2))

data3$ALB_factor<-cut(data3$ALB, c(-1000,35.01,1000), labels=c("≤35",">35"))
data3$ALB_factor<-factor(data3$ALB_factor,levels = c(">35","≤35"),labels = c(1,2))

data3$GLU_factor<-cut(data3$GLU, c(-1000,6.101,1000), labels=c("≤6.1",">6.1"))
data3$GLU_factor<-factor(data3$GLU_factor,levels = c("≤6.1",">6.1"),labels = c(1,2))

data3$NT_proBNP_factor<-cut(data3$NT_proBNP, c(-1000,299.99,900.01,100000), labels=c("<300","300-900",">900"))
data3$NT_proBNP_factor<-factor(data3$NT_proBNP_factor,levels = c("300-900","<300",">900"),labels = c(1,2,3))

summary(data3)

