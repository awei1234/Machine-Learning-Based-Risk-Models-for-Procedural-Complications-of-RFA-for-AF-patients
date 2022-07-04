library(MASS)
library(pROC)

### stepwise logistic regression(raw data)
Outcome <- "outcome"
# Outcome <- "outcome_h"
# Outcome <- "outcome_b"

CandidateVariables <- c("AF_category", "Gender","Age_factor", "BMI_factor","HR_factor", "SBP_factor","LVESD_factor",
                        "LVEDD_factor","LAD_factor","LVEF_factor","HAS_BLED_factor","CHA2DS2_VACS_factor","TSH_factor","FBG_factor",
                        "UA_factor","TT_factor", "PTINR_factor","CREA_factor","Ccr_factor","DD_factor",
                        "TnI_factor","LDH_factor","AST_factor","ALB_factor", "GLU_factor","CK_factor", "NT_proBNP_factor","Aspirin","Clopidogrel",
                        "Rivaroxaban","Heparin", "Statins", "ACEI_ARB","β.blocker","Diuretics","CCB","Ever.smoking","Stroke","PAD",
                        "Hypertension","Diabetes","Hyperlipidemia","CHD","CKD","Prior.RFA","Prior.PCI")

Formula <- formula(paste(paste(Outcome,"~", collapse=" "),
                         paste(CandidateVariables, collapse=" + ")))

model.full <- glm(Formula, data=data3,family=binomial)
model.final <- stepAIC(model.full, direction="both")


fit.log.step.fuhe <- glm(outcome ~ Gender + Age_factor +  BMI_factor + HR_factor + LAD_factor + 
                           CHA2DS2_VACS_factor + CREA_factor + TnI_factor + AST_factor + 
                           NT_proBNP_factor, data=data3,family=binomial) 

fit.log.step.jiye <- glm(formula = outcome_h ~ Gender + Age_factor + BMI_factor  + AST_factor + 
                           ALB_factor + NT_proBNP_factor,data=data3, family=binomial)

fit.log.step.chuxue <- glm(outcome_b ~ Gender + AF_category + Age_factor + LAD_factor2 + 
                             CREA_factor + DD_factor + TnI_factor +AST_factor + Antiplatelet.agents + 
                             Hypertension + Diabetes , data=data3,family=binomial)




### stepwise logistic regression(smote data)
## any complication
model.full1 <- glm(outcome~., data=data_smo1,family=binomial)

model.final <- stepAIC(model.full1, direction="both")

fit_step_smo_fuhe <- glm(outcome ~ AF_category + Gender + Age + Weight + HR + DBP + SBP + 
                           LVESD + LVEDD + LAD + LVEF + HAS_BLED + CHA2DS2_VACS + FBG + 
                           UA + TT + PTINR + CREA + Ccr + DD + TnI + LDH + AST + ALB + 
                           GLU + CK + NT_proBNP + Aspirin + Clopidogrel + Dabigatran + 
                           Rivaroxaban + Statins + ACEI_ARB + β.blocker + Diuretics + 
                           CCB + Ever.smoking + Ever.drinking + Angina + Heart.failure + 
                           Stroke + PAD + COPD + Hypertension + Diabetes + Hyperlipidemia + 
                           MI + CKD + Prior.PCI + Prior.CABG, data=data_smo1,family=binomial) 


## cardiac effusion
model.full2 <- glm(outcome_h~., data=data_smo2,family=binomial)
model.final <- stepAIC(model.full2, direction="both")

fit_step_smo_jiye <- glm(outcome_h ~ AF_category + Gender + Age + BMI + HR + SBP + LVESD + 
                           LVEDD + LAD + LVEF + HF + HAS_BLED + CHA2DS2_VACS + FBG + 
                           UA + PTINR + Ccr + DD + LDH + AST + ALB + GLU + CK + NT_proBNP + 
                           Aspirin + Clopidogrel + Other.antiplatelet.agents + Warfarin + 
                           Dabigatran + Heparin + Anticoagulants + Statins + ACEI_ARB + 
                           β.blocker + Ever.smoking + Ever.drinking + Heart.failure + 
                           Stroke + PAD + COPD + Hypertension + Diabetes + Hyperlipidemia + 
                           MI + CHD + CKD + Prior.RFA + Prior.PCI + Prior.CABG, data=data_smo2,family=binomial) 


## hemorrhage
model.full3 <- glm(outcome_b~., data=data_smo3,family=binomial)
model.final <- stepAIC(model.full3, direction="both")

fit_step_smo_fuhe <- glm(outcome_b ~ AF_category + Age + Weight + BMI + HR + DBP + SBP + 
                           LVESD + LVEDD + LAD + CHA2DS2_VACS + FBG + UA + TT + CREA + 
                           Ccr + DD + TnI + LDH + AST + ALB + NT_proBNP + Aspirin + 
                           Clopidogrel + Other.antiplatelet.agents + Antiplatelet.agents + 
                           Warfarin + Dabigatran + Rivaroxaban + Heparin + Statins + 
                           ACEI_ARB + β.blocker + CCB + Antihypertensive.agents + Ever.smoking + 
                           Ever.drinking + Heart.failure + PAD + Hyperlipidemia + MI + 
                           CKD + Prior.RFA + Prior.PCI, data=data_smo3,family=binomial) 




