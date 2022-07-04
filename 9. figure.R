setwd("/Users/weiyuna/Desktop/工作/合作项目/上海胸科/20211103导出数据/房颤/补充结局-不包含瓣膜手术")
# install.packages("patchwork")
# install.packages("ggpubr")
install.packages("ggsci")
library(ggplot2)
library(patchwork)
library(ggpubr)
library(ggsci)


### machine learning model feature selection
ml_feature<-read.csv("feature selection_new.csv")
p1 <- ggplot(ml_feature[which(ml_feature$outcomes=="Any complication"),],
       mapping = aes(x=factor(features), y=AUC, colour= Model,group=Model)) +
  geom_errorbar(aes(ymin=AUC_lower,
                    ymax=AUC_upper), width=.1,position=position_dodge(0.1))+
  geom_line(aes(linetype=Model,color=Model))+
  geom_point(aes(shape=Model,color=Model))+
  scale_color_manual(values = c("#631879FF","#1B1919FF","#008280FF","#3B4992FF","#A20056FF"))+
  # scale_color_aaas()+
  theme_bw()+
  scale_y_continuous(limits = c(0.7,1),breaks = seq(0.7,1,0.05))+
  guides(color=FALSE,shape=FALSE,linetype=FALSE)+
  xlab("Number of features selected")+
  ylab("AUC")


p2 <- ggplot(ml_feature[which(ml_feature$outcomes=="Cardiac effusion"),],
       mapping = aes(x=factor(features), y=AUC, colour= Model,group=Model)) +
  geom_errorbar(aes(ymin=AUC_lower,
                    ymax=AUC_upper), width=.1,position=position_dodge(0.1))+
  geom_line(aes(linetype=Model,color=Model))+
  geom_point(aes(shape=Model,color=Model))+
  scale_color_manual(values = c("#631879FF","#1B1919FF","#008280FF","#3B4992FF","#A20056FF"))+
  # scale_color_aaas()+
  theme_bw()+
  scale_y_continuous(limits = c(0.7,1),breaks = seq(0.7,1,0.05))+
  guides(color=FALSE,shape=FALSE,linetype=FALSE)+
  xlab("Number of features selected")+
  ylab("AUC")

p3 <-ggplot(ml_feature[which(ml_feature$outcomes=="Hemorrhage"),],
       mapping = aes(x=factor(features), y=AUC, colour= Model,group=Model)) +
  geom_errorbar(aes(ymin=AUC_lower,
                    ymax=AUC_upper), width=.1,position=position_dodge(0.1))+
  geom_line(aes(linetype=Model,color=Model))+
  geom_point(aes(shape=Model,color=Model))+
  scale_color_manual(values = c("#631879FF","#1B1919FF","#008280FF","#3B4992FF","#A20056FF"))+
  # scale_color_aaas()+
  theme_bw()+
  scale_y_continuous(limits = c(0.7,1),breaks = seq(0.7,1,0.05))+
  guides(color=FALSE,shape=FALSE,linetype=FALSE)+
  xlab("Number of features selected")+
  ylab("AUC")

# ggpubr::ggarrange(p1,p2,p3,ncol = 3,nrow = 1, widths=c(1.5,1.5,2),
#                   align = "v",labels = c("A1","B1","C1"))

p4 <- ggplot(ml_feature[which(ml_feature$outcomes=="Any complication"),],
             mapping = aes(x=factor(features), y=Accuracy, colour= Model,group=Model)) +
  geom_errorbar(aes(ymin=Accuracy_lower,
                    ymax=Accuracy_upper), width=.1,position=position_dodge(0.1))+
  geom_line(aes(linetype=Model,color=Model))+
  geom_point(aes(shape=Model,color=Model))+
  scale_color_manual(values = c("#631879FF","#1B1919FF","#008280FF","#3B4992FF","#A20056FF"))+
  # scale_color_aaas()+
  theme_bw()+
  scale_y_continuous(limits = c(0.7,1),breaks = seq(0.7,1,0.05))+
  guides(color=FALSE,shape=FALSE,linetype=FALSE)+
  xlab("Number of features selected")+
  ylab("Accuracy")

p5 <- ggplot(ml_feature[which(ml_feature$outcomes=="Cardiac effusion"),],
             mapping = aes(x=factor(features), y=Accuracy, colour= Model,group=Model)) +
  geom_errorbar(aes(ymin=Accuracy_lower,
                    ymax=Accuracy_upper), width=.1,position=position_dodge(0.1))+
  geom_line(aes(linetype=Model,color=Model))+
  geom_point(aes(shape=Model,color=Model))+
  scale_color_manual(values = c("#631879FF","#1B1919FF","#008280FF","#3B4992FF","#A20056FF"))+
  # scale_color_aaas()+
  theme_bw()+
  scale_y_continuous(limits = c(0.7,1),breaks = seq(0.7,1,0.05))+
  guides(color=FALSE,shape=FALSE,linetype=FALSE)+
  xlab("Number of features selected")+
  ylab("Accuracy")

p6 <- ggplot(ml_feature[which(ml_feature$outcomes=="Hemorrhage"),],
             mapping = aes(x=factor(features), y=Accuracy, colour= Model,group=Model)) +
  geom_errorbar(aes(ymin=Accuracy_lower,
                    ymax=Accuracy_upper), width=.1,position=position_dodge(0.1))+
  geom_line(aes(linetype=Model,color=Model))+
  geom_point(aes(shape=Model,color=Model))+
  scale_color_manual(values = c("#631879FF","#1B1919FF","#008280FF","#3B4992FF","#A20056FF"))+
  # scale_color_aaas()+
  theme_bw()+
  scale_y_continuous(limits = c(0.7,1),breaks = seq(0.7,1,0.05))+
  # guides(color=FALSE,shape=FALSE,linetype=FALSE)+
  xlab("Number of features selected")+
  ylab("Accuracy")

# ggpubr::ggarrange(p4,p5,p6,ncol = 3,nrow = 1, widths=c(1.5,1.5,2.2),labels = c("A2","B2","C2"))
# ggpubr::ggarrange(p1,p2,p3,p4,p5,p6,ncol = 3,nrow = 2, widths=c(1.5,1.5,1.5),
#                   labels = c("A1","B1","C1","A2","B2","C2"),
#                   align = "h",
#                   legend = "top",
#                   font.label = list(size=10,color = "black", face = "bold"))
## 排版
p1+p2+p3+p4+p5+p6+plot_layout(ncol = 3, guides = "collect")+
  plot_annotation(tag_levels = "A")


### machine learning model feature importance
var_imp <-read.csv("feature importance_new.csv")

## any complication
f1 <- ggplot(var_imp[which(var_imp$outcome=="Any complication"&var_imp$model=="LR"),],
       aes(x=reletive_score,y=reorder(factor(feature_name),reletive_score)))+
  # facet_grid(. ~ model,scales = 'free_x')+
  geom_bar(stat = "identity",position = "dodge",fill = "#6F99ADFF")+
  theme_bw()+
  scale_x_continuous(limits = c(0,1),breaks = seq(0,1,0.2))+
  xlab("Relative importance")+
  ylab("Feature name")+
  theme(text=element_text(size = 8),
        axis.title.x =element_text(size=8), 
        axis.title.y=element_text(size=8))


f2 <- ggplot(var_imp[which(var_imp$outcome=="Any complication"&var_imp$model=="SVM"),],
             aes(x=reletive_score,y=reorder(factor(feature_name),reletive_score)))+
  # facet_grid(. ~ model,scales = 'free_x')+
  geom_bar(stat = "identity",position = "dodge",fill = "#6F99ADFF")+
  theme_bw()+
  scale_x_continuous(limits = c(0,1),breaks = seq(0,1,0.2))+
  xlab("Relative importance")+
  ylab("Feature name")+
  theme(text=element_text(size = 8),
        axis.title.x =element_text(size=8), 
        axis.title.y=element_text(size=8))


f3 <- ggplot(var_imp[which(var_imp$outcome=="Any complication"&var_imp$model=="RF"),],
             aes(x=reletive_score,y=reorder(factor(feature_name),reletive_score)))+
  # facet_grid(. ~ model,scales = 'free_x')+
  geom_bar(stat = "identity",position = "dodge",fill = "#6F99ADFF")+
  theme_bw()+
  scale_x_continuous(limits = c(0,1),breaks = seq(0,1,0.2))+
  xlab("Relative importance")+
  ylab("Feature name")+
  theme(text=element_text(size = 8),
        axis.title.x =element_text(size=8), 
        axis.title.y=element_text(size=8))

f4 <- ggplot(var_imp[which(var_imp$outcome=="Any complication"&var_imp$model=="GBM"),],
             aes(x=reletive_score,y=reorder(factor(feature_name),reletive_score)))+
  # facet_grid(. ~ model,scales = 'free_x')+
  geom_bar(stat = "identity",position = "dodge",fill = "#6F99ADFF")+
  theme_bw()+
  scale_x_continuous(limits = c(0,1),breaks = seq(0,1,0.2))+
  xlab("Relative importance")+
  ylab("Feature name")+
  theme(text=element_text(size = 8),
        axis.title.x =element_text(size=8), 
        axis.title.y=element_text(size=8))

f5 <- ggplot(var_imp[which(var_imp$outcome=="Any complication"&var_imp$model=="XGBoost"),],
             aes(x=reletive_score,y=reorder(factor(feature_name),reletive_score)))+
  # facet_grid(. ~ model,scales = 'free_x')+
  geom_bar(stat = "identity",position = "dodge",fill = "#6F99ADFF")+
  theme_bw()+
  scale_x_continuous(limits = c(0,1),breaks = seq(0,1,0.2))+
  xlab("Relative importance")+
  ylab("Feature name")+
  theme(text=element_text(size = 8),
        axis.title.x =element_text(size=8), 
        axis.title.y=element_text(size=8))

# f1+f2+f3+f4+f5+plot_layout(ncol = 5, guides = "collect")+
#   plot_annotation(tag_levels = "A")


## cardiac effusion
f6 <- ggplot(var_imp[which(var_imp$outcome=="Cardiac effusion/tamponade"&var_imp$model=="LR"),],
             aes(x=reletive_score,y=reorder(factor(feature_name),reletive_score)))+
  # facet_grid(. ~ model,scales = 'free_x')+
  geom_bar(stat = "identity",position = "dodge",fill = "#4DBBD5FF")+
  theme_bw()+
  scale_x_continuous(limits = c(0,1),breaks = seq(0,1,0.2))+
  xlab("Relative importance")+
  ylab("Feature name")+
  theme(text=element_text(size = 8),
        axis.title.x =element_text(size=8), 
        axis.title.y=element_text(size=8))


f7 <- ggplot(var_imp[which(var_imp$outcome=="Cardiac effusion/tamponade"&var_imp$model=="SVM"),],
             aes(x=reletive_score,y=reorder(factor(feature_name),reletive_score)))+
  # facet_grid(. ~ model,scales = 'free_x')+
  geom_bar(stat = "identity",position = "dodge",fill = "#4DBBD5FF")+
  theme_bw()+
  scale_x_continuous(limits = c(0,1),breaks = seq(0,1,0.2))+
  xlab("Relative importance")+
  ylab("Feature name")+
  theme(text=element_text(size = 8),
        axis.title.x =element_text(size=8), 
        axis.title.y=element_text(size=8))

f8 <- ggplot(var_imp[which(var_imp$outcome=="Cardiac effusion/tamponade"&var_imp$model=="RF"),],
             aes(x=reletive_score,y=reorder(factor(feature_name),reletive_score)))+
  # facet_grid(. ~ model,scales = 'free_x')+
  geom_bar(stat = "identity",position = "dodge",fill = "#4DBBD5FF")+
  theme_bw()+
  scale_x_continuous(limits = c(0,1),breaks = seq(0,1,0.2))+
  xlab("Relative importance")+
  ylab("Feature name")+
  theme(text=element_text(size = 8),
        axis.title.x =element_text(size=8), 
        axis.title.y=element_text(size=8))

f9 <- ggplot(var_imp[which(var_imp$outcome=="Cardiac effusion/tamponade"&var_imp$model=="GBM"),],
             aes(x=reletive_score,y=reorder(factor(feature_name),reletive_score)))+
  # facet_grid(. ~ model,scales = 'free_x')+
  geom_bar(stat = "identity",position = "dodge",fill = "#4DBBD5FF")+
  theme_bw()+
  scale_x_continuous(limits = c(0,1),breaks = seq(0,1,0.2))+
  xlab("Relative importance")+
  ylab("Feature name")+
  theme(text=element_text(size = 8),
        axis.title.x =element_text(size=8), 
        axis.title.y=element_text(size=8))

f10 <- ggplot(var_imp[which(var_imp$outcome=="Cardiac effusion/tamponade"&var_imp$model=="XGBoost"),],
             aes(x=reletive_score,y=reorder(factor(feature_name),reletive_score)))+
  # facet_grid(. ~ model,scales = 'free_x')+
  geom_bar(stat = "identity",position = "dodge",fill = "#4DBBD5FF")+
  theme_bw()+
  scale_x_continuous(limits = c(0,1),breaks = seq(0,1,0.2))+
  xlab("Relative importance")+
  ylab("Feature name")+
  theme(text=element_text(size = 8),
        axis.title.x =element_text(size=8), 
        axis.title.y=element_text(size=8))


## hemorrhage
f11 <- ggplot(var_imp[which(var_imp$outcome=="Hemorrhage/hematoma"&var_imp$model=="LR"),],
              aes(x=reletive_score,y=reorder(factor(feature_name),reletive_score)))+
  # facet_grid(. ~ model,scales = 'free_x')+
  geom_bar(stat = "identity",position = "dodge",fill = "#91D1C2FF")+
  theme_bw()+
  scale_x_continuous(limits = c(0,1),breaks = seq(0,1,0.2))+
  xlab("Relative importance")+
  ylab("Feature name")+
  theme(text=element_text(size = 8),
        axis.title.x =element_text(size=8), 
        axis.title.y=element_text(size=8))


f12 <- ggplot(var_imp[which(var_imp$outcome=="Hemorrhage/hematoma"&var_imp$model=="SVM"),],
              aes(x=reletive_score,y=reorder(factor(feature_name),reletive_score)))+
  # facet_grid(. ~ model,scales = 'free_x')+
  geom_bar(stat = "identity",position = "dodge",fill = "#91D1C2FF")+
  theme_bw()+
  scale_x_continuous(limits = c(0,1),breaks = seq(0,1,0.2))+
  xlab("Relative importance")+
  ylab("Feature name")+
  theme(text=element_text(size = 8),
        axis.title.x =element_text(size=8), 
        axis.title.y=element_text(size=8))

f13 <- ggplot(var_imp[which(var_imp$outcome=="Hemorrhage/hematoma"&var_imp$model=="RF"),],
              aes(x=reletive_score,y=reorder(factor(feature_name),reletive_score)))+
  # facet_grid(. ~ model,scales = 'free_x')+
  geom_bar(stat = "identity",position = "dodge",fill = "#91D1C2FF")+
  theme_bw()+
  scale_x_continuous(limits = c(0,1),breaks = seq(0,1,0.2))+
  xlab("Relative importance")+
  ylab("Feature name")+
  theme(text=element_text(size = 8),
        axis.title.x =element_text(size=8), 
        axis.title.y=element_text(size=8))

f14 <- ggplot(var_imp[which(var_imp$outcome=="Hemorrhage/hematoma"&var_imp$model=="GBM"),],
              aes(x=reletive_score,y=reorder(factor(feature_name),reletive_score)))+
  # facet_grid(. ~ model,scales = 'free_x')+
  geom_bar(stat = "identity",position = "dodge",fill = "#91D1C2FF")+
  theme_bw()+
  scale_x_continuous(limits = c(0,1),breaks = seq(0,1,0.2))+
  xlab("Relative importance")+
  ylab("Feature name")+
  theme(text=element_text(size = 8),
        axis.title.x =element_text(size=8), 
        axis.title.y=element_text(size=8))

f15 <- ggplot(var_imp[which(var_imp$outcome=="Hemorrhage/hematoma"&var_imp$model=="XGBoost"),],
              aes(x=reletive_score,y=reorder(factor(feature_name),reletive_score)))+
  # facet_grid(. ~ model,scales = 'free_x')+
  geom_bar(stat = "identity",position = "dodge",fill = "#91D1C2FF")+
  theme_bw()+
  scale_x_continuous(limits = c(0,1),breaks = seq(0,1,0.2))+
  xlab("Relative importance")+
  ylab("Feature name")+
  theme(text=element_text(size = 8),
        axis.title.x =element_text(size=8),
        axis.title.y=element_text(size=8))

f3+f4+f5+f8+f9+f10+f13+f14+f15+plot_layout(ncol = 3, guides = "collect")+
  plot_annotation(tag_levels = "A")


### calibration plot
cal <-read.csv("calibration plot(smote)_new.csv")
cal1 <- ggplot(cal[which(cal$outcome=="any complication"),],
       aes(x=Pred, y=Obs),colour= model,group=model) +
  # geom_errorbar(aes(ymin=Obs_lower,
  #                   ymax=Obs_upper), width=.02)+
  annotate(geom = "segment", x = 0, y
           = 0, xend =1, yend = 1,color = "grey")+
  expand_limits(x = 0, y = 0) +
  scale_x_continuous(expand = c(0, 0),limits = c(-0.05,1.05)) +
  scale_y_continuous(expand = c(0, 0),limits = c(-0.05,1.05))+
  geom_line(aes(color=model),
            # linetype="dashed",
            size=0.6)+
  geom_point(aes(color=model),size=2.5,shape=18)+
  scale_color_manual(values = c("#631879FF","#1B1919FF","#008280FF","#3B4992FF","#A20056FF"))+
  guides(color=FALSE,shape=FALSE,linetype=FALSE)+
  theme_bw()+
  xlab("Predicted Probability")+
  ylab("Observed Probability")+
  theme(text=element_text(size = 8),
        axis.title.x =element_text(size=8),
        axis.title.y=element_text(size=8))


cal2 <- ggplot(cal[which(cal$outcome=="cardiac effusion"),],
       aes(x=Pred, y=Obs),colour= model,group=model) +
  # geom_errorbar(aes(ymin=Obs_lower,
  #                   ymax=Obs_upper), width=.02)+
  annotate(geom = "segment", x = 0, y
           = 0, xend =1, yend = 1,color = "grey")+
  expand_limits(x = 0, y = 0) +
  scale_x_continuous(expand = c(0, 0),limits = c(-0.05,1.05)) +
  scale_y_continuous(expand = c(0, 0),limits = c(-0.05,1.05))+
  geom_line(aes(color=model),
            # linetype="dashed",
            size=0.6)+
  geom_point(aes(color=model),size=2.5,shape=18)+
  scale_color_manual(values = c("#631879FF","#1B1919FF","#008280FF","#3B4992FF","#A20056FF"))+
  guides(color=FALSE,shape=FALSE,linetype=FALSE)+
  theme_bw()+
  xlab("Predicted Probability")+
  ylab("Observed Probability")+
  theme(text=element_text(size = 8),
        axis.title.x =element_text(size=8),
        axis.title.y=element_text(size=8))


cal3 <- ggplot(cal[which(cal$outcome=="hemorrhage"),],
       aes(x=Pred, y=Obs),colour= model,group=model) +
  # geom_errorbar(aes(ymin=Obs_lower,
  #                   ymax=Obs_upper), width=.02)+
  annotate(geom = "segment", x = 0, y
           = 0, xend =1, yend = 1,color = "grey")+
  expand_limits(x = 0, y = 0) +
  scale_x_continuous(expand = c(0, 0),limits = c(-0.05,1.05)) +
  scale_y_continuous(expand = c(0, 0),limits = c(-0.05,1.05))+
  geom_line(aes(color=model),
            # linetype="dashed",
            size=0.6)+
  geom_point(aes(color=model),size=2.5,shape=18)+
  scale_color_manual(values = c("#631879FF","#1B1919FF","#008280FF","#3B4992FF","#A20056FF"))+
  # theme(legend.position = "top")+
  theme_bw()+
  xlab("Predicted Probability")+
  ylab("Observed Probability")+
  theme(text=element_text(size = 8),
        axis.title.x =element_text(size=8),
        axis.title.y=element_text(size=8))

ggpubr::ggarrange(cal1,cal2,cal3,ncol = 3,nrow = 1,widths=c(1.5,1.5,1.9),labels = c("A","B","C"))

  