setwd("/Users/weiyuna/Desktop/工作/合作项目/上海胸科/20211103导出数据/房颤/补充结局-不包含瓣膜手术")
# install.packages("ggplot2")
# install.packages("patchwork")
# install.packages("ggpubr")
# install.packages("ggsci")
library(ggplot2)
library(patchwork)
library(ggpubr)
library(ggsci)


### machine learning model feature selection
ml_feature<-read.csv("feature_selection_0411.csv")
p1 <- ggplot(ml_feature[which(ml_feature$outcomes=="Any complication"),],
       mapping = aes(x=factor(features,levels=c("5","10","15","20","all")), y=AUC_mean, colour= Model,group=Model)) +
  geom_errorbar(aes(ymin=AUC_lower,
                    ymax=AUC_upper), width=.1,position=position_dodge(0.1))+
  geom_line(aes(color=Model))+
  geom_point(aes(color=Model))+
  scale_color_manual(values = c("#631879FF","#008280FF","#3B4992FF","#A20056FF"))+
  # scale_color_aaas()+
  theme_bw()+
  scale_y_continuous(limits = c(0.5,0.75),breaks = seq(0.5,0.75,0.05))+
  # guides(color=FALSE,shape=FALSE,linetype=FALSE)+
  xlab("Number of features selected")+
  ylab("AUC")



p2 <- ggplot(ml_feature[which(ml_feature$outcomes=="Cardiac effusion"),],
       mapping = aes(x=factor(features,levels=c("5","10","15","20","all")), y=AUC_mean, colour= Model,group=Model)) +
  geom_errorbar(aes(ymin=AUC_lower,
                    ymax=AUC_upper), width=.1,position=position_dodge(0.1))+
  geom_line(aes(color=Model))+
  geom_point(aes(color=Model))+
  scale_color_manual(values = c("#631879FF","#008280FF","#3B4992FF","#A20056FF"))+
  # scale_color_aaas()+
  theme_bw()+
  scale_y_continuous(limits = c(0.5,0.75),breaks = seq(0.5,0.75,0.05))+
  # guides(color=FALSE,shape=FALSE,linetype=FALSE)+
  xlab("Number of features selected")+
  ylab("AUC")

p3 <-ggplot(ml_feature[which(ml_feature$outcomes=="Hemorrhage"),],
       mapping = aes(x=factor(features,levels=c("5","10","15","20","all")), y=AUC_mean, colour= Model,group=Model)) +
  geom_errorbar(aes(ymin=AUC_lower,
                    ymax=AUC_upper), width=.1,position=position_dodge(0.1))+
  geom_line(aes(color=Model))+
  geom_point(aes(color=Model))+
  scale_color_manual(values = c("#631879FF","#008280FF","#3B4992FF","#A20056FF"))+
  # scale_color_aaas()+
  theme_bw()+
  scale_y_continuous(limits = c(0.5,0.85),breaks = seq(0.5,0.85,0.05))+
  guides(color=FALSE,shape=FALSE,linetype=FALSE)+
  xlab("Number of features selected")+
  ylab("AUC")

p1+p2+p3+plot_layout(ncol = 3, guides = "collect")+
  plot_annotation(tag_levels = "A")
ggsave('feature_selection.png',width = 15, height = 5,dpi=600)



#### 绘制不同结局不同模型重要特征排序雷达图
# install.packages("devtools")
# devtools::install_github("ricardo-bion/ggradar",dependencies = TRUE)
# install.packages("fmsb")
library(ggradar)
library(fmsb)

radar_data1 <- read.csv("radar plot1.csv",header=T,row.names = 1)
radar_data2 <- read.csv("radar plot2.csv",header=T,row.names = 1)
radar_data3 <- read.csv("radar plot3.csv",header=T,row.names = 1)

create_beautiful_radarchart <- function(data, color = "#00AFBB", 
                                        vlabels = colnames(data), vlcex = 1,
                                        caxislabels = NULL, title = NULL, ...){
  radarchart(
    data, axistype = 4,
    seg = 5,
    # Customize the polygon
    pcol = color, 
    pfcol = scales::alpha(color, 0.1),  # 颜色填充
    plwd = 1, 
    plty = 1,
    # Customize the grid
    cglcol = "grey", cglty = 2, cglwd = 1,
    # Customize the axis
    axislabcol = "grey",
    # Variable labels
    vlcex = vlcex, vlabels = vlabels,
    caxislabels = caxislabels, title = title, ...
  )
}

op <- par(mar = c(1, 2, 2, 2))
create_beautiful_radarchart(data = radar_data1,
                            color = c("#008280FF","#3B4992FF","#A20056FF"))
legend(
  x = "bottom", legend = rownames(radar_data1[-c(0,1,2),]), horiz = TRUE,
  bty = "n", pch = 20 , col = c("#008280FF","#3B4992FF","#A20056FF"),
  text.col = "black", cex = 1, pt.cex = 1.5
)


create_beautiful_radarchart(data = radar_data2,
                                  color = c("#008280FF","#3B4992FF","#A20056FF"))
legend(
  x = "bottom", legend = rownames(radar_data2[-c(0,1,2),]), horiz = TRUE,
  bty = "n", pch = 20 , col = c("#008280FF","#3B4992FF","#A20056FF"),
  text.col = "black", cex = 1, pt.cex = 1.5
)

create_beautiful_radarchart(data = radar_data3,
                                  color = c("#008280FF","#3B4992FF","#A20056FF"))
legend(
  x = "bottom", legend = rownames(radar_data3[-c(0,1,2),]), horiz = TRUE,
  bty = "n", pch = 20 , col = c("#008280FF","#3B4992FF","#A20056FF"),
  text.col = "black", cex = 1, pt.cex = 1.5
)

## 分别用最佳算法模型绘制雷达图，并将特征取并集后绘制不同结局特征重要性雷达图
radar_data_new1 <- read.csv("radar plot1_new.csv",header=T,row.names = 1)
radar_data_new2 <- read.csv("radar plot2_new.csv",header=T,row.names = 1)
radar_data_new3 <- read.csv("radar plot3_new.csv",header=T,row.names = 1)
radar_data_new4 <- read.csv("radar plot4_new.csv",header=T,row.names = 1)

create_beautiful_radarchart <- function(data, color = "#00AFBB", 
                                        vlabels = colnames(data), vlcex = 1,
                                        caxislabels = NULL, title = NULL, ...){
  radarchart(
    data, axistype = 4,
    seg = 5,
    # Customize the polygon
    pcol = color, 
    pfcol = scales::alpha(color, 0.1),  # 颜色填充
    plwd = 1, 
    plty = 1,
    # Customize the grid
    cglcol = "grey", cglty = 2, cglwd = 1,
    # Customize the axis
    axislabcol = "grey",
    # Variable labels
    vlcex = vlcex, vlabels = vlabels,
    caxislabels = caxislabels, title = title, ...
  )
}

op <- par(mar = c(1, 2, 2, 2))
create_beautiful_radarchart(data = radar_data_new1,color = c("#00AFBB"))
create_beautiful_radarchart(data = radar_data_new2,color = c("#00AFBB"))
create_beautiful_radarchart(data = radar_data_new3,color = c("#00AFBB"))

create_beautiful_radarchart(data = radar_data_new4,
                            color = c("#008280FF","#3B4992FF","#A20056FF"))
legend(
  x = "bottom", legend = rownames(radar_data_new4[-c(0,1,2),]), horiz = TRUE,
  bty = "n", pch = 20 , col = c("#008280FF","#3B4992FF","#A20056FF"),
  text.col = "black", cex = 1, pt.cex = 1.5
)

