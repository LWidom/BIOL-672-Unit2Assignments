# Louis Widom
# lpw8274@rit.edu
# Designed in Windows 10
# Last updated 06 November 2020

# List of required packages:
#  caret
#  class
#  dplyr
#  e1071
#  ggplot2
#  grid
##  imager
#  kernlab
#  liquidSVM
#  MASS
#  plyr
#  reshape2
# Associated data files (should be located in the same folder as this script):
#  winequality-red.csv
# ======================================================================================

# Begin Script
library('caret')
library('class')
library('dplyr')
library('e1071')
library('ggplot2')
library('grid')
#library('imager')
library('kernlab')
library('liquidSVM')
library('MASS')
library('plyr')
library('reshape2')
# Set the working directory to be the same location as this script
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Read in "Red Wine Quality" dataset available from UCI Machine Learning
# (available: https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009?select=winequality-red.csv)
wine = read.csv(paste(getwd(),'/winequality-red.csv',sep=''), header = TRUE, sep = ",", quote = "\"",
                dec = ".", fill = TRUE, comment.char = "")
# Assign variable names
FA <- wine$fixed_acidity
VA <- wine$volatile_acidity
CA <- wine$citric_acid
RS <- wine$residual_sugar
CH <- wine$chlorides
FS <- wine$free_sulfur_dioxide
TS <- wine$total_sulfur_dioxide
DE <- wine$density
PH <- wine$pH
SU <- wine$sulphates
AL <- wine$alcohol
# Regroup quality into 3 numerical groups. 1 = "poor", 2 = "average",
# 3 = "excellent"
wine$Category[wine$quality<5]=1
wine$Category[wine$quality>4 & wine$quality<7]=2
wine$Category[wine$quality>6]=3
QU <- wine$Category
# Make dataframe and subframe that omits quality
dataframe <- data.frame(FA,VA,CA,RS,CH,FS,TS,DE,PH,SU,AL,QU)
subframe <- data.frame(FA,VA,CA,RS,CH,FS,TS,DE,PH,SU,AL)

# Set random seed
rnd <- sample(1:150, 1)
set.seed(rnd)
# Get number of rows in the dataframe
nrows <- nrow(dataframe)
# Shuffle the data's rows
rows <- sample(nrows)
dataframe <- dataframe[rows, ]
# Determine the rough number of rows for 1/5 of the dataframe
split_rows <- ceiling(nrows/5)
# Split the dataframe into 5 groups
split_frame <- split(dataframe, (0:nrows %/% split_rows))
# Initialize lists to store the accuracies of each model
resultsSVM_linear <- vector(mode="list",length=5)
resultsSVM_poly <- vector(mode="list",length=5)
resultsSVM_radial <- vector(mode="list",length=5)
resultsKSVM_linear <- vector(mode="list",length=5)
resultsKSVM_poly <- vector(mode="list",length=5)
resultsKSVM_radial <- vector(mode="list",length=5)
resultsLSVM <- vector(mode="list",length=5)

# Loop through each each of the split frames to perform 5-fold cross-validation
for(i in 1:5 )
{# Establish training datasets
  index1 = i + 1
  if (index1 > 5)
  {index1 = index1 - i
  }
  mysample = split_frame[[index1]]
  # Join remaining split frames
  for (j in 1:3)
  {index2 = index1 + j
  if (index2 > 5)
  {index2 = index2 -i
  }
  mysample = rbind(mysample,split_frame[[index2]])
  }
  # Establish testing dataset and a subframe without the QU column
  mytestdata = split_frame[[i]]
  mytestsubframe = subset(mytestdata,select=-QU)
  
  # Ensure class is a vector matching length of a single data column in training data frame
  myquality = mysample[12:12]
  
  # Run svm from e1071 library
  # Start with linear kernel
  print(paste('svm e1071 linear kernel subset ',i))
  mytest_3 <-  e1071::svm(QU~FA+VA+CA+RS+CH+FS+TS+DE+PH+SU+AL, mysample, probability = TRUE, type = "C-classification", kernel = "linear")
  mypred_3 <- predict(mytest_3, mytestsubframe, probability = FALSE, decision.values = TRUE)
  mySVM_1 <- summary(mytest_3)
  # Generate new scatterplots to see performance of svm with linear kernel
  myplot1 <-ggplot(mysample, aes(RS, AL, colour = as.factor(QU))) + geom_point()
  myplot2 <-ggplot(mysample, aes(FS, FA, colour = as.factor(QU))) + geom_point()                                 
  myplot3 <-ggplot(mytestdata, aes(RS, AL, colour = as.factor(QU))) + geom_point()
  myplot4 <-ggplot(mytestdata, aes(FS, FA, colour = as.factor(QU))) + geom_point()
  myplot5_3 <-ggplot(mytestsubframe, aes(RS, AL, colour = mypred_3)) + geom_point()                                 
  myplot6_3 <-ggplot(mytestsubframe, aes(FS, FA, colour = mypred_3)) + geom_point()
  # Save plot as PDF
  pdf(paste('svm e1071 linear ',i,'.pdf'))
  pushViewport(viewport(layout = grid.layout(4, 2,heights=unit(c(1,5,5,5),"null"))))
  grid.text(paste("svm e1071 linear ",i), vp = viewport(layout.pos.row = 1, layout.pos.col = 1:2))
  print(myplot1, vp = viewport(layout.pos.row = 2, layout.pos.col = 1))
  print(myplot2, vp = viewport(layout.pos.row = 2, layout.pos.col = 2))
  print(myplot3, vp = viewport(layout.pos.row = 3, layout.pos.col = 1))
  print(myplot4, vp = viewport(layout.pos.row = 3, layout.pos.col = 2))
  print(myplot5_3, vp = viewport(layout.pos.row = 4, layout.pos.col = 1))
  print(myplot6_3, vp = viewport(layout.pos.row = 4, layout.pos.col = 2))
  dev.off()
  # Generate confusion matrix
  mymatrix_3 <- confusionMatrix(as.factor(mypred_3), as.factor(mytestdata$QU))
  print(mymatrix_3)
  # Store the accuracy in a list
  resultsSVM_linear[i] = mymatrix_3$overall[1]
  
  # Run svm from e1071 library with polynomial kernel
  print(paste('svm e1071 polynomial kernel subset ',i))
  mytest_4 <-  e1071::svm(QU~FA+VA+CA+RS+CH+FS+TS+DE+PH+SU+AL, mysample, probability = TRUE, type = "C-classification", kernel = "polynomial", degree = 3)
  mypred_4 <- predict(mytest_4, mytestsubframe, probability = FALSE, decision.values = TRUE)
  mySVM_2 <- summary(mytest_4)
  # Generate new scatterplots to see performance of svm with linear kernel
  myplot5_4 <-ggplot(mytestsubframe, aes(RS, AL, colour = mypred_4)) + geom_point()                                 
  myplot6_4 <-ggplot(mytestsubframe, aes(FS, FA, colour = mypred_4)) + geom_point()
  # Save plot as PDF
  pdf(paste('svm e1071 poly ',i,'.pdf'))
  pushViewport(viewport(layout = grid.layout(4, 2,heights=unit(c(1,5,5,5),"null"))))
  grid.text(paste("svm e1071 poly ",i), vp = viewport(layout.pos.row = 1, layout.pos.col = 1:2))
  print(myplot1, vp = viewport(layout.pos.row = 2, layout.pos.col = 1))
  print(myplot2, vp = viewport(layout.pos.row = 2, layout.pos.col = 2))
  print(myplot3, vp = viewport(layout.pos.row = 3, layout.pos.col = 1))
  print(myplot4, vp = viewport(layout.pos.row = 3, layout.pos.col = 2))
  print(myplot5_4, vp = viewport(layout.pos.row = 4, layout.pos.col = 1))
  print(myplot6_4, vp = viewport(layout.pos.row = 4, layout.pos.col = 2))
  dev.off()
  # Generate confusion matrix
  mymatrix_4 <- confusionMatrix(as.factor(mypred_4), as.factor(mytestdata$QU))
  print(mymatrix_4)
  # Store the accuracy in a list
  resultsSVM_poly[i] = mymatrix_4$overall[1]
  
  # Run svm from e1071 library with radial kernel
  print(paste('svm e1071 radial kernel subset ',i))
  mytest_5 <-  e1071::svm(QU~FA+VA+CA+RS+CH+FS+TS+DE+PH+SU+AL, mysample, probability = TRUE, type = "C-classification", kernel = "radial", gamma = 4.5)
  mypred_5 <- predict(mytest_5, mytestsubframe, probability = FALSE, decision.values = TRUE)
  mySVM_3 <- summary(mytest_5)
  # Generate new scatterplots to see performance of svm with linear kernel
  myplot5_5 <-ggplot(mytestsubframe, aes(RS, AL, colour = mypred_5)) + geom_point()                                 
  myplot6_5 <-ggplot(mytestsubframe, aes(FS, FA, colour = mypred_5)) + geom_point()
  # Save plot as PDF
  pdf(paste('svm e1071 radial ',i,'.pdf'))
  pushViewport(viewport(layout = grid.layout(4, 2,heights=unit(c(1,5,5,5),"null"))))
  grid.text(paste("svm e1071 radial ",i), vp = viewport(layout.pos.row = 1, layout.pos.col = 1:2))
  print(myplot1, vp = viewport(layout.pos.row = 2, layout.pos.col = 1))
  print(myplot2, vp = viewport(layout.pos.row = 2, layout.pos.col = 2))
  print(myplot3, vp = viewport(layout.pos.row = 3, layout.pos.col = 1))
  print(myplot4, vp = viewport(layout.pos.row = 3, layout.pos.col = 2))
  print(myplot5_5, vp = viewport(layout.pos.row = 4, layout.pos.col = 1))
  print(myplot6_5, vp = viewport(layout.pos.row = 4, layout.pos.col = 2))
  dev.off()
  # Generate confusion matrix
  mymatrix_5 <- confusionMatrix(as.factor(mypred_5), as.factor(mytestdata$QU))
  print(mymatrix_5)
  # Store the accuracy in a list
  resultsSVM_radial[i] = mymatrix_5$overall[1]
  
  # Run ksvm from kernlab library with linear kernel
  print(paste('ksvm linear kernel subset ',i))
  mytest_6 <- ksvm(as.matrix(mysample), myquality, kernel = 'vanilladot')
  mypred_6 <- round(predict(mytest_6, mytestdata, type='response'))
  mySVM_4 <- summary(mytest_6)
  # Generate new scatterplots to see performance of svm with linear kernel
  myplot5_6 <-ggplot(mytestsubframe, aes(RS, AL, colour = as.factor(mypred_6))) + geom_point()                                 
  myplot6_6 <-ggplot(mytestsubframe, aes(FS, FA, colour = as.factor(mypred_6))) + geom_point()
  # Save plot as PDF
  pdf(paste('ksvm linear ',i,'.pdf'))
  pushViewport(viewport(layout = grid.layout(4, 2,heights=unit(c(1,5,5,5),"null"))))
  grid.text(paste("ksvm linear ",i), vp = viewport(layout.pos.row = 1, layout.pos.col = 1:2))
  print(myplot1, vp = viewport(layout.pos.row = 2, layout.pos.col = 1))
  print(myplot2, vp = viewport(layout.pos.row = 2, layout.pos.col = 2))
  print(myplot3, vp = viewport(layout.pos.row = 3, layout.pos.col = 1))
  print(myplot4, vp = viewport(layout.pos.row = 3, layout.pos.col = 2))
  print(myplot5_6, vp = viewport(layout.pos.row = 4, layout.pos.col = 1))
  print(myplot6_6, vp = viewport(layout.pos.row = 4, layout.pos.col = 2))
  dev.off()
  # Generate confusion matrix
  #mypred_6 = round(mypred_6) # Round to integer
  mymatrix_6 <- confusionMatrix(as.factor(mypred_6), as.factor(mytestdata$QU))
  print(mymatrix_6)
  # Store the accuracy in a list
  resultsKSVM_linear[i] = mymatrix_6$overall[1]
  
  # Run ksvm from kernlab library with polynomial kernel
  print(paste('ksvm polynomial kernel subset ',i))
  mytest_7 <-  ksvm(as.matrix(mysample), myquality, kernel = 'polydot')
  mypred_7 <- round(predict(mytest_7, mytestdata, type='response'))
  mySVM_5 <- summary(mytest_7)
  # Generate new scatterplots to see performance of svm with linear kernel
  myplot5_7 <-ggplot(mytestsubframe, aes(RS, AL, colour = as.factor(mypred_7))) + geom_point()                                 
  myplot6_7 <-ggplot(mytestsubframe, aes(FS, FA, colour = as.factor(mypred_7))) + geom_point()
  # Save plot as PDF
  pdf(paste('ksvm poly ',i,'.pdf'))
  pushViewport(viewport(layout = grid.layout(4, 2,heights=unit(c(1,5,5,5),"null"))))
  grid.text(paste("ksvm poly ",i), vp = viewport(layout.pos.row = 1, layout.pos.col = 1:2))
  print(myplot1, vp = viewport(layout.pos.row = 2, layout.pos.col = 1))
  print(myplot2, vp = viewport(layout.pos.row = 2, layout.pos.col = 2))
  print(myplot3, vp = viewport(layout.pos.row = 3, layout.pos.col = 1))
  print(myplot4, vp = viewport(layout.pos.row = 3, layout.pos.col = 2))
  print(myplot5_7, vp = viewport(layout.pos.row = 4, layout.pos.col = 1))
  print(myplot6_7, vp = viewport(layout.pos.row = 4, layout.pos.col = 2))
  dev.off()
  # Generate confusion matrix
  #mypred_7 = round(mypred_7) # Round to integer
  mymatrix_7 <- confusionMatrix(as.factor(mypred_7), as.factor(mytestdata$QU))
  print(mymatrix_7)
  # Store the accuracy in a list
  resultsKSVM_poly[i] = mymatrix_7$overall[1]
  
  # Run ksvm from kernlab library with radial kernel
  print(paste('ksvm radial kernel subset ',i))
  mytest_8 <-  ksvm(as.matrix(mysample), myquality, kernel = 'rbfdot')
  mypred_8 <- round(predict(mytest_8, mytestdata, type='response'))
  mySVM_6 <- summary(mytest_8)
  # Generate new scatterplots to see performance of svm with linear kernel
  myplot5_8 <-ggplot(mytestsubframe, aes(RS, AL, colour = as.factor(mypred_8))) + geom_point()                                 
  myplot6_8 <-ggplot(mytestsubframe, aes(FS, FA, colour = as.factor(mypred_8))) + geom_point()
  # Save plot as PDF
  pdf(paste('ksvm radial ',i,'.pdf'))
  pushViewport(viewport(layout = grid.layout(4, 2,heights=unit(c(1,5,5,5),"null"))))
  grid.text(paste("ksvm radial ",i), vp = viewport(layout.pos.row = 1, layout.pos.col = 1:2))
  print(myplot1, vp = viewport(layout.pos.row = 2, layout.pos.col = 1))
  print(myplot2, vp = viewport(layout.pos.row = 2, layout.pos.col = 2))
  print(myplot3, vp = viewport(layout.pos.row = 3, layout.pos.col = 1))
  print(myplot4, vp = viewport(layout.pos.row = 3, layout.pos.col = 2))
  print(myplot5_8, vp = viewport(layout.pos.row = 4, layout.pos.col = 1))
  print(myplot6_8, vp = viewport(layout.pos.row = 4, layout.pos.col = 2))
  dev.off()
  # Generate confusion matrix
  #mypred_8 = round(mypred_8) # Round to integer
  mymatrix_8 <- confusionMatrix(as.factor(mypred_8), as.factor(mytestdata$QU))
  print(mymatrix_8)
  # Store the accuracy in a list
  resultsKSVM_radial[i] = mymatrix_8$overall[1]
  
  # Run svm from liquidSVM library
  print(paste('liquidSVM subset ',i))
  mytest_9 <- liquidSVM::svm(QU~FA+VA+CA+RS+CH+FS+TS+DE+PH+SU+AL, mysample, threads=2, display=1, gpus=0, do.select=TRUE, predict.prob=FALSE)
  mypred_9 <- round(predict(mytest_9, mytestsubframe, probability = FALSE, decision.values = TRUE))
  # Generate new scatterplots to see performance of svm with linear kernel
  myplot5_9 <-ggplot(mytestsubframe, aes(RS, AL, colour = as.factor(mypred_9))) + geom_point()                                 
  myplot6_9 <-ggplot(mytestsubframe, aes(FS, FA, colour = as.factor(mypred_9))) + geom_point()
  # Save plot as PDF
  pdf(paste('liquidSVM ',i,'.pdf'))
  pushViewport(viewport(layout = grid.layout(4, 2,heights=unit(c(1,5,5,5),"null"))))
  grid.text(paste("liquidSVM ",i), vp = viewport(layout.pos.row = 1, layout.pos.col = 1:2))
  print(myplot1, vp = viewport(layout.pos.row = 2, layout.pos.col = 1))
  print(myplot2, vp = viewport(layout.pos.row = 2, layout.pos.col = 2))
  print(myplot3, vp = viewport(layout.pos.row = 3, layout.pos.col = 1))
  print(myplot4, vp = viewport(layout.pos.row = 3, layout.pos.col = 2))
  print(myplot5_9, vp = viewport(layout.pos.row = 4, layout.pos.col = 1))
  print(myplot6_9, vp = viewport(layout.pos.row = 4, layout.pos.col = 2))
  dev.off()
  # Generate confusion matrix
  mymatrix_9 <- confusionMatrix(as.factor(mypred_9), as.factor(mytestdata$QU))
  print(mymatrix_9)
  # Store the accuracy in a list
  resultsLSVM[i] = mymatrix_9$overall[1]
  
}
# Convert results into vectors
resultsSVM_linear <- unlist(resultsSVM_linear)
resultsSVM_poly <- unlist(resultsSVM_poly)
resultsSVM_radial <- unlist(resultsSVM_radial)
resultsKSVM_linear <- unlist(resultsKSVM_linear)
resultsKSVM_poly <- unlist(resultsKSVM_poly)
resultsKSVM_radial <- unlist(resultsKSVM_radial)
resultsLSVM <- unlist(resultsLSVM)

# Calculate mean
resultsDF <- data.frame(resultsSVM_linear,resultsSVM_poly,resultsSVM_radial,resultsKSVM_linear,resultsKSVM_poly,resultsKSVM_radial,resultsLSVM)
# Summarize the results in terms of mean and standard deviation
model.melt <- melt(resultsDF)
model.summary <- ddply(model.melt,'variable',summarise,mean=mean(value),stdev=sd(value))
# Set up upper and lower error bars
model.summary <- transform(model.summary,lower=mean-stdev,upper=mean+stdev)

# Set up error bar chart
model_bar <- ggplot(model.summary,aes(x=variable,y=mean)) + geom_col(position="dodge") + 
  geom_errorbar(aes(ymax=upper,ymin=lower),position="dodge",data=model.summary) + xlab('model')+
  ylab('accuracy') + theme(axis.text.x = element_text(angle=45, hjust=1))
#Plot
pdf('SVM comparison.pdf')
print(model_bar)
dev.off()
print('Scatterplot guide: The top images show the training subset, the middle images show the testing
      subset, and the bottom images show the model predictions.')
print('Of the different combinations of support vector machines and kernel functions tested, ksvm (from
      the kernlab package) with a linear kernel or polynomial kernel had the best performance. In both
      cases, the accuracy was equal to 1.0. The results can be observed in \'SVM comparison.pdf\'. Most
      of the SVM methods performed better than the methods featured in \'WidomLP_Win_A1.R\' except for svm
      from the e1071 library when a linear or polynomial kernel was specified.')

#========================================================================================================
# Running liquidSVM on a large scale problem


# Read in "Blood Cell Images" dataset available from Paul Mooney
# (available: https://www.kaggle.com/paultimothymooney/blood-cells)
# imagepath = paste(getwd(),'/dataset2-master/dataset2-master/images',sep='')
# trainfiles_EOSINOPHIL <- list.files(path = paste(imagepath,'/TRAIN/EOSINOPHIL',sep=''), pattern = "*.jpeg", full.names=TRUE)
# trainfiles_LYMPHOCYTE <- list.files(path = paste(imagepath,'/TRAIN/LYMPHOCYTE',sep=''), pattern = "*.jpeg", full.names=TRUE)
# trainfiles_MONOCYTE <- list.files(path = paste(imagepath,'/TRAIN/MONOCYTE',sep=''), pattern = "*.jpeg", full.names=TRUE)
# trainfiles_NEUTROPHIL <- list.files(path = paste(imagepath,'/TRAIN/NEUTROPHIL',sep=''), pattern = "*.jpeg", full.names=TRUE)
# trainIm_EOSINOPHIL <- lapply(trainfiles_EOSINOPHIL, load.image )
# trainIm_LYMPHOCYTE <- lapply(trainfiles_LYMPHOCYTE, load.image )
# trainIm_MONOCYTE <- lapply(trainfiles_MONOCYTE, load.image )
# trainIm_NEUTROPHIL <- lapply(trainfiles_NEUTROPHIL, load.image )

