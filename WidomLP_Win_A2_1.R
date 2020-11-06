# Louis Widom
# lpw8274@rit.edu
# Designed in Windows 10
# Last updated 03 November 2020

# List of required packages:
#  caret
#  class
#  dplyr
#  e1071
#  ggplot2
#  grid
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
wine$Category[wine$quality<5]="1"
wine$Category[wine$quality>4 & wine$quality<7]="2"
wine$Category[wine$quality>6]="3"
wine$Category = factor(wine$Category,levels=c(1,2,3))
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
resultsKNN <- vector(mode="list",length=5)
resultsNB <- vector(mode="list",length=5)
resultsLDA <- vector(mode="list",length=5)

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
  myquality = mysample[,12]

  # Perform K Nearest Neighbors
  print(paste("K Nearest Neighbors subset ",i))
  mypred <- knn(mysample, mytestdata, myquality, k=3, l = 0, prob = FALSE, use.all = TRUE)
  # Generate scatterplots to evaluate assignments
  myplot1 <-ggplot(mysample, aes(RS, AL, colour = as.factor(QU))) + geom_point()
  myplot2 <-ggplot(mysample, aes(FS, FA, colour = as.factor(QU))) + geom_point()                                 
  myplot3 <-ggplot(mytestdata, aes(RS, AL, colour = as.factor(QU))) + geom_point()
  myplot4 <-ggplot(mytestdata, aes(FS, FA, colour = as.factor(QU))) + geom_point()                                 
  myKNN <- summary(mypred)
  myplot5 <-ggplot(mytestsubframe, aes(RS, AL, colour = mypred)) + geom_point()                                 
  myplot6 <-ggplot(mytestsubframe, aes(FS, FA, colour = mypred)) + geom_point()
  # Save plot as PDF
  pdf(paste('K Nearest Neighbors scatter ',i,'.pdf'))
  pushViewport(viewport(layout = grid.layout(4, 2,heights=unit(c(1,5,5,5),"null"))))
  grid.text(paste("K Nearest Neighbors ",i), vp = viewport(layout.pos.row = 1, layout.pos.col = 1:2))
  print(myplot1, vp = viewport(layout.pos.row = 2, layout.pos.col = 1))
  print(myplot2, vp = viewport(layout.pos.row = 2, layout.pos.col = 2))
  print(myplot3, vp = viewport(layout.pos.row = 3, layout.pos.col = 1))
  print(myplot4, vp = viewport(layout.pos.row = 3, layout.pos.col = 2))
  print(myplot5, vp = viewport(layout.pos.row = 4, layout.pos.col = 1))
  print(myplot6, vp = viewport(layout.pos.row = 4, layout.pos.col = 2))
  dev.off()
  # Generate confusion matrix
  mymatrix <- confusionMatrix(as.factor(as.integer(mypred)), as.factor(mytestdata$QU))
  print(mymatrix)
  # Store the accuracy in a list
  resultsKNN[i] = mymatrix$overall[1]

  # Perform Naive Bayes using the same test sample
  print(paste("Naive Bayes subset ",i))
  mytest_1 <-  naiveBayes(as.factor(QU)~FA+VA+CA+RS+CH+FS+TS+DE+PH+SU+AL,
                      mysample, laplace = 0)
  mypred_1 <- predict(mytest_1, mytestsubframe, probability = FALSE, decision.values = TRUE)
  myNB <- summary(mytest_1)
  # Generate new scatterplots to see Naive Bayes performance
  myplot5_1 <-ggplot(mytestsubframe, aes(RS, AL, colour = mypred_1)) + geom_point()                                 
  myplot6_1 <-ggplot(mytestsubframe, aes(FS, FA, colour = mypred_1)) + geom_point()
  # Save plot as PDF
  pdf(paste('Naive Bayes scatter ',i,'.pdf'))
  pushViewport(viewport(layout = grid.layout(4, 2,heights=unit(c(1,5,5,5),"null"))))
  grid.text(paste("Naive Bayes ",i), vp = viewport(layout.pos.row = 1, layout.pos.col = 1:2))
  print(myplot1, vp = viewport(layout.pos.row = 2, layout.pos.col = 1))
  print(myplot2, vp = viewport(layout.pos.row = 2, layout.pos.col = 2))
  print(myplot3, vp = viewport(layout.pos.row = 3, layout.pos.col = 1))
  print(myplot4, vp = viewport(layout.pos.row = 3, layout.pos.col = 2))
  print(myplot5_1, vp = viewport(layout.pos.row = 4, layout.pos.col = 1))
  print(myplot6_1, vp = viewport(layout.pos.row = 4, layout.pos.col = 2))
  dev.off()
  # Generate confusion matrix
  mymatrix_1 <- confusionMatrix(as.factor(mypred_1), as.factor(mytestdata$QU))
  print(mymatrix_1)
  # Store the accuracy in a list
  resultsNB[i] = mymatrix_1$overall[1]  

  # Run Linear Discriminant Analysis using the same tests sample
  print(paste("Linear Discriminant Analysis subset  ",i))
  mytest_2 <- lda(QU~FA+VA+CA+RS+CH+FS+TS+DE+PH+SU+AL, mysample)
  mypred_2 <- predict(mytest_2, mytestsubframe, prior = mytest_2$prior, method = c("plug-in", "predictive", "debiased"))
  mypred_2_class = mypred_2$class
  myLDA <- summary(mytest_2)
  # Generate new scatterplots to see LDA performance
  myplot5_2 <-ggplot(mytestsubframe, aes(RS, AL, colour = mypred_2_class)) + geom_point()                                 
  myplot6_2 <-ggplot(mytestsubframe, aes(FS, FA, colour = mypred_2_class)) + geom_point()
  # Save plot as PDF
  pdf(paste('Linear Discriminant Analysis scatter ',i,'.pdf'))
  pushViewport(viewport(layout = grid.layout(4, 2,heights=unit(c(1,5,5,5),"null"))))
  grid.text(paste("LDA ",i), vp = viewport(layout.pos.row = 1, layout.pos.col = 1:2))
  print(myplot1, vp = viewport(layout.pos.row = 2, layout.pos.col = 1))
  print(myplot2, vp = viewport(layout.pos.row = 2, layout.pos.col = 2))
  print(myplot3, vp = viewport(layout.pos.row = 3, layout.pos.col = 1))
  print(myplot4, vp = viewport(layout.pos.row = 3, layout.pos.col = 2))
  print(myplot5_2, vp = viewport(layout.pos.row = 4, layout.pos.col = 1))
  print(myplot6_2, vp = viewport(layout.pos.row = 4, layout.pos.col = 2))
  dev.off()
  # Generate confusion matrix
  mymatrix_2 <- confusionMatrix(as.factor(mypred_2_class), as.factor(mytestdata$QU))
  print(mymatrix_2)
  # Store the accuracy in a list
  resultsLDA[i] = mymatrix_2$overall[1]  
  #LD plot
  pdf(paste('LDA plot',i,'.pdf'))
  print(plot(mytest_2, main=paste('LDA subset',i)))
  dev.off()
}
# Convert results into a vectors
resultsKNN <- unlist(resultsKNN)
resultsNB <- unlist(resultsNB)
resultsLDA <- unlist(resultsLDA)

# Calculate mean
resultsDF <- data.frame(resultsKNN,resultsNB,resultsLDA)
# Summarize the results in terms of mean and standard deviation
model.melt <- melt(resultsDF)
model.summary <- ddply(model.melt,'variable',summarise,mean=mean(value),stdev=sd(value))
# Set up upper and lower error bars
model.summary <- transform(model.summary,lower=mean-stdev,upper=mean+stdev)

# Set up error bar chart
model_bar <- ggplot(model.summary,aes(x=variable,y=mean)) + geom_col(position="dodge") + 
  geom_errorbar(aes(ymax=upper,ymin=lower),position="dodge",data=model.summary) + xlab('model')+
    ylab('accuracy')
#Plot
pdf('Model Comparison 1.pdf')
print(model_bar)
dev.off()
print('Scatterplot guide: The top images show the training subset, the middle images show the testing subset, and the bottom images show the model predictions.')
print('Of the three tested models, K nearest neighbors had the highest mean accuracy, but remains similar to LDA (observable in \'Model comparison 1.pdf\'). Naive Bayes was the least effective method.')
