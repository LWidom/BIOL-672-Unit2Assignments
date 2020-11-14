# Louis Widom
# lpw8274@rit.edu
# Designed in Windows 10
# Last updated 13 November 2020

# List of required packages:
#  ada
#  caret
#  class
#  dplyr
#  e1071
#  ggplot2
#  grid
#  MASS
#  plyr
#  randomForest
#  reshape2
# Associated data files (should be located in the same folder as this script):
#  winequality-red.csv
# ======================================================================================

# Begin Script
library('ada')
library('caret')
library('class')
library('dplyr')
library('e1071')
library('ggplot2')
library('grid')
library('MASS')
library('plyr')
library('randomForest')
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
# Regroup quality into 2 groups (ada makes binary decisions). 1 = "low quality", 2 = "high quality"
wine$Category[wine$quality<6]="1"
wine$Category[wine$quality>5]="2"
wine$Category = factor(wine$Category,levels=c(1,2))
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
resultsForest <- vector(mode="list",length=5)
resultsBoost <- vector(mode="list",length=5)

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
  
  # Run Random Forest algorithm
  print(paste("Random Forest subset ",i))
  mytest <-  randomForest(as.factor(QU)~FA+VA+CA+RS+CH+FS+TS+DE+PH+SU+AL, mysample, ntree = 500)
  mypred <- predict(mytest, mytestsubframe, probability = FALSE, decision.values = TRUE)
  # Generate scatterplots to evaluate assignments
  myplot1 <-ggplot(mysample, aes(RS, AL, colour = as.factor(QU))) + geom_point()
  myplot2 <-ggplot(mysample, aes(FS, FA, colour = as.factor(QU))) + geom_point()                                 
  myplot3 <-ggplot(mytestdata, aes(RS, AL, colour = as.factor(QU))) + geom_point()
  myplot4 <-ggplot(mytestdata, aes(FS, FA, colour = as.factor(QU))) + geom_point()
  myplot5 <-ggplot(mytestsubframe, aes(RS, AL, colour = mypred)) + geom_point()                                 
  myplot6 <-ggplot(mytestsubframe, aes(FS, FA, colour = mypred)) + geom_point()
  # Save plot as PDF
  pdf(paste('Random Forest ',i,'.pdf'))
  pushViewport(viewport(layout = grid.layout(4, 2,heights=unit(c(1,5,5,5),"null"))))
  grid.text(paste("Random Forest ",i), vp = viewport(layout.pos.row = 1, layout.pos.col = 1:2))
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
  resultsForest[i] = mymatrix$overall[1]
  
  # Run adaboost algorithm
  print(paste("adaboost subset ",i))
  mytest_1 <-  ada(QU~FA+VA+CA+RS+CH+FS+TS+DE+PH+SU+AL, mysample, 500) 
  mypred_1 <- predict(mytest_1, mytestsubframe, probability = FALSE, decision.values = TRUE)
  # Generate scatterplots to evaluate assignments
  myplot5_1 <-ggplot(mytestsubframe, aes(RS, AL, colour = mypred_1)) + geom_point()                                 
  myplot6_1 <-ggplot(mytestsubframe, aes(FS, FA, colour = mypred_1)) + geom_point()
  # Save plot as PDF
  pdf(paste('adaboost ',i,'.pdf'))
  pushViewport(viewport(layout = grid.layout(4, 2,heights=unit(c(1,5,5,5),"null"))))
  grid.text(paste("adaboost ",i), vp = viewport(layout.pos.row = 1, layout.pos.col = 1:2))
  print(myplot1, vp = viewport(layout.pos.row = 2, layout.pos.col = 1))
  print(myplot2, vp = viewport(layout.pos.row = 2, layout.pos.col = 2))
  print(myplot3, vp = viewport(layout.pos.row = 3, layout.pos.col = 1))
  print(myplot4, vp = viewport(layout.pos.row = 3, layout.pos.col = 2))
  print(myplot5_1, vp = viewport(layout.pos.row = 4, layout.pos.col = 1))
  print(myplot6_1, vp = viewport(layout.pos.row = 4, layout.pos.col = 2))
  dev.off()
  # Generate confusion matrix
  mymatrix_1 <- confusionMatrix(as.factor(as.integer(mypred_1)), as.factor(mytestdata$QU))
  print(mymatrix_1)
  # Store the accuracy in a list
  resultsBoost[i] = mymatrix_1$overall[1]
  
}
# Convert results into a vectors
resultsForest <- unlist(resultsForest)
resultsBoost <- unlist(resultsBoost)

# Calculate mean
resultsDF <- data.frame(resultsForest,resultsBoost)
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
pdf('Bag vs Boost.pdf')
print(model_bar)
dev.off()
print('Scatterplot guide: The top images show the training subset, the middle images show the testing subset, and the bottom images show the model predictions. Note that quality ratings were only divided into 2 groups (as opposed to 3 in previous scripts).')
print('The Random Forest method performed slightly better than the adaboost algorithm. Its higher average accuracy from cross-validation can be observed in the file \'Bag vs Boost.pdf\'. It also had the highest accuracy out of any single trial, whereas adaboost had the lowest accuracy out of any single trial. However, the Random Forest method had a larger standard deviation than the adaboost algorithm.')