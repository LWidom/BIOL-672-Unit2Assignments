# Louis Widom
# lpw8274@rit.edu
# Designed in Windows 10
# Last updated 15 November 2020

# List of required packages:
#  caret
#  dplyr
#  e1071
#  ggplot2
#  grid
#  keras
#  MASS
#  neuralnet
#  plyr
#  reshape2
#  tensorflow
#  tictoc
# Associated data files (should be located in the same folder as this script):
#  winequality-red.csv
# ======================================================================================

# Begin Script
library('caret')
library('dplyr')
library('e1071')
library('ggplot2')
library('grid')
library('keras')
library('MASS')
library('neuralnet')
library('plyr')
library('reshape2')
library('tensorflow')
library('tictoc')
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
# Regroup quality into 3 numerical groups. 1 = "poor", 2 = "average", 3 = "excellent"
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
# Initialize lists to store the accuracies of each iteration
resultsNeuralNet <- vector(mode="list",length=5)

# This DOES NOT loop through each each of the split frames to perform cross-validation due to lengthy runtimes
i=1
# Establish training datasets
index1 = 2
mysample = split_frame[[index1]]
# Join remaining split frames
for (j in 1:3){
  index2 = index1 + j
  mysample = rbind(mysample,split_frame[[index2]])
}
# Establish testing dataset and a subframe without the QU column
mytestdata = split_frame[[i]]
mytestsubframe = subset(mytestdata,select=-QU)
  
# Run neural network classifier
print(paste('Neural Net subset ',i))
mytest <- neuralnet(QU~FA+VA+CA+RS+CH+FS+TS+DE+PH+SU+AL, mysample, hidden = 3, linear.output = FALSE, stepmax=1e+06)
print(plot(mytest))
mypred <- predict(mytest, mytestsubframe, rep = 1, all.units = FALSE)
# Generate scatterplots to see the neural network's performance
myplot1 <-ggplot(mysample, aes(RS, AL, colour = as.factor(QU))) + geom_point()
myplot2 <-ggplot(mysample, aes(FS, FA, colour = as.factor(QU))) + geom_point()                                 
myplot3 <-ggplot(mytestdata, aes(RS, AL, colour = as.factor(QU))) + geom_point()
myplot4 <-ggplot(mytestdata, aes(FS, FA, colour = as.factor(QU))) + geom_point()                       
myplot5 <-ggplot(mytestsubframe, aes(RS, AL, colour = max.col(mypred))) + geom_point()                                 
myplot6 <-ggplot(mytestsubframe, aes(FS, FA, colour = max.col(mypred))) + geom_point()
# Save plot as PDF
pdf(paste('Neural Net subset ',i,'.pdf'))
pushViewport(viewport(layout = grid.layout(4, 2,heights=unit(c(1,5,5,5),"null"))))
grid.text(paste("Neural Net ",i), vp = viewport(layout.pos.row = 1, layout.pos.col = 1:2))
print(myplot1, vp = viewport(layout.pos.row = 2, layout.pos.col = 1))
print(myplot2, vp = viewport(layout.pos.row = 2, layout.pos.col = 2))
print(myplot3, vp = viewport(layout.pos.row = 3, layout.pos.col = 1))
print(myplot4, vp = viewport(layout.pos.row = 3, layout.pos.col = 2))
print(myplot5, vp = viewport(layout.pos.row = 4, layout.pos.col = 1))
print(myplot6, vp = viewport(layout.pos.row = 4, layout.pos.col = 2))
dev.off()
# Generate confusion matrix
mymatrix <- confusionMatrix(as.factor(max.col(mypred)), as.factor(as.integer(mytestdata$QU)))
print(mymatrix)
print(plot(mytest))
print('Note that due to the complexity of the data, the neuralnet function frequently had difficulty converging on a solution. I compensated by adjusting the stepmax parameter, but this led to long runtimes. I have decided to remove 5-fold cross-validation from this script as a result.')
  
# Build deep learning network with Keras and Tensorflow
print('Deep Net with Keras and Tensorflow')
# Load cifar10 image dataset from keras
cifar10 <- dataset_cifar10() # 60,000 color images of 10 classes of ojects and animals
c(train_images, train_labels) %<-% cifar10$train
c(test_images, test_labels) %<-% cifar10$test
class_names = c('airplane',
                'automobile',
                'bird',
                'cat',
                'deer', 
                'dog',
                'frog',
                'horse',
                'ship',
                'truck')  # Class names for plotting
  
# Scale pixels from range (0 to 255) to range (0 to 1)
train_images <- train_images / 255
test_images <- test_images / 255
  
# Display first 25 images from data set and check format
par(mfcol=c(5,5))
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')
for (i in 1:25) { 
  img <- train_images[i, , , ]
  # Create a matrix to hold the RGB values
  rgb.mat<-array(dim=c(32,32,3))
  rgb.mat[,,1]<-img[,,1]
  rgb.mat[,,2]<-img[,,2]
  rgb.mat[,,3]<-img[,,3]
  # Plot the images
  plot.new()
  pre_image <-rasterImage(rgb.mat,0,0,1,1,xaxt='n',yaxt='n') + title(paste(class_names[train_labels[i] + 1]))
}
  
# Build the deep net model
# Measure the speed
tic("time elapsed")
# Initialize list to store the model's accuracy
resultsDeepNet <- vector(mode="list",length=5)
# Loop five times to examine the effects of added layers
for (j in 1:5) {
  print(paste('Deep Net with ',j,' hidden layer(s):'))
  model <- keras_model_sequential()
  # Add a new convolutional 2D layer each time through the loop
  if (j==1){
    model %>%
      layer_conv_2d(filter = 32, kernel_size = c(3,3),padding = "same",input_shape = c(32, 32, 3), activation = 'relu') %>%
      layer_flatten() %>%
      layer_dense(units = 10, activation = 'softmax')
  }
  if (j==2){
    model %>%
      layer_conv_2d(filter = 32, kernel_size = c(3,3),padding = "same",input_shape = c(32, 32, 3), activation = 'relu') %>%
      layer_conv_2d(filter = 32, kernel_size = c(3,3), activation = 'relu') %>%
      layer_flatten() %>%
      layer_dense(units = 10, activation = 'softmax')
  }
  if (j==3){
    model %>%
      layer_conv_2d(filter = 32, kernel_size = c(3,3),padding = "same",input_shape = c(32, 32, 3), activation = 'relu') %>%
      layer_conv_2d(filter = 32, kernel_size = c(3,3), activation = 'relu') %>%
      layer_conv_2d(filter = 32, kernel_size = c(3,3), activation = 'relu') %>%
      layer_flatten() %>%
      layer_dense(units = 10, activation = 'softmax')
  }
  if (j==4){
    model %>%
      layer_conv_2d(filter = 32, kernel_size = c(3,3),padding = "same",input_shape = c(32, 32, 3), activation = 'relu') %>%
      layer_conv_2d(filter = 32, kernel_size = c(3,3), activation = 'relu') %>%
      layer_conv_2d(filter = 32, kernel_size = c(3,3), activation = 'relu') %>%
      layer_conv_2d(filter = 32, kernel_size = c(3,3), activation = 'relu') %>%
      layer_flatten() %>%
      layer_dense(units = 10, activation = 'softmax')
  }
  if (j==5){
    model %>%
      layer_conv_2d(filter = 32, kernel_size = c(3,3),padding = "same",input_shape = c(32, 32, 3), activation = 'relu') %>%
      layer_conv_2d(filter = 32, kernel_size = c(3,3), activation = 'relu') %>%
      layer_conv_2d(filter = 32, kernel_size = c(3,3), activation = 'relu') %>%
      layer_conv_2d(filter = 32, kernel_size = c(3,3), activation = 'relu') %>%
      layer_conv_2d(filter = 32, kernel_size = c(3,3), activation = 'relu') %>%
      layer_flatten() %>%
      layer_dense(units = 10, activation = 'softmax')
  }
  # Compile the model
  model %>% compile(
    optimizer = 'adam', 
    loss = 'sparse_categorical_crossentropy',
    metrics = c('accuracy')
  )
  
  # Train model and summarize it 
  model %>% fit(train_images, train_labels, epochs = 10, verbose = 2)
  print(model)
  
  # Make probabilistic predictions
  predictions <- model %>% predict(test_images)
  # Get class predictions
  class_pred <- model %>% predict_classes(test_images)
  
  # Plot some classifications
  par(mfcol=c(5,5))
  par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')
  for (i in 1:25) { 
    img <- test_images[i, , , ]
    # Subtract 1 as labels go from 0 to 9
    predicted_label <- which.max(predictions[i, ]) - 1
    true_label <- test_labels[i]
    if (predicted_label == true_label) {
      color <- '#008800' # Green font
    } else {
      color <- '#bb0000' # Red font
    }
    # Create a matrix to hold the RGB values
    rgb.mat<-array(dim=c(32,32,3))
    rgb.mat[,,1]<-img[,,1]
    rgb.mat[,,2]<-img[,,2]
    rgb.mat[,,3]<-img[,,3]
    # Plot images with their predictions (correct in green, incorrect in red)
    plot.new()
    post_image <-rasterImage(rgb.mat,0,0,1,1,xaxt='n',yaxt='n') +
      title(paste0(class_names[predicted_label + 1]," (",class_names[true_label +1],")"),col.main=color)
  }
  
  # Generate Confusion matrix
  mymatrix_1 <- confusionMatrix(as.factor(class_pred), as.factor(test_labels))
  print(mymatrix_1)
  # Store the accuracy in a list
  resultsDeepNet[j] = mymatrix_1$overall[1]
  }
toc() # Print elapsed time
  
# Plot the accuracy as a function of the number of hidden layers
Accuracies <- unlist(resultsDeepNet)
Number_of_Hidden_Layers <- 1:5
DeepNetAccuracy1 <- data.frame(Number_of_Hidden_Layers,Accuracies)
DeepNetPlot <- ggplot(DeepNetAccuracy1, aes(x=Number_of_Hidden_Layers, y=Accuracies, group=1)) +
  geom_line() + geom_point()
print(DeepNetPlot)
print('Accuracy appeared to be roughly unaffected by the number of computing cores. It tended to improve with additional hidden layers, as can be seen in the penultimate plot.')
  
# Plot the speed as a function of the number of computer cores
cores <- c(2,2,2,4,4,4)
speeds <- c(9514.9,9074.22,9080.55,4597.72,4565.88,4591.63)
DeepNetSpeed1 <- data.frame(cores,speeds)
# Summarize the results in terms of mean and standard deviation
DeepNetSpeed1.summary <- ddply(DeepNetSpeed1,'cores',summarise,mean=mean(speeds),stdev=sd(speeds))
# Set up upper and lower error bars
DeepNetSpeed1.summary <- transform(DeepNetSpeed1.summary,lower=mean-stdev,upper=mean+stdev)
DeepNetSpeedPlot <- ggplot(DeepNetSpeed1.summary,aes(x=cores,y=mean)) + geom_col(position="dodge") + 
  geom_errorbar(aes(ymax=upper,ymin=lower),position="dodge",data=DeepNetSpeed1.summary) + xlab('# of Cores')+
  ylab('Time (sec)') + scale_x_discrete(limits=c('','2','','4'))
print(DeepNetSpeedPlot)
print('The time it took to train and test all 5 deep learning networks (with increasing numbers of hidden layers) was evaluated in triplicate on a laptop with an Intel i5-5200U (2 cores) and a desktop with an Intel i7-6700 (4 cores). As can be seen in the final plot, doubling the number of cores approximately halved the time.')
