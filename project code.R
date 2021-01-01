# business analytics project.
#install.packages("skimr")
#install.packages("tidyverse")
#install.packages("funModeling")
#install.packages("Hmisc")
#install.packages("MASS")
#install.packages("randomForest")
#install.packages("ROCR")
#install.packages("pROC")
library(funModeling) 
library(tidyverse) 
library(Hmisc)
library(skimr)
library(MASS)
library(randomForest)
library(class)
library(e1071)
library(pROC)

#-----------------------------------------------------------------------------
rm(list=ls())
#data importing
train=read.csv("C:/Users/hp/OneDrive/Desktop/IME PROJECT/train.csv",header = T,sep=",")
test=read.csv("C:/Users/hp/OneDrive/Desktop/IME PROJECT/test.csv",header = T,sep=",")
#-------------------------------------------------------------------------------
# Exploratory Data Analysis
head(train)
dim(train)  # 200   21
dim(test)
glimpse(train) # clearly respons variable y is categorical and  
# all predictor are float type
print(status(train))
plot_num(train)
skim(train)   #  clearly no missing observation
summary(train)
#-------------------------------------------------------------------------------
p=dim(train)[2]  # no of features =21
n=dim(train)[1]  # total no observation=200
Y_train=train$y   # response variable
X_train=train[,-which(colnames(train)=='y')]
Y_test=test$y
X_test=test[,-which(colnames(test)=='y')]
cor(X_train,Y_train)
error_matrix=matrix(0,nrow=2,ncol=6)
colnames(error_matrix)=c("Logistic","LDA","QDA", "SVM (kernel=linear)","KNN (k=83)","Random forest")
rownames(error_matrix)=c("train error","test error")
error=function(Ytrain_hat,Ytest_hat){
  train_error=mean(abs(Y_train-Ytrain_hat))
  test_error=mean(abs(Y_test-Ytest_hat))
  return(c("train error"=train_error,"test error"=test_error))
}
par(mfrow=c(1,1))
#-------------------------------Logistic----------------------------------------
model_log=glm(y~.,data=train,family = binomial)
summary(model_log)
train_pred=predict(model_log,type="response")
train_pred[train_pred>=0.5]=1
train_pred[train_pred<0.5]=0  # train_pred gives predicted class level on train set
test_pred=predict(model_log,newdata = test,type="response")
test_pred[test_pred>=0.5]=1
test_pred[test_pred<0.5]=0
train_error = error(train_pred,test_pred)[1]
test_error = error(train_pred,test_pred)[2]
error_matrix[,1]=c(train_error,test_error)
log_confusion_mat=table(test_pred,Y_test)    # THIS IS FOR TEST DATA
train_error
test_error
log_confusion_mat
attributes(test_pred)
#-----------------------------LDA-----------------------------------------------
model_lda=lda(y~.,data=train)
lda_pred_train=predict(model_lda,type="response")
lda_pred_test=predict(model_lda,newdata = test,type="response")
# the "class" component of lda prediction model contains LDA'S prediction
lda_train_pred_class=lda_pred_train$class
lda_test_pred_class=lda_pred_test$class

lda_train_error=mean(Y_train !=lda_train_pred_class)  # 0.1
lda_test_error=mean(Y_test!=lda_test_pred_class)     # 0.158
lda_confusion_mat=table(lda_test_pred_class,Y_test)
error_matrix[,2]=c(lda_train_error,lda_test_error)

lda_train_error
lda_test_error
lda_confusion_mat
#-----------------------------QDA-----------------------------------------------
model_qda=qda(y~.,data=train)
qda_pred_train=predict(model_qda,type="response")
qda_pred_test=predict(model_qda,newdata = test,type="response")
# the "class" component of qda prediction model contains qDA'S prediction
train_pred_class=qda_pred_train$class
test_pred_class=qda_pred_test$class
qda_train_error=mean(Y_train !=train_pred_class)  # 0.07
qda_test_error=mean(Y_test!=test_pred_class)     # 0.258
qda_confusion_mat=table(test_pred_class,Y_test)
error_matrix[,3]=c(qda_train_error,qda_test_error)
qda_train_error
qda_test_error
qda_confusion_mat
#-------------------------random Forest----------------------------------------
set.seed(111)
train$y=as.factor(train$y)  # convert response variable as factor
model_forest=randomForest(y~.,data=train)  #fit random forest model
model_forest
pred_forest_train=ifelse(as.numeric(predict(model_forest,newdata=X_train))==1,0,1)  # prediction of training set
pred_forest_test=ifelse(as.numeric(predict(model_forest,newdata=X_test))==1,0,1)  #prediction of test set
train_error_forest=mean(abs(Y_train-pred_forest_train)) # error=0.0
test_error_forest=mean(abs(Y_test-pred_forest_test))    #error=0.218
forest_confusion_mat=table(Y_test,pred_forest_test)  #confusion matrix
forest_confusion_mat
error_matrix[,6]=c(train_error_forest,test_error_forest)
train_error_forest
test_error_forest
forest_confusion_mat
##------------------------------K_NN--------------------------------------------
set.seed(121)
head(train)  #all the Data are scaled there is no need of scaling 
NROW(Y_train)# to find the number of observation
k=round(sqrt(NROW(Y_train))) #Generally k takes as square root of number of training data set
model_knn_train=knn(train=X_train,test=X_train,cl=Y_train,k=k) #fitting the train data set
model_knn_test=knn(train=X_train,test=X_test,cl=Y_train,k=k)  #fitting the test data set
pred_knn_train=ifelse(as.numeric(model_knn_train)==1,0,1)
pred_knn_test=ifelse(as.numeric(model_knn_test)==1,0,1)
train_error_knn=mean(abs(Y_train-pred_knn_train))#0.205
test_error_knn=mean(abs(Y_test-pred_knn_test)) #0.283
knn_confusion_mat=table(Y_test,pred_knn_test)  #confusion matrix
knn_confusion_mat  #71.7 Accurary
#optimal number of k  
i=1
k_opt=numeric(length=150)
for(i in 1:150){
  model_knn=knn(train=X_train,test=X_test,cl=Y_train,k=i)
  k_opt[i]=100*sum(Y_test==model_knn)/NROW(Y_test)
  k=i
  cat(k,"=",k_opt[i],'\n')
}
plot(k_opt,type="b",xlab="k_value",ylab="Accuracy")  

##optimal value of k=83
model_knn_train=knn(train=X_train,test=X_train,cl=Y_train,k=83) #fitting the train data set
model_knn_test=knn(train=X_train,test=X_test,cl=Y_train,k=83)  #fitting the test data set
pred_knn_train=ifelse(as.numeric(model_knn_train)==1,0,1)
pred_knn_test=ifelse(as.numeric(model_knn_test)==1,0,1)
train_error_knn_optimum=mean(abs(Y_train-pred_knn_train)) #0.23
test_error_knn_optimum=mean(abs(Y_test-pred_knn_test)) #0.225
knn_confusion_mat=table(Y_test,pred_knn_test)  #confusion matrix
knn_confusion_mat  #accuracy 77.5
error_matrix[,5]=c(train_error_knn_optimum,test_error_knn_optimum)
train_error_knn_optimum
test_error_knn_optimum
knn_confusion_mat

##-----------------------------------------SVM---------------------------------------
# svm, when kernel is linear
model_svm_train=svm(y~.,data=train,kernel="linear",cost=2)
svm_pred_train=predict(model_svm_train,type='response')
svm_pred_test=predict(model_svm_train,newdata=test,type='response')
svm_train_error=mean(svm_pred_train!=Y_train)  #0.1
svm_test_error=mean(svm_pred_test!=Y_test)   # 0.174
svm_confusion_matrix=table(svm_pred_test,Y_test)
svm_confusion_matrix
svm_train_error
svm_test_error
svm_confusion_matrix

# svm when kernel is radial
model_svm_train1=svm(y~.,data=train,kernel="radial",cost=0.5)
svm_pred_train1=predict(model_svm_train1,type='response')
svm_pred_test1=predict(model_svm_train1,newdata=test,type='response')
svm_train_error1=mean(svm_pred_train1!=Y_train)     #  0.05
svm_test_error1=mean(svm_pred_test1!=Y_test)       #  .188
svm_confusion_matrix1=table(svm_pred_test1,Y_test)
svm_confusion_matrix1

# svm when kernel is polynomial
model_svm_train2=svm(y~.,data=train,kernel="polynomial",cost=0.5)
svm_pred_train2=predict(model_svm_train2,type='response')
svm_pred_test2=predict(model_svm_train2,newdata=test,type='response')
svm_train_error2=mean(svm_pred_train2!=Y_train)     #  0.03
svm_test_error2=mean(svm_pred_test2!=Y_test)       #  .248
svm_confusion_matrix2=table(svm_pred_test2,Y_test)
svm_confusion_matrix2
error_matrix[,4]=c(svm_train_error,svm_test_error)
accuray_mat=(1-error_matrix)*100

#---------------------------------------------------------Comparision of Models-------------------------------------------------------
# comparision of Error and accuracy of different models

accuray_mat=(1-error_matrix)*100
row.names(accuray_mat)=c("training accuracy","test accuracy")
error_matrix    # contain misclassification error of different model on test set and train set
accuray_mat     # contain Accuracy of different model on test set and train set
par(mfrow=c(1,1))
test.error=error_matrix[2,]
train.error=error_matrix[1,]
error=rbind(test.error,train.error)
barplot(error,beside=T,col=c("darkblue","red"),legend = rownames(error),width = 0.5,xlab = "models",ylab = "Error",
        main = "comparision of test and train error for different models")
barplot(test.error,beside=T,xlab = "models",ylab = "Error",
        main = "comparision of test error for different models")
test_accuracy=1-test.error
barplot(test_accuracy,col="green",beside=T,xlab = "models",ylab = "Error",
        main = "comparision of accuracy for different models")

#---------------------------------------------------------SOME Explination Regarding LDA-------------------------------------------------
#  why LDA Performing Well
p_val=numeric(length=20)   # this will contain p value of shapiro test for each predictor
par(mfrow=c(2,5))
for(i in 1:20){
  
  plot(density(train[,i]))
  p_val[i]=shapiro.test(train[,i])
  
}
par(mfrow=c(1,1))
cor(X_train)
t(p_val)        # contain p val of shapiro test of all predictors
# split of train data according to class specific (class 0 and 1)
train_0=train[train$y==0,]
train_1=train[train$y==1,]
var_com=(var(train_0)-var(train_1))  # comparision of class specific variance 
                                      # matrix which shows both matrix are approx same


#------------------------------------------------------------ROC  CURVE---------------------------------
model_lda=lda(y~.,data=train)
lda_pred_train=predict(model_lda,type="response")
lda_pred_test=predict(model_lda,newdata = test,type="response")
library(ROCR)
pred <- prediction(as.numeric(lda_pred_test$posterior[,2]),as.numeric(test$y)) 
perf<-ROCR::performance(pred,"tpr","fpr")
plot(perf,col="blue",main="Comparision of ROC",print.auc=T)

# comparision of ROC Curve of LDA, LOGISTIC AND RANDOM FOREST
roc(Y_test,as.numeric(lda_test_pred_class))  # FOR LDA

roc(Y_test,as.numeric(test_pred))  # FOR LOGISTIC
roc(Y_test,as.numeric(qda_pred_test$posterior[,1]))       # random Forest
plot(perf,col="blue",main="comparision of ROC curve",add=T)  #LDA
# Logistic
model_log=glm(y~.,data=train,family = binomial)
test_pred=predict(model_log,newdata = test,type="response")
pred1= prediction(test_pred,test$y)
perf<-ROCR::performance(pred1,"tpr","fpr")
plot(perf,col="red",main="ROC courve for Logistic",print.auc=T,add=T)

# QDA
model_qda=qda(y~.,data=train)
qda_pred_test=predict(model_qda,newdata = test,type="response")
pred2= prediction(qda_pred_test$posterior[,2],test$y)
perf<-ROCR::performance(pred2,"tpr","fpr")
plot(perf,col="green",main="ROC courve for QDA",print.auc=T, add=T)

legend("bottomright",legend=c("LDA","LOGISTIC","QDA"),col=c("blue","red","green"),lwd=4)
