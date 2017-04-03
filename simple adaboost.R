#!/usr/bin/env Rscript
## usage: Rscript --vanilla --file=adaboost.R arg_1 arg_2 arg_3
## arg_1 is the desired training error
## arg_2 is the maximaze iterations
## arg_3 is the output file name for preserveing the final classifier

args = commandArgs(trailingOnly=TRUE)

# train data
index <- 1:100
x <- 0:99
y <- rep(c(1,1,1,-1,-1,1,1,1,1,1),10)
weight <- rep(1/length(index),100)
train <- list(index=index,x=x,y=y,weight=weight)

split <- splitPoint(train$x)
maxIteration <- as.integer(args[2])
indexIteration <- 0

# record the basic classifier: split point, split "direction", weight
finalModel <- list(splitpoint = vector("numeric",0), dir = vector("character",0), weight = vector("numeric",0))

# build candidate split points for predictor
splitPoint <- function(x){
    x1 <- x[-length(x)]
    x2 <- x[-1]
    return((x1+x2)/2)
}

# calculate the error rate for each split point
# I record the error rate for two classifiers of each split point, which has been improved in python version.
prediction <- function(traindata,split){
    trainLen <- length(traindata$x)
    pred <- c()
    for(i in 1:(trainLen-1)){
        # output for basic classifier that a sample is positive class if it's bigger than split point
        result_1 <- lapply(traindata$x,function(x){return(ifelse(x > split[i],1,-1))})
        
        # output for basic classifier that a sample is positive class if it's less than split point
        result_2 <- lapply(traindata$x,function(x){return(ifelse(x < split[i],1,-1))})
        
        # record error rate for each classcifier in different part of vector
        pred[i] <- sum(traindata$weight * mapply('!=',as.vector(result_1,"numeric"),traindata$y))
        pred[i+trainLen-1] <- sum(traindata$weight * mapply('!=',as.vector(result_2,"numeric"),traindata$y))
        
    }
    return(pred)
}

# record the basic classifier's split point, split "direction" and weight of model
recordModel <- function(finalModel,pred,split){ 
    baseModelLen <- length(finalModel[[1]])
    indexModel <- which.min(pred)[1]
    
    # find the index of the best split point
    predHalf <-length(pred)/2
    splitIndex <- ifelse(indexModel > predHalf,indexModel-predHalf,indexModel)
    
    # fetch and calculate the parameter of basic classifier
    finalModel$splitpoint[baseModelLen+1] <- split[splitIndex]
    finalModel$dir[baseModelLen+1] <- ifelse(indexModel <= predHalf,'UP','DOWN' )
    finalModel$weight[baseModelLen+1] <- 0.5*log(1/min(pred)-1)
    
    # return the recorded model
    assign(deparse(substitute(newfinalModel)),finalModel,envir=parent.frame())
}

# update the weight of training samples according to the latest basic classifier
reWeight <- function(train,finalModel){
    # the index of latest basic model
    recentModelIndex <- length(finalModel$splitpoint)
    
    # recalculate the output of the latest basic classifier
    prediction <- lapply(train$x,function(x){
        if(finalModel$dir[recentModelIndex] == "UP")
            return(ifelse(x > finalModel$splitpoint[recentModelIndex],1,-1))
        else
            return(ifelse(x < finalModel$splitpoint[recentModelIndex],1,-1))
    })
    
    # calculate the new weight
    newWeight <- train$weight * exp(-0.5*log(1/min(pred)-1)*train$y*as.vector(prediction,"numeric"))
    
    # normalize and update the weight of samples
    newWeight <- newWeight/sum(newWeight)
    train$weight <- newWeight
    assign(deparse(substitute(newtrain)),train,envir=parent.frame())
}

# return the output of the final classifier based on those basic classifiers we have built
finalClassifier <- function(x,finalModel){
    seperate <- 0
    
    # walk through every basic classifier in _finalModel_ to predict the output of _x_
    for(i in 1:length(finalModel$splitpoint)){
        # accumulate the weighted output of each basic model
        if(finalModel$dir[i] == "UP")
            seperate <- seperate + finalModel$weight[i] * ifelse(x > finalModel$splitpoint[i],1,-1)
        else
            seperate <- seperate + finalModel$weight[i] * ifelse(x < finalModel$splitpoint[i],1,-1)
    }
    return(sign(seperate))
}

# calculate the training error of the  final classifier
trainError <- function(train,finalClassifier){
    trainResult <- lapply(train$x,finalClassifier,finalModel=finalModel)
    error <- sum(mapply("!=",train$y,as.vector(trainResult))) / length(train$x)
    return(error)
}

# the main loop
repeat{ 
    pred <- prediction(train,split)
    recordModel(finalModel,pred,split)
    finalModel <- newfinalModel
    rm("newfinalModel")
    reWeight(train,finalModel)
    train <- newtrain
    rm("newtrain")
    
    indexIteration <- indexIteration + 1
    if(trainError(train,finalClassifier) < as.numeric(args[1]) || indexIteration == maxIteration){
        write.csv(finalModel,file=args[3],row.names=F)
        break
    }
}
