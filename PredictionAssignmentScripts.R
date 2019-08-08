library(caret)
library(gbm)
library(ggplot2)
library(randomForest)

# Load data

baseURL      <- "https://d396qusza40orc.cloudfront.net/predmachlearn/"
trainingFile <- "pml-training.csv"
testingFile  <- "pml-testing.csv"

sourceData <- function(fname) {
    fpath <- paste0("./data/",fname)
    furl  <- paste0(baseURL,fname)
    if (!file.exists("./data"))  { dir.create("./data") }
    if (!file.exists(fpath))     { download.file(furl, fpath, method = "curl") }
    read.csv(fpath,na.strings = c("","NA","#DIV/0!"))
}

trainingSet <- sourceData(trainingFile)
testingSet  <- sourceData(testingFile)

# Drop majority NAs
allNA    <- apply(trainingSet,2,function(x) sum(is.na(x)))
allNA    <- allNA > 0.97*length(trainingSet$X)
allNA[1] <- TRUE # Patch X out
trainingSet <- trainingSet[,!allNA]
testingSet  <- testingSet[,!allNA]

# Drop variables with no predcitive value
noPred<-which(names(trainingSet) %in% c("X","user_name","raw_timestamp_part_1","raw_timestamp_part_2","cvtd_timestamp","new_window"))
trainingSet <- trainingSet[,-noPred]
testingSet  <- testingSet[,-noPred]

set.seed(12358)

# Reshuflle data set
trainingSet <- trainingSet[sample(length(trainingSet$classe),1000),] ## ACHTUNG REMOVE TRIM


# Pre-train subset

inSubSet    <- createDataPartition(trainingSet$classe, p = 0.1, list=FALSE)
subSet      <- trainingSet[inSubSet,]
inPreTrain  <- createDataPartition(subSet$classe, p=0.8, list=FALSE)
preTrainSet <- subSet[ inPreTrain,]
preTestSet  <- subSet[-inPreTrain,]

# Exploration

q1 <- qplot(roll_arm,data=trainingSet,geom="density", col=classe)
q2 <- qplot(pitch_arm,data=trainingSet,geom="density", col=classe)
q3 <- qplot(yaw_arm,data=trainingSet,geom="density", col=classe)
q4 <- qplot(total_accel_arm,data=trainingSet,geom="density", col=classe)
grid.arrange(q1, q2, q3, q4, nrow=2, ncol=2)

smpset <- sample(length(trainingSet$classe),1000)
edaset <- grep("roll_",names(trainingSet))
f1 <- featurePlot(x=trainingSet[smpset,edaset],y=trainingSet$classe[smpset],plot="pairs")
edaset <- grep("_forearm$",names(trainingSet))
f2 <- featurePlot(x=trainingSet[smpset,edaset],y=trainingSet$classe[smpset],plot="pairs")
edaset <- grep("_dumbbell_z",names(trainingSet))
f3 <- featurePlot(x=trainingSet[smpset,edaset],y=trainingSet$classe[smpset],plot="pairs")
grid.arrange(f1, f2, f3, nrow=1, ncol=3)

# Cross validation set-up: determine best mtry

resultCV <- rfcv(preTrainSet[,-54],preTrainSet$classe)
with(resultCV, plot(n.var, error.cv, log="x", type="o", lwd=2))
mtry     <- resultCV$n.var[which.min(resultCV$error.cv)]
tgrid    <- expand.grid(.mtry = mtry)
control  <- trainControl(method="repeatedcv", number=10, repeats=3)

# Training subset

modelRF  <- train(classe~., data=preTrainSet, method="rf",
                metric = "Accuracy", trControl = control, tuneGrid = tgrid, prox = TRUE)
predRF   <- predict(modelRF, preTestSet)

modelGBM <- train(classe~., data=preTrainSet, method="gbm",
                metric = "Accuracy", trControl = control, verbose = FALSE)
predGBM  <- predict(modelGBM,preTestSet)

comparison <- resamples(list(RF=modelRF,BOOSTING=modelGBM))
summary(comparison)

# Training  - final

inFinal    <- createDataPartition(trainingSet$classe, p = 0.8, list=FALSE)
trainFinal <- trainingSet[ inFinal,]
 testFinal <- trainingSet[-inFinal,]

model      <- train(classe~., data=trainFinal, method="rf",
                    metric = "Accuracy", trControl = control, tuneGrid = tgrid, prox = TRUE)
validation <- predict(model,testFinal)

insample   <- confusionMatrix(validation,testFinal$classe)

insample$overall

# Prediction - final

prediction <- predict(model, testingSet)
resultset  <- data.frame(problem_id=testingSet$problem_id, classe = prediction)
print(resultset)