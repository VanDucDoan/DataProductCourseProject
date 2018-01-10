library(shiny)
library(randomcoloR) # library for colors
library(stringr)
library(dplyr)
library(nnet)
library(rpart)
library(rpart.plot)
library(randomForest)
library(e1071)
library(readr)
library(xgboost)
library(caret)

# generate your documentation
generateDocumentation <- function() {
    out.markdown <- ''
    withProgress(message = 'Generating Documentation',
                 value = 0, {
                     out.markdown <- rmarkdown::render(input = "helpDocumentation.Rmd",
                                                       output_format = "html_document")
                     setProgress(1)
                 })
    read_file(out.markdown)
}

# Define server logic
shinyServer(function(input, output) {

    options(shiny.maxRequestSize = 30 * 1024**2)

    # read data from the file uploaded
    loadDataFrame <- reactive({

        req(input$file)

        df <- read.csv(input$file$datapath,
                       header = input$header,
                       sep = input$sep,
                       quote = input$quote)

        return(df)
    })

    # read feature columns
    getLabelColumn <- reactive({

        # label column
        labelColumn <- input$labelColumn
        labelColumn <- as.integer(labelColumn) # convert from character to integer

        # return a column index
        return(labelColumn)
    })

    # read label column
    getFeatureColumns <- reactive({

        # feature columns
        strFeatureColumns <- input$featureColumns
        strFeatureColumns <- strsplit(strFeatureColumns, ",")
        featureColumns <- as.integer(strFeatureColumns[[1]])

        # return a vector of column indices
        return(featureColumns)
    })

    # get classification method
    getClassificationMethod <- reactive({

        # get classification method
        method <- input$classificationMethod

        return(method)
    })

    # get the training set ratio
    getTrainingSetRatio <- reactive({
        ratio <- input$trainingSetRatio
        ratio <- as.numeric(ratio) # convert from string to numeric

        return(ratio)
    })

    # show the dataset when 'Data' tab is chosen
    output$dataFrame <- renderTable({

        # reload the data frame
        dataFrame <- loadDataFrame()

        # temp df
        df <- dataFrame
        df$id <- rownames(df)
        rownames(df) <- seq.int(1, nrow(df), 1)

        if (input$disp == "head") {
            return(head(df))
        }
        else {
            return(df)
        }
    })

    output$summaryData <- renderPrint({

        # reload the data frame
        dataFrame <- loadDataFrame()

        return(summary(dataFrame))
    })

    # plot the data chosen in 2-D when EDA is chosen
    output$plotEDA <- renderPlot({

        # initialize seed
        set.seed(12345)

        # reload dataFrame
        dataFrame <- loadDataFrame()

        # label column
        labelColumn <- getLabelColumn()
        textLabelColumn <- names(dataFrame)[labelColumn]

        # number of classes in dataset
        nClasses <- length(unique(dataFrame[, textLabelColumn]))

        nColors <- 20 # number of colors
        palette <- distinctColorPalette(nColors) # color for each class

        # plot pair of features in dataset
        pairs(dataFrame[, -which(names(dataFrame) == textLabelColumn)],
              main = "Pair graph of features in dataset", pch = 21, col = palette[1:nClasses][unclass(dataFrame[, textLabelColumn])])
        par(xpd = TRUE)
        legend(0.05, 1.05, as.vector(unique(dataFrame[, textLabelColumn])),
               fill = palette[1:nClasses])
    })

    # classification methods
    getAllTrainingValidatingResults <- reactive({

        # reload dataFrame
        dataFrame <- loadDataFrame()

        # label column
        labelColumn <- getLabelColumn()
        textLabelColumn <- names(dataFrame)[labelColumn]

        # feature columns
        featureColumns <- getFeatureColumns()
        textFeatureColumns <- names(dataFrame)[featureColumns]

        # get the training set ratio
        trainingSetRatio <- getTrainingSetRatio()

        # extract the neccesary data from the dataset
        df <- dataFrame %>% select(c(textFeatureColumns, textLabelColumn))
        df[, textLabelColumn] <- as.factor(df[, textLabelColumn])

        # divide the the dataset into training data and validating data
        set.seed(12345)

        train <- sample(nrow(df), trainingSetRatio * nrow(df))
        df.train <- df[train, ]
        df.valid <- df[-train, ]

        # formula for classification
        formulateClass <- as.formula(paste(textLabelColumn, paste(textFeatureColumns, sep = "", collapse = " + "), sep = " ~ "))

        # number of classes in dataset
        classNames <- unique(dataFrame[, textLabelColumn])

        # machine learning algorithms
        method <- getClassificationMethod()
        if (method == "LR") {
            fit.logit <- multinom(formulateClass, data = df.train) # use multi-class logistic regression

            # calculate training prediction and validating prediction
            train.logit.probs <- predict(fit.logit, df.train, "probs") # outputs are probabilities of each class
            valid.logit.probs <- predict(fit.logit, df.valid, "probs") # outputs are probabilities of each class

            # position of class predicted
            train.logit.pos <- apply(train.logit.probs, 1, which.is.max)
            valid.logit.pos <- apply(valid.logit.probs, 1, which.is.max)

            # convert to matrix
            train.logit.pos <- as.matrix(train.logit.pos)
            valid.logit.pos <- as.matrix(valid.logit.pos)

            # convert to name of class predicted
            convertClassNames <- function(index, classNames) {return(classNames[index])}
            train.logit <- apply(train.logit.pos, 1, convertClassNames, classNames)
            valid.logit <- apply(valid.logit.pos, 1, convertClassNames, classNames)

            return(list("model" = fit.logit,
                        "featureColumnNames" = textFeatureColumns,
                        "labelColumnName" = textLabelColumn,
                        "trainData" = df.train,
                        "trainPred" = train.logit,
                        "validData" = df.valid,
                        "validPred" = valid.logit))
        }
        else if (method == "DT") {

            # initialize seed
            set.seed(12345)

            # build decision tree
            fit.dtree <- rpart(formulateClass, data = df.train, method = "class",
                               parms = list(split = "information"))

            # position where xerror is minimum
            xerror.min.pos <- which.is.max(-fit.dtree$cptable[, "xerror"])

            # the range of xerror for choosing appropriate cp
            xerror.min <- fit.dtree$cptable[xerror.min.pos, "xerror"] - fit.dtree$cptable[xerror.min.pos, "xstd"]
            xerror.max <- fit.dtree$cptable[xerror.min.pos, "xerror"] + fit.dtree$cptable[xerror.min.pos, "xstd"]

            # find appropriate cp so that its xerror is is the above range
            goodCP <- fit.dtree$cptable[(fit.dtree$cptable[, "xerror"] >= xerror.min) & (fit.dtree$cptable[, "xerror"] <= xerror.max), "CP"]

            # prune the decision tree
            fit.dtree.pruned <- prune(fit.dtree, cp = goodCP[1])

            # calculate training prediction and validating prediction
            train.dtree.pruned <- predict(fit.dtree.pruned, df.train, type = "class")
            valid.dtree.pruned <- predict(fit.dtree.pruned, df.valid, type = "class")

            return(list("model" = fit.dtree.pruned,
                        "featureColumnNames" = textFeatureColumns,
                        "labelColumnName" = textLabelColumn,
                        "trainData" = df.train,
                        "trainPred" = train.dtree.pruned,
                        "validData" = df.valid,
                        "validPred" = valid.dtree.pruned))
        }
        else if (method == "RF") {

            # initialize seed
            set.seed(12345)

            # forest
            fit.forest <- randomForest(formulateClass, data = df.train, na.action = na.roughfix, importance = TRUE)

            # calculate training prediction and validating prediction
            train.forest <- predict(fit.forest, df.train)
            valid.forest <- predict(fit.forest, df.valid)

            return(list("model" = fit.forest,
                        "featureColumnNames" = textFeatureColumns,
                        "labelColumnName" = textLabelColumn,
                        "trainData" = df.train,
                        "trainPred" = train.forest,
                        "validData" = df.valid,
                        "validPred" = valid.forest))
        }
        else if (method == "SVM") {

            # set seed
            set.seed(12345)

            # call SVM
            fit.svm <- svm(formulateClass, data = df.train)

            # calculate training prediction and validating prediction
            train.svm <- predict(fit.svm, df.train) # calculate probabilities of each class
            valid.svm <- predict(fit.svm, df.valid) # calculate probabilities of each class

            return(list("model" = fit.svm,
                        "featureColumnNames" = textFeatureColumns,
                        "labelColumnName" = textLabelColumn,
                        "trainData" = df.train,
                        "trainPred" = train.svm,
                        "validData" = df.valid,
                        "validPred" = valid.svm))
        }
        else if (method == "XGB") {

            # initialize seed
            set.seed(12345)

            # number of classes in dataset
            nClasses <- length(unique(df.train[, textLabelColumn]))

            # label of output is factor
            label.level <- unique(df.train[, textLabelColumn])

            # convert to number
            value.level <- seq.int(0, nClasses - 1, 1)

            # define a function to convert level.factor to value.level
            label2value <- function(x, labelLevel, valueLevel) { return(valueLevel[labelLevel == x])}

            # training variables and label then make xgb.DMatrix for XGBoost
            train.variables <- as.matrix(df.train[, textFeatureColumns])
            train.label.level <- df.train[, textLabelColumn]
            train.value.level <- apply(as.matrix(train.label.level), 1, label2value, label.level, value.level)
            train.matrix <- xgb.DMatrix(data = train.variables, label = train.value.level)

            # validating variables and label
            valid.variables <- as.matrix(df.valid[, textFeatureColumns])
            valid.label.level <- df.valid[, textLabelColumn]
            valid.value.level <- apply(as.matrix(valid.label.level), 1, label2value, label.level, value.level)
            valid.matrix <- xgb.DMatrix(data = valid.variables, label = valid.value.level)

            # parameters for XGBoost
            xgb_params <- list("objective" = "multi:softprob",
                               "eval_metric" = "mlogloss",
                               "num_class" = nClasses)

            nround <- 50 # number of XGBoost rounds

            # Fit cv.nfold * cv.nround XGB models and save OOF predictions
            fit.xgb <- xgb.train(params = xgb_params, data = train.matrix, nrounds = nround)

            # calculate training prediction and validating prediction
            train.xgb.value.vector <- predict(fit.xgb, newdata = train.matrix)
            valid.xgb.value.vector <- predict(fit.xgb, newdata = valid.matrix)

            # convert back to value level in matrix
            train.xgb.value <- matrix(train.xgb.value.vector, nrow = nClasses, ncol = nrow(train.variables)) %>%
                t() %>% data.frame()
            valid.xgb.value <- matrix(valid.xgb.value.vector, nrow = nClasses, ncol = nrow(valid.variables)) %>%
                t() %>% data.frame()

            # define a function to convert back value.level to level.factor
            value2label <- function(x, valueLevel, labelLevel) { return(labelLevel[valueLevel == x])}

            # convert back to label level
            train.xgb <- apply(data.frame(apply(train.xgb.value, 1, which.is.max) - 1),
                               1, value2label, value.level, label.level)
            valid.xgb <- apply(data.frame(apply(valid.xgb.value, 1, which.is.max) - 1),
                               1, value2label, value.level, label.level)

            return(list("model" = fit.xgb,
                        "featureColumnNames" = textFeatureColumns,
                        "labelColumnName" = textLabelColumn,
                        "trainData" = df.train,
                        "trainPred" = train.xgb,
                        "validData" = df.valid,
                        "validPred" = valid.xgb))
        }
    })

    # training results
    output$trainingResults <- renderPrint({

        # get the training & validating results
        allTrainingValidatingResults <- getAllTrainingValidatingResults()

        # confusion matrix
        confusionMat <- confusionMatrix(allTrainingValidatingResults$trainData[, allTrainingValidatingResults$labelColumnName],
                                        allTrainingValidatingResults$trainPred)

        # label
        label <- as.data.frame(allTrainingValidatingResults$trainData[, allTrainingValidatingResults$labelColumnName])
        names(label) <- c("label")
        label$id <- rownames(allTrainingValidatingResults$trainData)
        rownames(label) <- seq.int(1, nrow(label), 1)

        # predicted
        predicted <- as.data.frame(allTrainingValidatingResults$trainPred)
        names(predicted) <- c("predicted")
        predicted$id <- rownames(allTrainingValidatingResults$trainData)
        rownames(predicted) <- seq.int(1, nrow(predicted), 1)

        # merge label and predicted one
        trainingPred <- merge(label, predicted, by = "id")

        # return the training results
        return(list("summaryModel" = summary(allTrainingValidatingResults$model),
                    "trainingPred" = trainingPred,
                    "confusionMatrix" = confusionMat))
    })

    # plot the training results
    output$trainingPlots <- renderPlot({

        # get the training & validating results
        allTrainingValidatingResults <- getAllTrainingValidatingResults()

        # plot training results
        method <- getClassificationMethod()
        if (method == "LR") {

        }
        else if (method == "DT") {
            prp(allTrainingValidatingResults$model, type = 2, extra = 104,
                fallen.leaves = TRUE, main="Decision Tree")
        }
        else if (method == "RF") {

        }
        else if (method == "SVM") {

        }
    })

    # validating results
    output$validatingResults <- renderPrint({

        # get the training & validating results
        allTrainingValidatingResults <- getAllTrainingValidatingResults()

        # confusion matrix
        confusionMat <- confusionMatrix(allTrainingValidatingResults$validData[, allTrainingValidatingResults$labelColumnName],
                                        allTrainingValidatingResults$validPred)

        # label
        label <- as.data.frame(allTrainingValidatingResults$validData[, allTrainingValidatingResults$labelColumnName])
        names(label) <- c("label")
        label$id <- rownames(allTrainingValidatingResults$validData)
        rownames(label) <- seq.int(1, nrow(label), 1)

        # predicted
        predicted <- as.data.frame(allTrainingValidatingResults$validPred)
        names(predicted) <- c("predicted")
        predicted$id <- rownames(allTrainingValidatingResults$validData)
        rownames(predicted) <- seq.int(1, nrow(predicted), 1)

        # merge label and predicted one
        validatingPred <- merge(label, predicted, by = "id")

        # return the validating results
        return(list("validatingPred" = validatingPred,
                    "confusionMatrix" = confusionMat))
    })

    # plot the labeling results
    output$labelingPlots <- renderPlot({

        # initialize seed
        set.seed(12345)

        # get the training & validating results
        allTrainingValidatingResults <- getAllTrainingValidatingResults()

        # validating data
        validData <- allTrainingValidatingResults$validData

        # number of classes in dataset
        nClasses <- length(unique(validData[, allTrainingValidatingResults$labelColumnName]))

        nColors <- 20 # number of colors
        palette <- distinctColorPalette(nColors) # color for each class

        # plot pair of features in dataset
        pairs(validData[, (names(validData) != allTrainingValidatingResults$labelColumnName)],
              main = "Validating dataset (label)", pch = 21, col = palette[1:nClasses][unclass(validData[, allTrainingValidatingResults$labelColumnName])])
        par(xpd = TRUE)
        legend(0.05, 1.05, as.vector(unique(validData[, allTrainingValidatingResults$labelColumnName])),
               fill = palette[1:nClasses])
    })

    # plot the validating results
    output$validatingPlots <- renderPlot({

        # initialize seed
        set.seed(12345)

        # get the training & validating results
        allTrainingValidatingResults <- getAllTrainingValidatingResults()

        # validating data
        validData <- allTrainingValidatingResults$validData
        validData$id <- rownames(allTrainingValidatingResults$validData)
        rownames(validData) <- seq.int(1, nrow(validData), 1)

        # predicted
        predicted <- as.data.frame(allTrainingValidatingResults$validPred)
        names(predicted) <- c("predicted")
        predicted$id <- rownames(allTrainingValidatingResults$validData)
        rownames(predicted) <- seq.int(1, nrow(predicted), 1)

        # merge data and label
        validatingDataPred <- merge(validData, predicted, by = "id")

        # number of classes in dataset
        nClasses <- length(unique(validatingDataPred[, "predicted"]))

        nColors <- 20 # number of colors
        palette <- distinctColorPalette(nColors) # color for each class

        # plot pair of features in dataset
        pairs(validatingDataPred[, (names(validatingDataPred) != allTrainingValidatingResults$labelColumnName) &
                                     (names(validatingDataPred) != "predicted") & (names(validatingDataPred) != "id")],
              main = "Validating dataset (predicted)", pch = 21, col = palette[1:nClasses][unclass(validatingDataPred[, "predicted"])])
        par(xpd = TRUE)
        legend(0.05, 1.05, as.vector(unique(validatingDataPred[, "predicted"])),
               fill = palette[1:nClasses])
    })

    output$helpDocumentation <- renderText({
        HTML(generateDocumentation())
    })
})
