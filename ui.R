library(shiny)

# define UI for application
shinyUI(fluidPage(

    # application title
    titlePanel("Typical Machine Learning Algorithms for Classification Problems"),

    # sidebar layout for loading data file
    sidebarLayout(

        # sidebar panel for inputs
        sidebarPanel(

            # select a file to upload
            fileInput("file", "Load data in *.CSV format",
                      multiple = TRUE,
                      accept = c("text/csv",
                                 "text/comma-separated-values,text/plain",
                                 ".csv")),

            # checkbox if file has header
            checkboxInput("header", "Header", TRUE),

            # select separator
            radioButtons("sep", "Separator",
                         choices = c(Comma = ",",
                                     Semicolon = ";",
                                     Tab = "\t"),
                         selected = ","),

            # select quotes
            radioButtons("quote", "Quote",
                         choices = c(None = "",
                                     "Double Quote" = '"',
                                     "Single Quote" = "'"),
                         selected = '"'),

            # select number of rows to display
            radioButtons("disp", "Display",
                         choices = c(Head = "head",
                                     All = "all"),
                         selected = "head"),

            # horizontal line
            tags$hr(style = "border-color: purple;"),

            # select feature columns
            textInput("featureColumns", label = "Feature Columns", value = "1,2,3,4"),

            # select label columns
            textInput("labelColumn", label = "Label Column", value = "5"),

            # horizontal line
            tags$hr(style = "border-color: purple;"),

            # choose classification methods
            radioButtons("classificationMethod", "Classification Method",
                         choices = c("Logistic Regression" = "LR",
                                     "Decision Tree" = "DT",
                                     "Random Forest" = "RF",
                                     "Support Vector Machine" = "SVM",
                                     "XGBoost" = "XGB"),
                         selected = "LR"),

            # select label columns
            textInput("trainingSetRatio", label = "Training Set Ratio", value = "0.7"),

            # push Submit when parameter settings are finished
            submitButton("Submit")
        ),

        # sidebar panel for selecting features and parameters of algorithms

        # main panel for displaying data, EDA results, and classification results
        mainPanel(
            tabsetPanel(type = "tabs",
                        tabPanel("Data Visualization",
                                 tableOutput("dataFrame"),
                                 verbatimTextOutput("summaryData")),
                        tabPanel("Exploratory Data Analysis", plotOutput("plotEDA")),
                        tabPanel("Training Results",
                                 verbatimTextOutput("trainingResults"),
                                 plotOutput("trainingPlots")),
                        tabPanel("Validating Results",
                                 verbatimTextOutput("validatingResults"),
                                 plotOutput("labelingPlots"),
                                 plotOutput("validatingPlots")),
                        tabPanel("Help", htmlOutput("helpDocumentation"))
            )
        )

    )
))
