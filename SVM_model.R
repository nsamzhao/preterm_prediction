SVM_model = function(feature_genes,training_df = subset_train_df,model_kernel = c("linear","polynomial","radial"),gamma = 0.01,validation_proportion = 0.2,set_seed = 100){
  library(e1071)
  library(dismo)
  
  set.seed(set_seed)
  
  # Subset the dataset based on feature genes
  validation_df = list()
  training_dataset = list()
  svm_model = list()
  prediction_df = list()
  pcorrect = list()
  validation_df_num = list()
  
  ## Shuffle the data frame row-wise
  set.seed(set_seed)
  training_df = training_df[sample(1:nrow(training_df)),]
  training_df_num = training_df[,2:101]
  training_df_num = apply(training_df_num, 2, as.numeric)
  training_df = cbind(training_df$Label,training_df_num)
  training_df = as.data.frame(training_df)
  colnames(training_df)[1] = "Label"
  training_df$Label = as.factor(training_df$Label)
  
  ## Stratified cross validation and consider the each subset of test data has the same proportion of the categories 
  folds = kfold(training_df, k=(1/validation_proportion), by=training_df$Label)
  
  for(i in 1:(1/validation_proportion)){
    ## generate validation sets and training sets
    validation_df[[i]] = training_df[folds == i,]
    training_dataset[[i]] = training_df[folds != i,]
    
    # SVM model
    svm_model[[i]] = svm(Label ~ ., data = as.data.frame(training_dataset[[i]]),kernel = model_kernel,gamma = gamma)
    
    ## test the data and generate labels
    prediction_df[[i]] = predict(svm_model[[i]],as.data.frame(validation_df[[i]][,-1]))
    
    ## compare with the true value and calculate the error rate
    pcorrect[i] = length(which(prediction_df[[i]] == validation_df[[i]]$Label))/length(validation_df[[i]]$Label) * 100
    
  }
  
  pcorrect = as.vector(as.numeric(pcorrect))
  index = which(pcorrect == max(pcorrect))
  ## calculate average pcorrect for the model using across validation
  ave_pcorrect = mean(as.vector(as.numeric(pcorrect)))
  
  return(list(svm_model,pcorrect,ave_pcorrect))
}
