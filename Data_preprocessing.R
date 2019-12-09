library(tidyr)
library(dplyr)
library(ggplot2)
# Read the data
load("~/Desktop/EECS 545/545Project/eset_SC2_v20.RData")
#load("~/Desktop/545Project/HTA20_RMA.RData")

# Separate training and test dataset (435 samples for training)
train_meta = subset(anoSC2,subset = anoSC2$Train == 1)
train_data = esetSC2[,which(colnames(esetSC2) %in% train_meta$SampleID)]

# Sanity check for training data
# How many samples in each group
train_meta %>% group_by(Group) %>% summarise(total_num = n())
train_meta %>% group_by(Platform) %>% summarise(total_num = n())
train_meta %>% group_by(Platform,Group) %>% summarise(total_num = n())

# Quickly PCA to test whether Group or Platform dominate the results
pca_results = prcomp(t(train_data),center = TRUE,scale. = TRUE)
pca_results_df = pca_results$x
all.equal(rownames(pca_results_df),train_meta$SampleID)
pca_visualization = cbind(pca_results_df,train_meta)

ggplot(pca_visualization,aes(PC1,PC2,color = Group)) + geom_point()
ggplot(pca_visualization,aes(PC1,PC2,color = Platform)) + geom_point()
ggplot(pca_visualization,aes(PC1,PC2,color = GA)) + geom_point()

# Quickly PCA to test whether train and test data are separated
# First classification(Control VS SPTD)
train_meta1 = subset(train_meta,train_meta$Group != "PPROM")
train_data1 = train_data[,which(colnames(train_data) %in% train_meta1$SampleID)]

## Feature selection using DE genes between control and sPTD
control_label1 = subset(train_meta1,train_meta1$Group == "Control")
control_data1 = train_data1[,which(colnames(train_data1) %in% control_label1$SampleID)]
control_data1 = t(control_data1)
control_label2 = subset(train_meta1,train_meta1$Group == "sPTD")
sptd_data = train_data1[,which(colnames(train_data1) %in% control_label2$SampleID)]
sptd_data = t(sptd_data)

wilox_results = vector()
for (i in 1:ncol(control_data1)) {
  wilox_results[i] = (wilcox.test(control_data1[,i],sptd_data[,i],alternative = "two.sided",paired = FALSE))$p.value
}
sorted_wilcox_results = as.data.frame(wilox_results)
sorted_wilcox_results$Gene = colnames(control_data1)
sorted_wilcox_results$FDR = p.adjust(sorted_wilcox_results$wilox_results,method = "fdr")
sorted_wilcox_results$Bon = p.adjust(sorted_wilcox_results$wilox_results,method = "bonferroni")
sorted_wilcox_results = sorted_wilcox_results[order(sorted_wilcox_results$FDR),]

# Prepare for SVM model(sample by feature matrix)
train_data1 = t(train_data1)
train_data1 = cbind(train_meta1$Group,train_data1)
train_data1 = as.data.frame(train_data1,stringsAsFactors = F)
colnames(train_data1)[1] = "Label"

# Select test data to calculate the accuracy
set.seed(100)
test_df = train_data1[sample(1:nrow(train_data1),size = 30),]
training_df_label = setdiff(rownames(train_data1),rownames(test_df))
training_df = train_data1[which(rownames(train_data1) %in% training_df_label),]

feature_genes = sorted_wilcox_results$Gene[1:100]

subset_train_df = train_data1[,which(colnames(train_data1) %in% feature_genes)]
subset_train_df = cbind(train_data1$Label,subset_train_df)
colnames(subset_train_df)[1] = "Label"
subset_train_df$Label = as.factor(subset_train_df$Label)

# Cross validation

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

# Try linear kernel
linear_results = SVM_model(feature_genes,subset_train_df,model_kernel = "linear",validation_proportion = 0.2,set_seed = 100)
linear_results[[3]]
sigmoid_results = SVM_model(feature_genes,subset_train_df,model_kernel = "sigmoid",validation_proportion = 0.2,set_seed = 100)
sigmoid_results[[3]]
polynomial_results = SVM_model(feature_genes,subset_train_df,model_kernel = "polynomial",validation_proportion = 0.2,set_seed = 100)
polynomial_results[[3]]

# Gaussian kernel is the best
gaussian_results = SVM_model(feature_genes,subset_train_df,model_kernel = "radial",validation_proportion = 0.2,set_seed = 100)
gaussian_results[[3]]

# Select the best bandwidth
gamma = seq(0.0001,0.1,by = 0.002)
gamma_results = vector()
for (i in 1:length(gamma)) {
  gamma_results[i] = SVM_model(feature_genes,subset_train_df,model_kernel = "radial",validation_proportion = 0.2,set_seed = 100,gamma = gamma[i])[[3]]
}
gamma[7]

gamma = seq(0.01,0.02,by = 0.001)
gamma_results = vector()
for (i in 1:length(gamma)) {
  text[i] = SVM_model(feature_genes,subset_train_df,model_kernel = "radial",validation_proportion = 0.2,set_seed = 100,gamma = gamma[i])[[3]]
}

# Try a different feature selection results
feature_genes1 = read.table("~/Desktop/EECS 545/545Project/de_genes.txt",sep = "\t",stringsAsFactors = F,col.names = F)
feature_genes1 = as.vector(feature_genes1$FALSE.)

intersected_genes = intersect(feature_genes1,feature_genes)

gaussian_results1 = SVM_model(feature_genes1,subset_train_df,model_kernel = "radial",validation_proportion = 0.2,set_seed = 100,gamma = 0.01)
