library(caret)
library(dplyr)
library(tidymodels)
library(kernlab)  

svm_spec <- svm_rbf(
  cost = tune(),        
  rbf_sigma = tune()  
) %>%
  set_mode("classification") %>%
  set_engine("kernlab")  

svm_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(svm_spec)

svm_grid <- grid_regular(
  cost(range = c(0.001, 10)),     
  rbf_sigma(range = c(0.001, 1)),  
  levels = 5                       
)

tune_results_svm <- tune_grid(
  svm_workflow,
  resamples = bike_folds,
  grid = svm_grid,
  metrics = metric_set(accuracy, roc_auc)
)

best_params_svm <- select_best(tune_results_svm, metric = "roc_auc")
best_params_svm

best_svm_spec <- svm_rbf(
  cost = best_params_svm$cost,  
  rbf_sigma = best_params_svm$rbf_sigma
) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

best_svm_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(best_svm_spec)

best_svm_fit <- fit(best_svm_workflow, data = bike_train)

save(bike_train, tune_results_svm, svm_grid, best_svm_fit, file = "svm_model_results.rda")