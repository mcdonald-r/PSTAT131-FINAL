library(randomForest)
library(caret)
library(dplyr)
library(tidymodels)

rf_spec <- rand_forest(
  trees = tune(),     
  mtry = tune(),    
  min_n = tune()       
) %>%
  set_mode("classification") %>%
  set_engine("randomForest")

rf_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(rf_spec)

rf_grid <- grid_regular(
  trees(range = c(100, 500)),  
  mtry(range = c(2, 10)),
  min_n(range = c(2, 10)),
  levels = 5
)

tune_results_rf <- tune_grid(
  rf_workflow,
  resamples = bike_folds,
  grid = rf_grid,
  metrics = metric_set(accuracy, roc_auc)
)

best_params_rf <- select_best(tune_results_rf, metric = "roc_auc")
best_params_rf

best_rf_spec <- rand_forest(
  trees = best_params_rf$trees,  
  mtry = best_params_rf$mtry,    
  min_n = best_params_rf$min_n   
) %>%
  set_mode("classification") %>%
  set_engine("randomForest")

best_rf_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(best_rf_spec)

best_rf_fit <- fit(best_rf_workflow, data = bike_train)

save(bike_train, tune_results_rf, rf_grid, best_rf_fit, file = "rf_model_results.rda")