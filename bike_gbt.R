library(tidymodels)  
library(dplyr)

gbt_spec <- boost_tree(
  trees = tune(),        
  tree_depth = tune(),   
  learn_rate = tune(),   
  min_n = tune()         
) %>%
  set_mode("classification") %>%   
  set_engine("xgboost")            

gbt_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(gbt_spec)

gbt_grid <- grid_regular(
  trees(range = c(100, 500)),       
  tree_depth(range = c(3, 10)),     
  learn_rate(range = c(0.01, 0.3)), 
  min_n(range = c(5, 20)),          
  levels = 5                         
)

tune_results_gbt <- tune_grid(
  gbt_workflow,
  resamples = bike_folds,           
  grid = gbt_grid,                 
  metrics = metric_set(accuracy, roc_auc)  
)

best_params_gbt <- select_best(tune_results_gbt, metric = "roc_auc")
best_params_gbt

best_gbt_spec <- boost_tree(
  trees = best_params_gbt$trees,        
  tree_depth = best_params_gbt$tree_depth,   
  learn_rate = best_params_gbt$learn_rate,   
  min_n = best_params_gbt$min_n         
) %>%
  set_mode("classification") %>%   
  set_engine("xgboost")  

best_gbt_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(best_gbt_spec)

best_gbt_fit <- fit(best_gbt_workflow, data = bike_train)

save(bike_train, tune_results_gbt, gbt_grid, best_gbt_fit, file = "gbt_model_results.rda")