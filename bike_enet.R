library(tidymodels)

enet_spec <- multinom_reg(
  penalty = tune(),  
  mixture = tune()  
) %>%
  set_mode("classification") %>%
  set_engine("glmnet")

enet_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(enet_spec)

enet_grid <- grid_regular(
  penalty(range = c(-3, 0)),
  mixture(range = c(0, 1)),  
  levels = 10
)

tune_results_enet <- tune_grid(
  enet_workflow,
  resamples = bike_folds,
  grid = enet_grid,
  metrics = metric_set(accuracy, roc_auc),
  control = control_grid(save_pred = TRUE, parallel_over = "everything")
)

best_params_enet <- select_best(tune_results_enet, metric = "roc_auc")

best_enet_spec <- multinom_reg(
  penalty = best_params_enet$penalty,
  mixture = best_params_enet$mixture
) %>%
  set_mode("classification") %>%
  set_engine("glmnet")

best_enet_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(best_enet_spec)

best_enet_fit <- fit(best_enet_workflow, data = bike_train)

save(bike_train, tune_results_enet, enet_grid, best_enet_fit, file = "enet_model_results.rda")