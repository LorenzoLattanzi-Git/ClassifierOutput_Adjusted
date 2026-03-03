# Load necessary libraries
library(readr)
library(dplyr)
library(caret)
library(RSNNS)
library(nnet)
# Load the dataset
liver_data <- read_csv("E:/Data Science and Business Informatics/Statistics for Data Science/Dataset/bupa-data.csv")

# Ensure Class is a factor
liver_data$Class <- as.factor(liver_data$Class)

# Initialize parameters
num_selections <- 10          # Number of different samples for training and test sets
num_replications <- 10        # Number of times the entire experiment will be repeated within each selection
train_size_per_class <- 50    # Number of samples per class in the training set
test_set_size <- 100          # Total number of samples in the test set
prior_probs <- c(0.2)         # Prior probability for class 1 (can be a vector if testing multiple priors)
CLASS_1 <- "1"                # Labels for the two classes in the dataset
CLASS_2 <- "2"
num_hidden <- 10              # Number of hidden neurons in the neural network
seed_value <- 200             # Seed value for reproducibility

# Function to create a training set
generate_training_set <- function(data, size_per_class, seed) {
  set.seed(seed)
  class1_sample <- data %>% filter(Class == CLASS_1) %>% sample_n(size_per_class)
  class2_sample <- data %>% filter(Class == CLASS_2) %>% sample_n(size_per_class)
  bind_rows(class1_sample, class2_sample)
}

# Function to create a test set from the remaining data
generate_test_set_from_remaining <- function(original_data, training_set, prior_prob, seed) {
  set.seed(seed)
  remaining_data <- anti_join(original_data, training_set, by = names(original_data))
  
  n_class1 <- round(test_set_size * prior_prob)
  n_class2 <- test_set_size - n_class1
  
  test_class1 <- remaining_data %>% filter(Class == CLASS_1) %>% sample_n(min(n_class1, nrow(remaining_data %>% filter(Class == CLASS_1))))
  test_class2 <- remaining_data %>% filter(Class == CLASS_2) %>% sample_n(min(n_class2, nrow(remaining_data %>% filter(Class == CLASS_2))))
  
  test_set <- bind_rows(test_class1, test_class2)
  return(test_set)
}

# EM algorithm to update prior probabilities
em_algorithm <- function(prob_df, pt_omega1, pt_omega2, tol = 1e-6, max_iter = 100) {
  if (!all(c("g1_xk", "g2_xk") %in% names(prob_df))) {
    stop("The input data frame must contain 'g1_xk' and 'g2_xk' columns.")
  }
  
  # Initializes the prior probability for class w1 with the provided starting value.
  p_omega1 <- pt_omega1
  # Initializes the prior probability for class w2
  p_omega2 <- pt_omega2
  # iter: Iteration counter. The loop runs up to max_iter times.
  for (iter in 1:max_iter) {
    # Compute the denominator for calculating responsibilities (used to normalize)
    denominator <- prob_df$g1_xk * p_omega1 + prob_df$g2_xk * p_omega2
    
    # Calculate the responsibilities for class w1 and w2 using the current estimates of the priors
    responsibilities_omega1 <- (prob_df$g1_xk * p_omega1) / pmax(denominator, .Machine$double.eps)
    responsibilities_omega2 <- 1 - responsibilities_omega1
    
    # Update the new prior probability for class w1 and w2
    new_p_omega1 <- mean(responsibilities_omega1)
    new_p_omega2 <- mean(responsibilities_omega2)
    
    #abs(new_p_omega1 - p_omega1) < tol: Checks if the change in the prior probability of w1 is below the tolerance level tol.
    if (abs(new_p_omega1 - p_omega1) < tol && abs(new_p_omega2 - p_omega2) < tol) {
      cat("EM converged in", iter, "iterations\n")
      break
    }
    
    #Updates the prior probabilities for the next iteration.
    p_omega1 <- new_p_omega1
    p_omega2 <- new_p_omega2
  }
  
  # Check if the algorithm reached the maximum number of iterations without converging
  if (iter == max_iter) {
    cat("EM algorithm did not converge within the maximum number of iterations\n")
  }
  
  # Returns the final estimates of the prior probabilities for classes w1 and w2
  return(list(p_omega1 = p_omega1, p_omega2 = p_omega2))
}

# Function to estimate prior probabilities using the confusion matrix method
confusion_matrix_method <- function(train_cm, test_cm) {
  # Extract confusion matrix components for both test and training sets
  a_test <- test_cm[1, 1] #TP
  b_test <- test_cm[1, 2] #FN
  c_test <- test_cm[2, 1] #FP
  d_test <- test_cm[2, 2] #TN
  
  a_train <- train_cm[1, 1] #TP
  b_train <- train_cm[1, 2] #FN
  c_train <- train_cm[2, 1] #FP
  d_train <- train_cm[2, 2] #TN
  
  # Calculate the elements for the system of linear equations
  left_side <- matrix(c(
    a_train / (a_train + c_train), c_train / (a_train + c_train),
    b_train / (b_train + d_train), d_train / (b_train + d_train)
  ), nrow = 2, byrow = TRUE)
  
  right_side <- c(
    (a_test + b_test) / (a_test + b_test + c_test + d_test),
    (c_test + d_test) / (a_test + b_test + c_test + d_test)
  )
  
  # Solve the system of linear equations to estimate the prior probabilities
  priors <- solve(left_side, right_side)
  
  # Return estimated prior probabilities for class 1 and class 2
  return(list(p_w1 = priors[1], p_w2 = priors[2]))
}

# Function to initialize prior probabilities
initialize_prior_probabilities <- function(class_data, class1, class2) {
  # Calculate the prior probability for class 1 and class 2
  p_class1 <- mean(class_data == class1)
  p_class2 <- mean(class_data == class2)
  c(Class1 = p_class1, Class2 = p_class2)
}

# Initialize lists to store results
priors_list_em <- list()  # For EM method
priors_list_cm <- list()  # For Confusion Matrix method

# Initialize a list to store accuracies for calculating average accuracy
accuracy_list_before_adjustment <- list()
accuracy_list_after_em <- list()
accuracy_list_after_cm <- list()
accuracy_list_after_known_priori <- list()  

# Main loop for selections of datasets
for (selection in 1:num_selections) {
  cat("Selection:", selection, "\n")
  
  # Create the training and test sets for this selection
  training_set <- generate_training_set(liver_data, train_size_per_class, seed_value + selection)
  test_set <- generate_test_set_from_remaining(liver_data, training_set, prior_probs[1], seed_value + selection)
  
  # Prepare inputs for testing
  test_inputs <- as.matrix(test_set %>% select(-Class))
  test_actual <- test_set$Class
  
  # Loop over replications for each selection
  for (rep in 1:num_replications) {
    cat("  Replication:", rep, "\n")
    
    # Prepare input and target data for training
    train_inputs <- as.matrix(training_set %>% select(-Class))
    train_targets <- class.ind(as.factor(training_set$Class))
    
    # Train the model using RSNNS with Scaled Conjugate Gradient algorithm
    set.seed(seed_value + selection * 100 + rep)
    mlp_model <- mlp(x = train_inputs, y = train_targets,
                     size = c(num_hidden),  # Number of hidden units, can be a vector for multiple layers
                     learnFunc = "SCG",  # Use "SCG" (Scaled Conjugate Gradient)
                     learnFuncParams = c(0.01, 0.001),  # Parameters: learning rate, learning rate decay
                     maxit = 100)  # Maximum number of iterations
    
    # Predict class probabilities for training set
    train_pred_probs <- predict(mlp_model, train_inputs)
    train_predictions <- apply(train_pred_probs, 1, which.max)
    train_predictions <- factor(train_predictions, levels = 1:2, labels = c(CLASS_1, CLASS_2))
    train_actual <- training_set$Class
    
    # Calculate the confusion matrix for the training set
    train_cm <- confusionMatrix(factor(train_predictions, levels = levels(train_actual)), train_actual)
    
    for (prior_prob in prior_probs) {
      cat("    Prior Probability for Class 1:", prior_prob, "\n")
      
      # Generate the test set with a specific prior probability
      set.seed(seed_value + selection * 100 + rep)
      test_set <- generate_test_set_from_remaining(liver_data, training_set, prior_prob, seed_value + selection * 100 + rep)
      test_inputs <- as.matrix(test_set %>% select(-Class))
      test_actual <- test_set$Class
      
      # Predict class probabilities for test set
      test_pred_probs <- predict(mlp_model, test_inputs)
      test_predictions <- apply(test_pred_probs, 1, which.max)
      test_predictions <- factor(test_predictions, levels = 1:2, labels = c(CLASS_1, CLASS_2))
      
      # Calculate accuracy before adjustment
      test_accuracy_before_adjustment <- mean(test_predictions == test_actual)
      
      # Estimation by the EM method
      prob_df <- data.frame(g1_xk = test_pred_probs[, 1], g2_xk = test_pred_probs[, 2])
      prior_probs_init <- initialize_prior_probabilities(as.factor(training_set$Class), CLASS_1, CLASS_2)
      pt_omega1 <- prior_probs_init["Class1"]
      pt_omega2 <- prior_probs_init["Class2"]
      em_results <- em_algorithm(prob_df, pt_omega1, pt_omega2)
      
      # Extract the updated prior probabilities from EM results
      p_omega1 <- em_results$p_omega1
      p_omega2 <- em_results$p_omega2
      
      # Compute posterior probabilities using EM method
      posteriors_em <- data.frame(
        P_omega1_given_X_em = (test_pred_probs[, 1] * p_omega1) / (test_pred_probs[, 1] * p_omega1 + test_pred_probs[, 2] * p_omega2),
        P_omega2_given_X_em = (test_pred_probs[, 2] * p_omega2) / (test_pred_probs[, 1] * p_omega1 + test_pred_probs[, 2] * p_omega2)
      )
      
      # Predict classes using updated posteriors from EM method
      test_predictions_after_em <- ifelse(posteriors_em$P_omega1_given_X_em >= posteriors_em$P_omega2_given_X_em, CLASS_1, CLASS_2)
      test_predictions_after_em <- factor(test_predictions_after_em, levels = c(CLASS_1, CLASS_2))
      
      # Calculate accuracy after EM adjustment
      test_accuracy_after_em <- mean(test_predictions_after_em == test_actual)
      
      
      # Estimation by the confusion matrix method
      test_cm <- confusionMatrix(factor(test_predictions, levels = levels(test_actual)), test_actual)
      
      # Call the function to estimate priors
      priors <- confusion_matrix_method(train_cm, test_cm)
      
      # Access estimated priors
      p_w1 <- priors$p_w1  # Estimated prior for class 1
      p_w2 <- priors$p_w2  # Estimated prior for class 2
      
      # Compute the posterior probabilities using Bayes' theorem by estimated priors by CM
      posteriors_cm <- data.frame(
        P_w1_given_X_cm = (test_pred_probs[, 1] * p_w1) / (test_pred_probs[, 1] * p_w1 + test_pred_probs[, 2] * p_w2),
        P_w2_given_X_cm = (test_pred_probs[, 2] * p_w2) / (test_pred_probs[, 1] * p_w1 + test_pred_probs[, 2] * p_w2)
      )
      
      # Predict classes using updated posteriors from Confusion Matrix method
      test_predictions_after_cm <- ifelse(posteriors_cm$P_w1_given_X_cm >= posteriors_cm$P_w2_given_X_cm, CLASS_1, CLASS_2)
      test_predictions_after_cm <- factor(test_predictions_after_cm, levels = c(CLASS_1, CLASS_2))
      
      # Calculate accuracy after CM adjustment
      test_accuracy_after_cm <- mean(test_predictions_after_cm == test_actual)
      
      # Compute the posterior probabilities using Bayes' theorem by True Priors
      posteriors_known_priori <- data.frame(
        P_omega1_given_X_known_priori = (test_pred_probs[, 1] * prior_prob) / (test_pred_probs[, 1] * prior_prob + test_pred_probs[, 2] * (1 - prior_prob)),
        P_omega2_given_X_known_priori = (test_pred_probs[, 2] * (1 - prior_prob)) / (test_pred_probs[, 1] * prior_prob + test_pred_probs[, 2] * (1 - prior_prob))
      )
      
      # Predict classes using updated posteriors from True Priors 
      test_predictions_after_known_priori <- ifelse(posteriors_known_priori$P_omega1_given_X_known_priori >= posteriors_known_priori$P_omega2_given_X_known_priori, CLASS_1, CLASS_2)
      test_predictions_after_known_priori <- factor(test_predictions_after_known_priori, levels = c(CLASS_1, CLASS_2))
      
      # Calculate accuracy after using True Priors adjustment
      test_accuracy_after_known_priori <- mean(test_predictions_after_known_priori == test_actual)
      
      # Store the accuracies in the list
      accuracy_list_before_adjustment <- c(accuracy_list_before_adjustment, test_accuracy_before_adjustment)
      accuracy_list_after_em <- c(accuracy_list_after_em, test_accuracy_after_em)
      accuracy_list_after_cm <- c(accuracy_list_after_cm, test_accuracy_after_cm)
      accuracy_list_after_known_priori <- c(accuracy_list_after_known_priori, test_accuracy_after_known_priori)
      priors_list_em[[paste("Selection", selection, "Rep", rep, sep = "_")]] <- p_omega1
      priors_list_cm[[paste("Selection", selection, "Rep", rep, sep = "_")]] <- p_w1
    }
  }
}

# Convert the list of p_omega1 (estimate by em) values to a numeric vector
p_omega1_values_vector <- unlist(priors_list_em)
# Compute the average value of p_omega1 (estimate by em)
average_prior_em <- mean(p_omega1_values_vector)

# Convert the list of p_w1 (estimate by confusion matrix) values to a numeric vector
p_w1_values_vector <- unlist(priors_list_cm)
# Compute the average value of p_w1 (estimate by confusion matrix)
average_prior_cm <- mean(p_w1_values_vector)

# Calculate average accuracies
avg_accuracy_before <- mean(unlist(accuracy_list_before_adjustment))
avg_accuracy_after_em <- mean(unlist(accuracy_list_after_em))
avg_accuracy_after_cm <- mean(unlist(accuracy_list_after_cm))
avg_accuracy_after_known_priori <- mean(unlist(accuracy_list_after_known_priori))

# Print the results
cat("Average Priors Estimated by EM:", average_prior_em, "\n")
cat("Average Priors Estimated by Confusion Matrix:", average_prior_cm, "\n")
cat("Average Accuracy No Adjustment:", avg_accuracy_before, "\n")
cat("Average Accuracy After Adjustment using EM:", avg_accuracy_after_em, "\n")
cat("Average Accuracy After Adjustment using Confusion Matrix:", avg_accuracy_after_cm, "\n")
cat("Average Accuracy After Adjustment using True Priors:", avg_accuracy_after_known_priori, "\n")




