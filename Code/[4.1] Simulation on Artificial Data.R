# Load necessary libraries
library(readr)
library(dplyr)
library(nnet)
library(caret)

# Load the dataset
ringnorm_data <- read_csv("E:/Data Science and Business Informatics/Statistics for Data Science/Dataset/Ringnorm.xls")

# Check if the data is correctly loaded as a data frame
if (!is.data.frame(ringnorm_data)) {
  stop("The loaded data is not a data frame. Please check the data source.")
}

# Ensure Class is a factor
ringnorm_data$Class <- as.factor(ringnorm_data$Class)

# Initialize parameters
num_replications<- 10  # Number of times the entire experiment will be repeated to ensure consistency
train_size_per_class <- 500          # Number of samples per class in the training set
test_set_size <- 1000                # Total number of samples in the test set
prior_probs <- seq(0.10, 0.90, by = 0.10)  # Sequence of prior probabilities to test, ranging from 0.10 to 0.90
CLASS_1 <- "b'1'"                    # Labels for the two classes in the dataset
CLASS_2 <- "b'2'"
num_hidden <- 10                     # Number of hidden neurons in the neural network
seed_value <- 123

# Function to create a training set
generate_training_set <- function(data, size_per_class, seed) {
  set.seed(seed) # Set the seed for reproducibility to ensure the same samples are drawn every time
  class1_sample <- data %>% filter(Class == CLASS_1) %>% sample_n(size_per_class) #Filter the dataset to get all samples for CLASS_1
  class2_sample <- data %>% filter(Class == CLASS_2) %>% sample_n(size_per_class) #Filter the dataset to get all samples for CLASS_2
  bind_rows(class1_sample, class2_sample) #Combine the samples into a single training set
}

# Test Set Creation Function. This function generates a test set from the data remaining (data remaining = the original dataset - training set)	
generate_test_set_from_remaining <- function(original_data, training_set, prior_prob, seed) { # original_data is the Ringnorm dataset. Seed is the value for random sampling to ensure the reproducibility
  set.seed(seed) 
  remaining_data <- anti_join(original_data, training_set, by = names(original_data))  # Create the remaining data by excluding the training set
  n_class1 <- round(test_set_size * prior_prob)
  n_class2 <- test_set_size - n_class1
  
  test_class1 <- remaining_data %>% filter(Class == CLASS_1) %>% sample_n(min(n_class1, nrow(remaining_data %>% filter(Class == CLASS_1)))) #calculate the number of samples need from CLASS_1
  test_class2 <- remaining_data %>% filter(Class == CLASS_2) %>% sample_n(min(n_class2, nrow(remaining_data %>% filter(Class == CLASS_2)))) #calculate the number of samples need from CLASS_1
  
  test_set <- bind_rows(test_class1, test_class2) #filter the Class to return only CLASS_1(CLASS_2). the sample_n(min(n_class1, nrow(remaining_data %>% filter(Class == CLASS_1) means sampling the necessary number of records from CLASS_1(CLASS_2), in order to ensure it doesn't exceed the available records from training set
  return(test_set)
}

# Initialize lists to store results
results_list <- list()
priors_list_cm <- list()    #for confusion matrix
priors_list_em <- list()    #for EM method

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
    
    # Update the prior probability for class w1 and w2
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
  a_test <- test_cm$table[1, 1] #TP
  b_test <- test_cm$table[1, 2] #FN
  c_test <- test_cm$table[2, 1] #FP
  d_test <- test_cm$table[2, 2] #TN
  
  a_train <- train_cm$table[1, 1] #TP
  b_train <- train_cm$table[1, 2] #FN
  c_train <- train_cm$table[2, 1] #FP
  d_train <- train_cm$table[2, 2] #TN
  
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

#class_data: A vector containing the class labels of the dataset.class1, class2: The two different class labels.
initialize_prior_probabilities <- function(class_data, class1, class2) {
  #p_class1: The proportion of the data that belongs to class1, calculated as the mean of the Boolean vector where class_data equals class1.
  p_class1 <- mean(class_data == class1)
  p_class2 <- mean(class_data == class2)
  #c(Class1 = p_class1, Class2 = p_class2): Returns a named vector with the prior probabilities of class1 and class2.
  c(Class1 = p_class1, Class2 = p_class2)
}

# Improved function for calculating the log-likelihood ratio
log_likelihood_ratio <- function(prob_df, prior_probs_init, prior_probs_updated) {
  # Small value to avoid log(0)
  eps <- 1e-10
  # Calculate the likelihood of xk given omega_i for the initial prior probabilities H0
  likelihood_init <- prob_df$g1_xk * prior_probs_init[1] + prob_df$g2_xk * prior_probs_init[2]
  # Ensure that likelihood values are not zero to prevent log(0)
  likelihood_init <- pmax(likelihood_init, eps)  
  # Calculate the likelihood of xk given omega_i for the updated prior probabilities H1
  likelihood_updated <- prob_df$g1_xk * prior_probs_updated[1] + prob_df$g2_xk * prior_probs_updated[2]
  # Ensure that likelihood values are not zero to prevent log(0)
  likelihood_updated <- pmax(likelihood_updated, eps)
  # Calculate log-likelihood for initial and updated priors
  log_likelihood_init <- sum(log(likelihood_init))
  log_likelihood_updated <- sum(log(likelihood_updated))
  # Calculate the log-likelihood ratio
  llr <- 2 * (log_likelihood_updated - log_likelihood_init)
  return(llr)
}

# Likelihood Ratio Test (LRT) function
#The LRT is used to determine if the updated model (with prior_probs_updated) provides a 
#Significantly better fit to the data than the initial model (prior_probs_init). 
likelihood_ratio_test <- function(prob_df, prior_probs_init, prior_probs_updated) {
  # Calculate the log-likelihood ratio
  llr <- log_likelihood_ratio(prob_df, prior_probs_init, prior_probs_updated)
  lrt_statistic <- llr
  # Degrees of freedom: n - 1 (where n = number of classes, here 2)
  df <- 1
  # Calculate the p-value from the chi-square distribution
  p_value <- pchisq(lrt_statistic, df = df, lower.tail = FALSE)
  return(list(statistic = lrt_statistic, p_value = p_value))
}

# Main loop for replications
for (rep in 1:num_replications) {
  cat("Replication:", rep, "\n")
  
  # Create the training set
  training_set <- generate_training_set(ringnorm_data, train_size_per_class, seed_value + rep)
  
  # Prepare input and target data for training
  train_inputs <- as.matrix(training_set %>% select(-Class))
  train_targets <- class.ind(as.factor(training_set$Class))
  
  # Train the model using nnet (MLP) with a fixed seed
  set.seed(seed_value + rep)
  mlp_model <- nnet(x = train_inputs, y = train_targets, size = num_hidden, maxit = 1200, linout = FALSE)
  
  # Predict class probabilities for training set
  train_pred_probs <- predict(mlp_model, train_inputs, type = "raw")
  train_predictions <- colnames(train_pred_probs)[max.col(train_pred_probs, ties.method = "first")]
  train_actual <- training_set$Class
  
  # Calculate confusion matrix for training set
  train_cm <- confusionMatrix(factor(train_predictions, levels = levels(train_actual)), train_actual)
  
  for (prior_prob in prior_probs) {
    cat("  Prior Probability for Class b'1':", prior_prob, "\n")
    
    # Generate the test set with a specific prior probability
    test_set <- generate_test_set_from_remaining(ringnorm_data, training_set, prior_prob, seed_value + rep)
    
    # Prepare test inputs
    test_inputs <- as.matrix(test_set %>% select(-Class))
    test_actual <- test_set$Class
    
    # Predict class probabilities for test set
    test_pred_probs <- predict(mlp_model, test_inputs, type = "raw")
    test_predictions <- colnames(test_pred_probs)[max.col(test_pred_probs, ties.method = "first")]
    
    # Calculate the accuracy before adjustment
    test_accuracy_before_adjustment <- mean(test_predictions == test_actual)
    
    # Calculate the posterior probabilities using Bayes' theorem with known prior, this function applied formular (2.4) in the paper
    posteriors_known_priori <- data.frame(
      P_w1_given_X_known_priori = (test_pred_probs[, 1] * prior_prob) / 
        (test_pred_probs[, 1] * prior_prob + test_pred_probs[, 2] * (1 - prior_prob)),
      P_w2_given_X_known_priori = (test_pred_probs[, 2] * (1 - prior_prob)) / 
        (test_pred_probs[, 1] * prior_prob + test_pred_probs[, 2] * (1 - prior_prob))
    )
    
    # Predict classes based on adjusted probabilities by known priori
    test_predictions_adjusted_known_priori <- ifelse(posteriors_known_priori$P_w1_given_X_known_priori > 0.5, CLASS_1, CLASS_2)
    test_accuracy_after_adjustment_known_priori <- mean(test_predictions_adjusted_known_priori == test_actual)
    
    # Store results by known priori
    result_key <- paste("Prior", prior_prob, "Rep", rep, sep = "_") #use a key to store and retrieve results in/from the results_list
    results_list[[result_key]] <- list( #it is a list where each entry corresponds to the results of each experiment
      accuracy_test_before_adjustment_known_priori = test_accuracy_before_adjustment,
      accuracy_test_after_adjustment_known_priori = test_accuracy_after_adjustment_known_priori
    )
    
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
    
    # Re-calculate test predictions using adjusted posterior probabilities
    test_predictions_adjusted_cm <- ifelse(posteriors_cm$P_w1_given_X_cm > 0.5, CLASS_1, CLASS_2)
    test_accuracy_after_adjustment_by_cm <- mean(test_predictions_adjusted_cm == test_actual)
    
    # Store results by using priori probability estimated by confusion matrix method
    results_list[[result_key]]$accuracy_test_after_adjustment_by_cm <- test_accuracy_after_adjustment_by_cm
    priors_list_cm[[result_key]] <- c(p_w1, p_w2) #store the priori probability estimated by confusion matrix method
    
    # Estimation by the EM method
    prob_df <- data.frame(g1_xk = test_pred_probs[, 1], prob_df <- data.frame(
        g1_xk = test_pred_probs[, 1], # Predicted probabilities for class w1
        g2_xk = test_pred_probs[, 2]  # Predicted probabilities for class w2
      ))
    prior_probs_init <- initialize_prior_probabilities(as.factor(training_set$Class), CLASS_1, CLASS_2)
    pt_omega1 <- prior_probs_init["Class1"]
    pt_omega2 <- prior_probs_init["Class2"]
    em_results <- em_algorithm(prob_df, pt_omega1, pt_omega2)
    lrt_result <- likelihood_ratio_test(prob_df, c(pt_omega1, pt_omega2), c(em_results$p_omega1, em_results$p_omega2))
    
    # Extract the updated prior probabilities from EM results
    p_omega1 <- em_results$p_omega1
    p_omega2 <- em_results$p_omega2
    
    # Compute posterior probabilities using EM method
    posteriors_em <- data.frame(
      P_omega1_given_X_em = (test_pred_probs[, 1] * p_omega1) / (test_pred_probs[, 1] * p_omega1 + test_pred_probs[, 2] * p_omega2),
      P_omega2_given_X_em = (test_pred_probs[, 2] * p_omega2) / (test_pred_probs[, 1] * p_omega1 + test_pred_probs[, 2] * p_omega2)
    )
    
    # Use adjusted probabilities to class predictions
    adjusted_predictions <- ifelse(posteriors_em$P_omega1_given_X_em > posteriors_em$P_omega2_given_X_em, CLASS_1, CLASS_2)
    
    # Test_predictions_adjusted_em: converts adjusted_predictions to a factor, ensuring that the levels (possible class labels) match those in the test set.
    test_predictions_adjusted_em <- factor(adjusted_predictions, levels = levels(test_set$Class))
    test_accuracy_after_adjustment_by_em <- mean(test_predictions_adjusted_em == test_set$Class)
    
    # Store results by using priori probability estimated by EM method
    results_list[[result_key]]$accuracy_test_after_adjustment_by_em <- test_accuracy_after_adjustment_by_em #storethe accuracy of the classification model after adjustment by priori probability estimated from EM
    results_list[[result_key]]$lrt_statistics <- lrt_result$statistic #store the Likelihood Ratio Test for the specific combination of prior probability and replication
    results_list[[result_key]]$lrt_p_values <- lrt_result$p_value #store the p-value from the Likelihood Ratio Test
    priors_list_em[[result_key]] <- c(em_results$p_omega1, em_results$p_omega2) #store the estimated prior probabilities by EM of class 1 and 2 
  }
}

# Calculate average accuracy and estimated priors across replications, accuracy means classification rate
# Average accuracy of classification no adjusment
average_accuracy_test_before <- sapply(prior_probs, function(prior_prob) {
  accuracy_values <- sapply(1:num_replications, function(rep) {
    result_key <- paste("Prior", prior_prob, "Rep", rep, sep = "_")
    return(results_list[[result_key]]$accuracy_test_before_adjustment)
  })
  return(mean(accuracy_values))
})

# Average accuracy of classification after adjusting by EM
average_accuracy_test_after_EM <- sapply(prior_probs, function(prior_prob) {
  accuracy_values <- sapply(1:num_replications, function(rep) {
    result_key <- paste("Prior", prior_prob, "Rep", rep, sep = "_")
    return(results_list[[result_key]]$accuracy_test_after_adjustment_by_em)
  })
  return(mean(accuracy_values))
})

# Average accuracy of classification after adjusting by confusion matrix method
average_accuracy_test_after_cm <- sapply(prior_probs, function(prior_prob) {
  accuracy_values <- sapply(1:num_replications, function(rep) {
    result_key <- paste("Prior", prior_prob, "Rep", rep, sep = "_")
    return(results_list[[result_key]]$accuracy_test_after_adjustment_by_cm)
  })
  return(mean(accuracy_values))
})

# Average accuracy of classification after adjusting by True Priors
average_accuracy_test_after_known_priori <- sapply(prior_probs, function(prior_prob) {
  accuracy_values <- sapply(1:num_replications, function(rep) {
    result_key <- paste("Prior", prior_prob, "Rep", rep, sep = "_")
    return(results_list[[result_key]]$accuracy_test_after_adjustment_known_priori)
  })
  return(mean(accuracy_values))
})

# Average priori probability of class 1 (p_omega1) that estimated by confusion matrix method
avg_omega1_cm <- sapply(prior_probs, function(prior_prob) {
  prior_values <- sapply(1:num_replications, function(rep) {
    result_key <- paste("Prior", prior_prob, "Rep", rep, sep = "_")
    return(priors_list_cm[[result_key]][1])
  })
  return(mean(prior_values))
})

# Average priori probability of class 1 (p_omega1) that estimated by EM method
avg_omega1_em <- sapply(prior_probs, function(prior_prob) {
  omega1_values <- sapply(1:num_replications, function(rep) {
    result_key <- paste("Prior", prior_prob, "Rep", rep, sep = "_")
    return(priors_list_em[[result_key]][1])
  })
  return(mean(omega1_values))
})

# Function to calculate the number of significant tests across all replications for each prior
calculate_significant_tests_counts <- function(results_list, prior_probs, num_replications, significance_level = 0.01) {
  # Use sapply to iterate over each prior probability
  significant_counts <- sapply(prior_probs, function(prior_prob) {
    count <- 0 # Initialize a counter for significant tests for this prior
    
    for (rep in 1:num_replications) {
      # Construct the key to access the results for the specific prior and replication
      result_key <- paste("Prior", prior_prob, "Rep", rep, sep = "_")
      # Extract the likelihood ratio test (LRT) p-value from the results
      lrt_p_value <- results_list[[result_key]]$lrt_p_values
      
      # Check if the LRT p-value is less than the significance threshold
      if (lrt_p_value < significance_level) {
        count <- count + 1 # Increment the count if the test is significant
      }
    }
    return(count)
  })
  
  # Assign names to the counts based on the corresponding prior probabilities
  names(significant_counts) <- paste0("Prior_", prior_probs)
  return(significant_counts)
}

# Calculate the significant tests counts for all prior probabilities
significant_tests_counts <- calculate_significant_tests_counts(results_list, prior_probs, num_replications, significance_level = 0.01)

# Create the estimation table
estimation_table <- data.frame(
  True_Priors = prior_probs * 100,  
  Estimated_Prior_EM = avg_omega1_em * 100,
  Estimated_Prior_CM = avg_omega1_cm * 100,  
  Significant_Tests = significant_tests_counts  # Add significant tests counts
)
row.names(estimation_table) <- 1:nrow(estimation_table)

# Print the updated estimation table
print(estimation_table)

# Prepare classification table 
classification_table <- data.frame(
  True_Priors = prior_probs * 100,  
  No_Adjustment = average_accuracy_test_before * 100,  
  After_Adjustment_EM = average_accuracy_test_after_EM * 100,
  After_Adjustment_Confusion_Matrix = average_accuracy_test_after_cm * 100,  
  After_Adjustment_True_Priors = average_accuracy_test_after_known_priori * 100  
)

# Print the classification table
print(classification_table)

