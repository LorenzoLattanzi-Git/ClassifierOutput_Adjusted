# Load necessary libraries
library(readr)
library(dplyr)
library(tidyr)
library(nnet)
library(caret)
library(e1071)
library(ggplot2)

# Load the dataset
ringnorm_data <- read_csv("E:/Data Science and Business Informatics/Statistics for Data Science/Dataset/Ringnorm.xls")

# Ensure Class is a factor
ringnorm_data$Class <- as.factor(ringnorm_data$Class)

# Define constants
CLASS_1 <- "b'1'"
CLASS_2 <- "b'2'"
num_hidden <- 10
seed_value <- 123 # The seed for random number generation to ensure reproducibility

# Initialize parameters
train_sizes <- list(c(500, 500), c(250, 250), c(100, 100), c(50, 50))
test_sizes <- list(c(200, 800), c(100, 400), c(40, 160), c(20, 80))
prior_probs <- c(0.20, 0.80)  # True priors for test set
num_replications <- 10

# Function to create a training set
generate_training_set <- function(data, size_per_class, seed) {
  set.seed(seed) # Set the seed for reproducibility to ensure the same samples are drawn every time
  class1_sample <- data %>% filter(Class == CLASS_1) %>% sample_n(size_per_class)
  class2_sample <- data %>% filter(Class == CLASS_2) %>% sample_n(size_per_class)
  bind_rows(class1_sample, class2_sample)
}

# Function to create a test set from remaining data
generate_test_set_from_remaining <- function(original_data, training_set, test_size, prior_prob, seed) {
  set.seed(seed)
  remaining_data <- anti_join(original_data, training_set, by = names(original_data))
  n_class1 <- round(test_size * prior_prob)
  n_class2 <- test_size - n_class1
  
  test_class1 <- remaining_data %>% filter(Class == CLASS_1) %>% sample_n(min(n_class1, nrow(remaining_data %>% filter(Class == CLASS_1))))
  test_class2 <- remaining_data %>% filter(Class == CLASS_2) %>% sample_n(min(n_class2, nrow(remaining_data %>% filter(Class == CLASS_2))))
  
  test_set <- bind_rows(test_class1, test_class2)
  return(list(test_set = test_set, n_class1 = n_class1, n_class2 = n_class2))
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

# Function to calculate Mean Absolute Deviation (MAD)
calculate_mad <- function(bayes_probs, mlp_probs) {
  # Calculate the Mean Absolute Deviation (MAD)
  mad <- mean(abs(bayes_probs - mlp_probs))
  return(mad)
}

# Initialize lists to store results
priors_list_em <- list()  # For EM method
priors_list_cm <- list()  # For Confusion Matrix method
results_list <- list()
accuracy_list <- list()

# Function to initialize prior probabilities
initialize_prior_probabilities <- function(class_data, class1, class2) {
  # Calculate the prior probability for class 1 and class 2
  p_class1 <- mean(class_data == class1)
  p_class2 <- mean(class_data == class2)
  c(Class1 = p_class1, Class2 = p_class2)
}

# Adjusted Code Segment
for (train_size in train_sizes) {
  for (test_size in test_sizes) {
    for (rep in 1:num_replications) {
      cat("Train Size:", train_size[1], train_size[2], 
          "Test Size:", test_size[1], test_size[2], 
          "Replication:", rep, "\n")
      
      # Create the training set
      training_set <- generate_training_set(ringnorm_data, train_size[1], seed_value + rep)
      
      # Prepare input and target data for training
      train_inputs <- as.matrix(training_set %>% select(-Class))
      train_targets <- class.ind(as.factor(training_set$Class))
      
      # Train the model MLP
      set.seed(seed_value + rep)
      mlp_model <- nnet(x = train_inputs, y = train_targets, size = num_hidden, maxit = 1200, linout = FALSE)
      
      # Predict class probabilities for training set
      train_pred_probs <- predict(mlp_model, train_inputs)
      train_predictions <- colnames(train_pred_probs)[max.col(train_pred_probs, ties.method = "first")]
      train_actual <- training_set$Class
      train_cm <- confusionMatrix(factor(train_predictions, levels = levels(train_actual)), train_actual)
      
      # Train the Naive Bayes model
      set.seed(seed_value + rep)
      nb_model <- naiveBayes(
        Class ~ .,                     # Formula
        data = training_set,            # Training data
        laplace = 1,                    # Apply Laplace smoothing with a value of 1
        prior = c(0.5, 0.5),            # Set equal priors for the two classes (if you have 2 classes)
        na.action = na.omit             # Remove rows with NA values
      )
      
      # Generate test set
      test_set_info <- generate_test_set_from_remaining(ringnorm_data, training_set, sum(test_size), prior_probs[1], seed_value + rep)
      test_set <- test_set_info$test_set
      test_inputs <- as.matrix(test_set %>% select(-Class))
      test_actual <- test_set$Class
      
      # Predict class probabilities for test set
      test_pred_probs <- predict(mlp_model, test_inputs, type = "raw")
      test_predictions <- colnames(test_pred_probs)[max.col(test_pred_probs, ties.method = "first")]
      result_key <- paste("Prior", prior_probs[1], "Rep", rep, sep = "_")
      
      # Make predictions with Naive Bayes
      nb_predictions <- predict(nb_model, test_set, type = "raw")
      prob_df_nb <- data.frame(
        b1_xk = nb_predictions[, 1]
      )
      
      # Calculate MAD
      mad <- calculate_mad(nb_predictions[, 1], test_pred_probs[, 1])
      
      # calculate the accuracy before adjustment
      test_accuracy_before_adjustment <- mean(test_predictions == test_actual)
      
      # calculate the posterior probabilities using Bayes' theorem with known prior, this function applied formular (2.4) in the paper
      posteriors_known_priori <- data.frame(
        P_w1_given_X_known_priori = (test_pred_probs[, 1] * 0.2) / 
          (test_pred_probs[, 1] * 0.2 + test_pred_probs[, 2] * 0.8),
        P_w2_given_X_known_priori = (test_pred_probs[, 2] * 0.8) / 
          (test_pred_probs[, 1] * 0.2 + test_pred_probs[, 2] * 0.8)
      )
      
      # predict classes based on adjusted probabilities by known priori
      test_predictions_adjusted_known_priori <- ifelse(posteriors_known_priori$P_w1_given_X_known_priori > 0.5, CLASS_1, CLASS_2)
      test_accuracy_after_adjustment_known_priori <- mean(test_predictions_adjusted_known_priori == test_actual)
      
      # Run EM algorithm
      # Prepare data for EM algorithm
      prob_df <- data.frame(
        g1_xk = test_pred_probs[, 1], # Predicted probabilities for class w1
        g2_xk = test_pred_probs[, 2]  # Predicted probabilities for class w2
      )
      prior_probs_init <- initialize_prior_probabilities(as.factor(training_set$Class), CLASS_1, CLASS_2)
      pt_omega1 <- prior_probs_init["Class1"]
      pt_omega2 <- prior_probs_init["Class2"]
      em_results <- em_algorithm(prob_df, pt_omega1, pt_omega2)
      
      # Extract the updated prior probabilities from EM results
      p_omega1 <- em_results$p_omega1
      p_omega2 <- em_results$p_omega2
      priors_list_em[[result_key]] <- c(p_omega1, p_omega2)
      
      # Extract the prior probability for class w1 (p_omega1) from each element in 'priors_list_em'
      prior_1_em <- sapply(priors_list_em, function(x) x[1])
      
      #compute the posterior probabilities estimated by EM method
      posteriors_em <- data.frame(
        P_omega1_given_X_em = (test_pred_probs[, 1] * p_omega1) / 
          (test_pred_probs[, 1] * p_omega1 + test_pred_probs[, 2] * (1 - p_omega1)),
        P_omega2_given_X_em = (test_pred_probs[, 2] * (1 - p_omega1)) / 
          (test_pred_probs[, 1] * p_omega1 + test_pred_probs[, 2] * (1 - p_omega1))
      )
      
      # Use adjusted probabilities to class predictions
      adjusted_predictions <- ifelse(posteriors_em$P_omega1_given_X_em > posteriors_em$P_omega2_given_X_em, CLASS_1, CLASS_2)
      
      # Test_predictions_adjusted_em: converts adjusted_predictions to a factor, ensuring that the levels (possible class labels) match those in the test set.
      test_predictions_adjusted_em <- factor(adjusted_predictions, levels = levels(test_set$Class))
      test_accuracy_after_adjustment_by_em <- mean(test_predictions_adjusted_em == test_set$Class)
      
      # Estimation by the confusion matrix method
      test_cm <- confusionMatrix(factor(test_predictions, levels = levels(test_actual)), test_actual)
      
      # Call the function to estimate priors
      priors <- confusion_matrix_method(train_cm, test_cm)
      # Access estimated priors
      p_w1 <- priors$p_w1  # Estimated prior for class 1
      p_w2 <- priors$p_w2  # Estimated prior for class 2
      
      priors_list_cm[[result_key]] <- c(p_w1, p_w2)
      # Extracts the first prior (p_w1) from each element in the priors_list_cm and stores them in prior_1_cm
      prior_1_cm <- sapply(priors_list_cm, function(x) x[1])
      
      # Compute the posterior probabilities using Bayes' theorem by estimated priors by CM
      posteriors_cm <- data.frame(
        P_w1_given_X_cm = (test_pred_probs[, 1] * p_w1) / (test_pred_probs[, 1] * p_w1 + test_pred_probs[, 2] * p_w2),
        P_w2_given_X_cm = (test_pred_probs[, 2] * p_w2) / (test_pred_probs[, 1] * p_w1 + test_pred_probs[, 2] * p_w2)
      )
      # Re-calculate test predictions using adjusted posterior probabilities
      test_predictions_adjusted_cm <- ifelse(posteriors_cm$P_w1_given_X_cm > posteriors_cm$P_w2_given_X_cm, CLASS_1, CLASS_2)
      test_accuracy_after_adjustment_by_cm <- mean(test_predictions_adjusted_cm == test_actual)
      
      # Store results
      results_list[[paste0("train", train_size[1], "_test", test_size[1], "_rep", rep)]] <- list(
        mean_absolute_deviation = mad,  # Mean Absolute Deviation
        estimated_prior_em = prior_1_em,
        estimated_prior_cm = prior_1_cm  )
      
      # Store results of accuracy
      accuracy_list[[paste0("train", train_size[1], "_test", test_size[1], "_rep", rep)]] <- list(
        accuracy_no_adjustment = test_accuracy_before_adjustment,
        accuracy_after_em = test_accuracy_after_adjustment_by_em,
        accuracy_after_cm = test_accuracy_after_adjustment_by_cm, 
        accuracy_by_known_priori = test_accuracy_after_adjustment_known_priori
      )
    }
  }
}

# Initialize results container
results_summary <- list()
accuracy_summary <- list()

# Collect results for each combination of train and test sizes
for (train_size in train_sizes) {
  for (test_size in test_sizes) {
    # Collect all replication results for this combination
    subset_results <- lapply(1:num_replications, function(rep) {
      result_key <- paste0("train", train_size[1], "_test", test_size[1], "_rep", rep)
      results_list[[result_key]]
    })
    accuracy_results <- lapply(1:num_replications, function(rep) {
      result_key <- paste0("train", train_size[1], "_test", test_size[1], "_rep", rep)
      accuracy_list[[result_key]]
    })
    
    # Convert list to data frame
    df <- bind_rows(subset_results)
    accuracy <- bind_rows(accuracy_results)
    
    # Calculate metrics for results summary
    results_summary[[paste0("train", train_size[1], "_test", test_size[1])]] <- data.frame(
      Training_Set = paste0("(", train_size[1], ", ", train_size[2], ")"),
      Test_Set = paste0("(", test_size[1], ", ", test_size[2], ")"),
      Mean_Absolute_Deviation = mean(df$mean_absolute_deviation),
      Estimated_Prior_EM = mean(df$estimated_prior_em) * 100,  # Convert to percentage
      Estimated_Prior_CM = mean(df$estimated_prior_cm) * 100   # Convert to percentage
    )
    
    # Calculate metrics for accuracy summary
    accuracy_summary[[paste0("train", train_size[1], "_test", test_size[1])]] <- data.frame(
      Training_Set = paste0("(", train_size[1], ", ", train_size[2], ")"),
      Test_Set = paste0("(", test_size[1], ", ", test_size[2], ")"),
      Avg_accuracy_no_adjustment = mean(accuracy$accuracy_no_adjustment, na.rm = TRUE),
      Avg_accuracy_after_em = mean(accuracy$accuracy_after_em, na.rm = TRUE),
      Avg_accuracy_after_cm = mean(accuracy$accuracy_after_cm, na.rm = TRUE),
      Avg_accuracy_by_known_priori = mean(accuracy$accuracy_by_known_priori, na.rm = TRUE)
    )
  }
}

# Convert list to data frames
results_df <- bind_rows(results_summary)
results_accuracy <- bind_rows(accuracy_summary)

# Print the classification tables in the desired format
print(results_df)
print(results_accuracy)

# Create a combined label for grouping
example_df <- results_accuracy %>%
  mutate(
    Group_Label = paste0("Training=", Training_Set, "; Test=", Test_Set)
  )

# Define the specific order of labels
desired_order <- c(
  "Training=(500, 500); Test=(200, 800)",
  "Training=(250, 250); Test=(200, 800)",
  "Training=(100, 100); Test=(200, 800)",
  "Training=(50, 50); Test=(200, 800)",
  "Training=(500, 500); Test=(100, 400)",
  "Training=(250, 250); Test=(100, 400)",
  "Training=(100, 100); Test=(100, 400)",
  "Training=(50, 50); Test=(100, 400)",
  "Training=(500, 500); Test=(40, 160)",
  "Training=(250, 250); Test=(40, 160)",
  "Training=(100, 100); Test=(40, 160)",
  "Training=(50, 50); Test=(40, 160)",
  "Training=(500, 500); Test=(20, 80)",
  "Training=(250, 250); Test=(20, 80)",
  "Training=(100, 100); Test=(20, 80)",
  "Training=(50, 50); Test=(20, 80)"
)
# Convert to long format 
example_long <- example_df %>%
  pivot_longer(cols = c(Avg_accuracy_no_adjustment, Avg_accuracy_after_em, Avg_accuracy_after_cm, Avg_accuracy_by_known_priori),
               names_to = "Condition",
               values_to = "ClassificationRate")

# Convert Group_Label to a factor with levels in the desired order
example_long$Group_Label <- factor(
  example_long$Group_Label,
  levels = desired_order
)
# Adjust the legend labels
example_long$Condition <- factor(example_long$Condition, 
                                 levels = c("Avg_accuracy_by_known_priori", 
                                            "Avg_accuracy_no_adjustment", 
                                            "Avg_accuracy_after_em", 
                                            "Avg_accuracy_after_cm"),
                                 labels = c("Using true priors", 
                                            "No adjustment", 
                                            "After adjustment by EM", 
                                            "After adjustment by Confusion Matrix"))

# Plot the data with the adjusted legend labels
p_example <- ggplot(example_long, aes(x = Group_Label, y = ClassificationRate, color = Condition, shape = Condition, group = Condition)) +
  geom_line() +
  geom_point() +
  labs(
    x = "Number of Observations in Training and Test Sets",
    y = "Average Classification Rate (%)",
    title = "Average Classification Rate by Condition"
  ) +
  theme_minimal() +
  theme(
    legend.position = "top",
    axis.text.x = element_text(angle = 90, hjust = 1)  # Rotate x-axis labels for readability
  ) +
  scale_y_continuous(labels = scales::percent_format())  # Convert y-axis labels to percentage format

# Print the plot
print(p_example)


