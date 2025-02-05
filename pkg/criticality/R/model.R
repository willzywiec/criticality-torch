# Model Function
#
#' This function builds the deep neural network metamodel architecture using Torch.
#' @param dataset Training and test data
#' @param layers String that defines the deep neural network architecture (e.g., "64-64")
#' @param loss Loss function
#' @param opt.alg Optimization algorithm
#' @param learning.rate Learning rate
#' @param ext.dir External directory (full path)
#' @return A deep neural network metamodel
#' @export
#' @import torch
#' @import magrittr

Model <- function(dataset,
                  layers = '8192-256-256-256-256-16',
                  loss = 'sse',
                  opt.alg = 'adamax',
                  learning.rate = 0.00075,
                  ext.dir) {
  
  # Parse the layers string
  layers <- strsplit(layers, '-') %>% unlist() %>% as.integer()

  # Define the model architecture using nn.Module
  model <- nn_module(
    initialize = function() {
      self$fc1 <- nn_linear(dim(dataset$training.df)[2], layers[1])
      
      # Dynamically create the layers
      if (length(layers) > 1) {
        self$fc2 <- nn_linear(layers[1], layers[2])
      }
      if (length(layers) > 2) {
        self$fc3 <- nn_linear(layers[2], layers[3])
      }
      if (length(layers) > 3) {
        self$fc4 <- nn_linear(layers[3], layers[4])
      }
      if (length(layers) > 4) {
        self$fc5 <- nn_linear(layers[4], layers[5])
      }
      if (length(layers) > 5) {
        self$fc6 <- nn_linear(layers[5], layers[6])
      }
      if (length(layers) > 6) {
        self$fc7 <- nn_linear(layers[6], layers[7])
      }
      if (length(layers) > 7) {
        self$fc8 <- nn_linear(layers[7], layers[8])
      }
      if (length(layers) > 8) {
        self$fc9 <- nn_linear(layers[8], layers[9])
      }
      
      # Output layer
      self$output <- nn_linear(layers[length(layers)], 1)
    },
    
    forward = function(x) {
      # Forward pass through the network
      x <- torch_relu(self$fc1(x))
      if (length(layers) > 1) {
        x <- torch_relu(self$fc2(x))
      }
      if (length(layers) > 2) {
        x <- torch_relu(self$fc3(x))
      }
      if (length(layers) > 3) {
        x <- torch_relu(self$fc4(x))
      }
      if (length(layers) > 4) {
        x <- torch_relu(self$fc5(x))
      }
      if (length(layers) > 5) {
        x <- torch_relu(self$fc6(x))
      }
      if (length(layers) > 6) {
        x <- torch_relu(self$fc7(x))
      }
      if (length(layers) > 7) {
        x <- torch_relu(self$fc8(x))
      }
      if (length(layers) > 8) {
        x <- torch_relu(self$fc9(x))
      }
      x <- self$output(x)
      return(x)
    }
  )

  # Define loss function
  if (loss == 'sse') {
    criterion <- nn_mse_loss()
  }
  
  # Define optimizer based on selected algorithm
  if (opt.alg == 'adadelta') {
    optimizer <- optim_adadelta(model$parameters, lr = learning.rate)
  } else if (opt.alg == 'adagrad') {
    optimizer <- optim_adagrad(model$parameters, lr = learning.rate)
  } else if (opt.alg == 'adam') {
    optimizer <- optim_adam(model$parameters, lr = learning.rate)
  } else if (opt.alg == 'adamax') {
    optimizer <- optim_adamax(model$parameters, lr = learning.rate)
  } else if (opt.alg == 'nadam') {
    optimizer <- optim_nadam(model$parameters, lr = learning.rate)
  } else if (opt.alg == 'rmsprop') {
    optimizer <- optim_rmsprop(model$parameters, lr = learning.rate)
  }
  
  # Return the model and optimizer
  return(list(model = model, criterion = criterion, optimizer = optimizer))
}
