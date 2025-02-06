# criticality-torch

A collection of functions for modeling fissile material operations in nuclear facilities  
  
## Install the latest release from GitHub:  
![R-CMD-check](https://github.com/willzywiec/criticality-torch/workflows/R-CMD-check/badge.svg)
```r
devtools::install_github('willzywiec/criticality-torch/pkg/criticality')
```
This version of criticality has been modified to use Torch instead of Keras + TensorFlow, due to TF not supporting CUDA 11.2+ in Windows. These modifications were mostly performed with ChatGPT.  
