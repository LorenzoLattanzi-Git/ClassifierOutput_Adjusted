# Adjusting the Output of a Classifier to New a Priori Probabilities

**Course:** Statistics for Data Science, University of Pisa (A.Y. 2023/24)  
**Authors:**  [Lorenzo Lattanzi](https://github.com/LorenzoLattanzi-Git)  , [Minh Duc Pham](https://github.com/DucPhamBP) 

---

## Project Overview

In supervised learning, classifiers are often trained on datasets whose class distributions do not match real-world scenarios. This mismatch can reduce classification performance. The goal of this project is to reproduce and extend the methods described in the paper to adjust classifier outputs when prior class probabilities change.

We implemented and tested multiple approaches:

- **Direct Adjustment via Bayes’ Theorem** – when the new priors are known.
- **Confusion Matrix Method** – estimation of unknown priors.
- **Expectation-Maximization Algorithm** – iterative estimation of unknown priors.
- **Likelihood Ratio Test** – statistical test for changes in priors.

The project includes experiments on both artificial datasets (Ringnorm) and real-world datasets, allowing assessment of performance, robustness, and practical effectiveness. The work demonstrates both theoretical understanding and practical re-implementation using R.

---

## Datasets & Experiments

- **Artificial Data:** Ringnorm dataset with varying training and test sizes; repeated simulations to evaluate robustness.
- **Real Data:** Breast, Diabetes, and Liver datasets for classification adjustment.

### Evaluation Metrics

- Classification rate improvement after adjustment
- MAD (Mean Absolute Deviation) between adjusted and true posteriors
- Robustness analysis across different training/test sizes

---

## Results Highlights

- Adjusting classifier outputs consistently improves classification accuracy.
- EM method generally yields results closer to the true priors than the Confusion Matrix method.
- Adjustments provide robustness even with small training or test sets.

---

## Requirements

- **R** ≥ 4.0  
- Packages: `nnet`, `caret`, `dplyr`, `ggplot2`

---

## Reference

Marco Saerens, Patrice Latinne, Christine Decaestecker (2001). *Adjusting the Outputs of a Classifier to New a Priori Probabilities: A Simple Procedure*. Neural Computation, 14(1):21–41.  
[Link to paper](http://direct.mit.edu/neco/article-pdf/14/1/21/815040/089976602753284446.pdf)