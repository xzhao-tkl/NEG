# Neuron Empricial Gradient (NEG)

This repository contains the official implementation for the ACL 2025 paper:

**Neuron Empirical Gradient: Discovering and Quantifying Neurons Global Linear Controllability**

## Overview

This project introduces a novel method to empirically measure and analyze the global linear controllability of individual neurons in LLMs. 
Our approach provides new insights into neuron-level interpretability and intervention.



## Features

#### **`NeurGrad`: Neuron Intervention & NEG calculation & NeurGrad**  
    
- Implements neuron intervention experiment to quantify the linear controllability of individual neurons. 

- Provides the calculation method for Neuron Empirical Gradient (NEG) (NeurGrad) and the efficient NEG estimation method NeurGrad, enabling scalable analysis of neuron linearity in large language models.

#### **`NeuronProbe`: Skill Neuron Probing & MCEval8K Benchmark**  

- Includes skill-neuron probing methods to assess whether NEG captures diverse language skills at the neuron level. 

- Contains code for constructing the MCEval8K benchmark—a multi-genre, multi-choice evaluation suite designed to test knowledge and skill diversity in LLMs.

## Getting Started

1. **Clone the repository:**
    ```bash
    git clone https://github.com/xzhao-tkl/NEG.git
    cd NEG
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. Modify the path for your project
    ```python3
    ### In NeuronProbe.__init__.py
    DATA_CACHE_ROOT = "<path/to/your/data/cache/root>" 
    ### In NeurGrad.__init__.py
    MODEL_PATH_ROOT = "<path/to/your/local/models/root>"
    ```

4. **Run experiments:**
    See the `examples/` directory and the instructions in each folders for details.

## Citation

If you use this code, please cite our paper:

```
@inproceedings{xzhao2025neuron,
  title={Neuron Empirical Gradient: Discovering and Quantifying Neurons Global Linear Controllability},
  author={Xin Zhao and Zehui Jiang and Naoki Yoshinaga},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2025}
}
```

## Contact

For questions or collaborations, please open an issue or contact the authors.
