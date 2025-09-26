# Homework 1: Multilayer Perceptron Training

**Course:** ECE49595CV / ECE59500CV – Introduction to Computer Vision  
**Semester:** Fall 2025  
**Due Date:** Friday, September 26, 2025 @ 5:00 PM ET  
**Submission Method:** Brightspace  
**Execution Platform:** RCAC Scholar (CPU only)

---

## 📘 Overview

This assignment involves implementing and training a **Multilayer Perceptron (MLP)** using **naive gradient descent** from scratch. You will:

- Build MLPs with configurable architecture
- Train them on two datasets
- Evaluate performance across different configurations

---

## 🧠 Concepts

- **Linear Layer:**  
  $$\text{linear}(x; W, b) = Wx + b$$

- **Sigmoid Activation:**  
  $$\text{sigmoid}(x) = \frac{1}{1 + e^{-x}}$$

- **Single Layer Perceptron (SLP):**  
  $$\text{slp}(x; W, b) = \text{sigmoid}(\text{linear}(x; W, b))$$

- **Multilayer Perceptron (MLP):**  
  $$\text{mlp}(x; W_1, ..., W_n, b_1, ..., b_n) = \text{slp}(...\text{slp}(x; W_1, b_1)...; W_n, b_n)$$

- **Loss Function (L2 Distance):**  
  $$\text{loss}(u, v) = (u - v) \cdot (u - v)$$

- **Gradient Descent:**  
  $$x_{i+1} = x_i - \eta \nabla f(x_i)$$

---

## 🧪 Tasks

1. Implement MLP training using naive gradient descent.
2. Create two datasets:
   - **XOR Function:** 2 inputs → 1 output
   - **Two-bit Binary Adder:** 5 inputs → 3 outputs
3. Experiment with:
   - Training/test splits
   - Hyperparameters (layers, hidden units)
   - Learning rates
   - Number of iterations
4. Evaluate test loss for each configuration.

---

## ⚙️ Requirements

- **Languages Allowed:** C, C++, Java, Python (basic standard libraries only)
- **No Libraries:** No PyTorch, TensorFlow, JAX, or other AD tools. But able to use the math library for the exp function and numpy for arrays.
- **No GPUs:** CPU-only execution
- **No Parallelization Required**
- **Must run in < 5 minutes on RCAC Scholar**

---

## 📂 Submission Instructions

1. Create a directory named after your **Purdue Career Account**.
2. Include all files, especially a **`run`** bash script.
3. Make `run` executable:
   ```bash
   chmod a+x run
   ```

---

## ▶️ How to run (local)

- Full experiments sweep (default when using the bash `run` script; prints to console):
  ```bash
  ./run
  ```

- Quick smoke test sweep:
  ```bash
  ./run quick
  ```

- Other modes:
  ```bash
  ./run xor    # XOR-focused sweep
  ./run adder  # Adder-focused sweep
  ./run both   # Small both-datasets sweep
  ```

- Train one configuration:
  ```bash
  python3 train.py --dataset xor --layers 1 --hidden 4 --lr 0.5 --iters 2000 --seed 0 --test-ratio 0.25
  ```

- Full experiments sweep (edit ranges as desired, prints summary to console):
  ```bash
  python3 experiments.py --dataset both --layers 0,1,2 --hidden 2,4,8 --lr 0.1,0.5,1.0 --iters 500,1000,2000 --test-ratio 0.25,0.5
  ```

If you want to save the console output to a file yourself:

- Bash/macOS/Linux:
  ```bash
  python3 experiments.py --quick | tee results.txt
  ```
- PowerShell (Windows):
  ```powershell
  python .\experiments.py --quick | Tee-Object -FilePath results.txt
  ```

Note: The `run` script is a bash script intended for Linux/macOS or Windows environments with WSL/Cygwin. On vanilla Windows PowerShell, call the Python entry points directly as shown above.

All code uses NumPy for arrays (no autograd frameworks), sigmoid activations, L2 loss, and batch gradient descent.

---

## ⚙️ Implementation notes

- Arrays: Implemented with NumPy for vectorized forward/backward; CPU-only.
- Activation: Numerically stable sigmoid to avoid overflow/underflow.
- Loss: L2 as mean over samples of per-sample squared-error sum across outputs.
- Optimization: Full-batch gradient descent; gradients averaged once in dL/da, not again at parameter update.
- Initialization: Weights drawn from uniform and scaled by 1/sqrt(fan_in); biases initialized to 0.

Tips
- XOR has only 4 samples; with test_ratio=0.25 the test set often has 1 sample, so test loss can fluctuate a lot. For sanity checks, use `--test-ratio 0.0` or a fixed split.
- Two-bit adder is harder; try `--layers 2 --hidden 16 --lr 0.25 --iters 5000` for stronger performance.

---

## ▶️ Windows quick start

From PowerShell, run:

```powershell
cd "c:\Users\Simon\Desktop\Courses\ECE 59500 CV\ece59595cv-hw-1"; python .\experiments.py --quick
```

---

## 🧩 File summary

- `mymlp.py` — MLP implementation (sigmoid activations, L2 loss, batch GD)
- `datasets.py` — XOR and two-bit adder datasets, plus train/test split
- `train.py` — CLI to train one model and report losses
- `experiments.py` — Parameter sweep runner; prints results to console
- `run` — Bash entrypoint; runs sweeps and prints to console

