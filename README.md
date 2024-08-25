This file outlines the purpose of the repository, provides a description of the Tobit model and the specific log-likelihood function used, along with guidance on how to use the scripts in the repository.

---

# Tobit Regression Model Implementation

This repository contains Python code for implementing Tobit regression, a statistical method used for modeling censored dependent variables. The code calculates the Maximum Likelihood Estimation (MLE) based on a reparameterized log-likelihood function.

## About Tobit Regression

Tobit regression is employed when the dependent variable is censored in some interval. A common example is when values below a certain threshold are censored into a single category (commonly zero). This model assumes that there exists a latent variable which follows a linear relationship with the independent variables, affected by a normally distributed error term. The observed outcome is a censored version of this latent variable.

### Model Formula

The latent variable \(y_i^*\) is given by:

\[ y_i^* = X_i \beta + \epsilon_i \]

where:
- \(X_i\) is a vector of explanatory variables,
- \(\beta\) is a vector of coefficients,
- \(\epsilon_i\) follows a normal distribution with mean 0 and variance \(\sigma^2\).

The observed variable <img src="https://latex.codecogs.com/svg.latex? y_i^*" /> is defined as:

<p align="center"> <img src="https://latex.codecogs.com/svg.latex?y_i&space;=&space;\begin{cases}&space;y_i^*&space;&\text{if&space;}&space;y_i^*&space;>&space;y_L&space;\\&space;y_L&space;&\text{otherwise}&space;\end{cases}" title="y_i = \begin{cases} y_i^* & \text{if } y_i^* > y_L \\ y_L & \text{otherwise} \end{cases}" /> </p>

### Log-Likelihood Function

The reparameterization introduced by Olsen is used in this implementation. The parameters are defined as \(\beta = \delta / \gamma\) and \(\sigma^2 = \gamma^{-2}\). The log-likelihood function is then expressed as:

<p align="center"> <img src="https://latex.codecogs.com/svg.latex?\log&space;\mathcal{L}(\delta,&space;\gamma)&space;=&space;\sum_{y_j&space;>&space;y_L}&space;\left[\log&space;\gamma&space;&plus;&space;\log&space;\phi(\gamma&space;y_j&space;-&space;X_j&space;\delta)\right]&space;&plus;&space;\sum_{y_j&space;=&space;y_L}&space;\log&space;\Phi(\gamma&space;y_L&space;-&space;X_j&space;\delta)" title="\log \mathcal{L}(\delta, \gamma) = \sum_{y_j > y_L} \left[\log \gamma + \log \phi(\gamma y_j - X_j \delta)\right] + \sum_{y_j = y_L} \log \Phi(\gamma y_L - X_j \delta)" /> </p>


\[ 
\log \mathcal{L}(\delta, \gamma) = \sum_{y_j > y_L} \left[\log \gamma + \log \phi(\gamma y_j - X_j \delta)\right] + \sum_{y_j = y_L} \log \Phi(\gamma y_L - X_j \delta)
\]

where:
- \(\phi(\cdot)\) is the standard normal probability density function,
- \(\Phi(\cdot)\) is the standard normal cumulative distribution function.

## Repository Structure

- `tobit_reg.py`: Contains the main implementation of the Tobit regression.
- `utils.py`: Helper functions for model diagnostics and data handling.
- `examples/`: Directory containing notebooks and scripts demonstrating the application and performance of the model.

## Installation

Clone this repository using:

```bash
git clone https://github.com/stnwanekezie/myrecipes.git
```

Ensure you have Python installed, and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To run the Tobit regression model, import the main function from the script and provide it with your data:

```python
from tobit.tobit_reg import Tobit

# Assume X and y are your data arrays
results = Tobit(y, X c_lw=y_L, c_up=y_U, verbose=True).fit()
print(results.summary())
```

## Contributing

Contributions are welcome! Please open an issue to discuss proposed changes or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

---

### Notes

- Replace `https://github.com/yourusername/tobit-regression.git` with your actual GitHub repository URL.
- Make sure to add actual scripts (`tobit_regression.py`, `utils.py`, etc.) that correspond to the descriptions provided in the README.
- Adapt the installation and usage instructions based on the actual dependencies and implementation details of your project.

This README provides a clear, structured overview suitable for a GitHub project aimed at users interested in statistics, econometrics, or data science fields.
