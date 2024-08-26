This file outlines the purpose of the repository, provides a description of the Tobit model and the specific log-likelihood function used, along with guidance on how to use the scripts in the repository.

---

# Tobit Regression Model Implementation

This repository contains Python code for implementing Tobit regression, a statistical method used for modeling censored dependent variables. The code calculates the Maximum Likelihood Estimation (MLE) based on a reparameterized log-likelihood function.

## About Tobit Regression

Tobit regression is employed when the dependent variable is censored in some interval. A common example is when values below a certain threshold are censored into a single category (commonly zero). This model assumes that there exists a latent variable which follows a linear relationship with the independent variables, affected by a normally distributed error term. The observed outcome is a censored version of this latent variable.

### Model Formula

The latent variable <img src="https://latex.codecogs.com/svg.latex?y_i^*" title="y_i^*" /> is given by:

<p align="center"><img src="https://latex.codecogs.com/svg.latex?y_i^*=X_i\beta+\epsilon_i" title="y_i^* = X_i \beta + \epsilon_i" /></p>

where:
- <img src="https://latex.codecogs.com/svg.latex?X_i" title="X_i" /> is a vector of explanatory variables,
- <img src="https://latex.codecogs.com/svg.latex?\beta" title="\beta" /> is a vector of coefficients,
- <img src="https://latex.codecogs.com/svg.latex?\epsilon_i" title="\epsilon_i" /> follows a normal distribution with mean 0 and variance <img src="https://latex.codecogs.com/svg.latex?\sigma^2" title="\sigma^2" />.

The observed variable is defined as:

<p align="center"><img src="https://latex.codecogs.com/svg.latex?y_i%20=%20\begin{cases}%20y_L%20&%20\text{if%20}%20y_i^*%20\leq%20y_L%20\\y_i^*%20&%20\text{if%20}%20y_L%20<%20y_i^*%20<%20y_U%20\\%20y_U%20&%20\text{otherwise}%20\end{cases}" title="y_i = \begin{cases} y_L & \text{if } y_i^* \leq y_L \\ y_i^* & \text{if } y_L < y_i^* < y_U \\ y_U & \text{otherwise} \end{cases}" /></p>

### Log-Likelihood Function

Both the standard LLH function and its reparameterization introduced by Olsen is used in this implementation. The latter parameters are defined as <img src="https://latex.codecogs.com/svg.latex?\beta=\delta/\gamma" title="\beta = \delta / \gamma" /> and <img src="https://latex.codecogs.com/svg.latex?\sigma^2=\gamma^{-2}" title="\sigma^2 = \gamma^{-2}" />. The log-likelihood function is then expressed as:

<p align="center"><img src="https://latex.codecogs.com/svg.latex?\log\mathcal{L}(\delta,\gamma)=\sum_{y_j=y_L}\log\Phi(\gamma%20y_L-X_j\delta)+\sum_{y_L<y_j<y_U}\left[\log\gamma+\log\phi(\gamma%20y_j-X_j\delta)\right]+\sum_{y_j=y_U}\log\Phi(X_j\delta-\gamma%20y_U)" title="\log \mathcal{L}(\delta, \gamma) = \sum_{y_j > y_L} \left[\log \gamma + \log \phi(\gamma y_j - X_j \delta)\right] + \sum_{y_j = y_L} \log \Phi(\gamma y_L - X_j \delta)" /></p>

## Repository Structure

- `tobit/tobit_reg.py`: Contains the main implementation of the Tobit regression.
- `requirements.txt`: List of modules requisite to run the module.
Run `tobit_reg.py` directly for an example of the model application.

## Installation

Clone this repository using:

```bash
git clone https://github.com/stnwanekezie/myrecipes.git
cd tobit-regression
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
results = Tobit(y, X, c_lw=y_L, c_up=y_U, verbose=True).fit()
print(results.summary())
```

## Contributing

Contributions are welcome! Please open an issue to discuss proposed changes or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

---
