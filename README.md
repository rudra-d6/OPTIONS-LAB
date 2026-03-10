Options Lab – Black–Scholes vs Monte Carlo

An interactive option pricing simulator built in Python with Streamlit.
This project implements both the Black–Scholes model (closed-form solution) and a Monte Carlo simulation to price European call and put options.
The app allows you to explore how different parameters (spot price, strike, volatility, maturity, interest rates) affect option pricing, and compare the speed vs accuracy of exact vs simulated methods.



Features

Black–Scholes pricer (fast analytical solution).
Monte Carlo simulation (numerical approximation).
Interactive Streamlit web app with sliders & inputs.
Side-by-side comparison of Black–Scholes and Monte Carlo results.
Configurable number of simulations.
Clean, styled UI with result boxes and difference output.



Background

Options are financial contracts that give the holder the right (but not the obligation) to buy (call) or sell (put) an asset at a specified strike price before expiry.
Black–Scholes model: A closed-form formula for pricing European options, assuming lognormal asset prices and constant volatility.
Monte Carlo method: A flexible simulation technique that generates many possible future stock price paths, then averages the discounted payoffs.
Comparing the two highlights the trade-off between analytical speed and simulation flexibility.



Installation

Clone the repo and install dependencies:
git clone https://github.com/rudra-d6/options-lab.git
cd options-lab
python -m pip install -r requirements.txt



Usage

Launch the app with:
"streamlit run app.py"
Within a gitbash terminal


Adjust parameters interactively:

Spot price (S)
Strike price (K)
Time to maturity (T)
Risk-free rate (r)
Volatility (σ)
Option type (call/put)
Monte Carlo simulations



The app displays:
Black–Scholes option price
Monte Carlo option price
Difference between the two methods



Testing

Run the unit tests with:
python -m pytest -q



Tests check:

Known reference cases (ATM call/put).
Put–call parity.
Edge cases (T=0, σ=0).
Monte Carlo convergence to Black–Scholes.



Tech Stack

Python
NumPy
SciPy
Streamlit
Matplotlib
pytest



Roadmap
 Add payoff distribution charts.
 Plot Monte Carlo convergence.
 Export results to CSV/PDF.
 Add variance reduction techniques (antithetic variates, control variates).



License
This project is licensed under the MIT License.

