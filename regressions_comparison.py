''' In this code I compare the different types of regressions that I have applied for the calculation of the
Meteoric Lines (OLS, ODLS, PWLS) '''

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import odr
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

df_full = pd.read_csv("wiser_gnip-monthly-gr-gnipmgr01_Athens_Thission.csv", usecols=[12,16,18,23,24,25])
df_full = df_full.sort_values(by='Date')  # Because Date column didn't have the right order
df_full['Date'] = pd.to_datetime(df_full['Date'])
df_full.set_index('Date', inplace=True)
df_full['D-excess'] = df_full['H2'] - 8*df_full['O18']

# Drop the rows of my dataframe where either H2 or O18 values are NaN
df = df_full.dropna(subset=['O18', 'H2'])

# Regressions for the Meteoric Lines
# ODLS
x_odls = df['O18']
y_odls = df['H2']
# Define the model function
def linear_model(params, x):
    a, b = params
    return a * x + b

# Create a Model object
odr_model = odr.Model(linear_model)

# Create a RealData object
data = odr.RealData(x_odls, y_odls)

# Set up ODR with the model and data
odr_dinstance = odr.ODR(data, odr_model, beta0=[0.2, 1.])

# Run the regression
odr_result = odr_dinstance.run()
odr_result.pprint()

y_hat_odls = odr_result.beta[0] * x_odls + odr_result.beta[1]

# PWLS
y_pwls = df['H2']
x_pwls = sm.add_constant(df['O18'])
# Put precipitation values as weights
weights = df['Precipitation']

# Perform weighted least squares regression
model_pwls = sm.WLS(y_pwls, x_pwls, weights=weights )
results_pwls = model_pwls.fit()
results_pwls.summary()

y_hat_pwls = results_pwls.params[1] * x_pwls['O18'] + results_pwls.params[0]

# Calculation of residuals for my models
resid_odls = y_odls - y_hat_odls
resid_pwls = y_pwls - y_hat_pwls

print(f'ODLS regression mean residual value = {resid_odls.mean(): .3f}')
print(f'PWLS regression mean residual value = {resid_pwls.mean(): .3f}')

residuals = pd.concat([resid_odls, resid_pwls], axis=1)
plt.boxplot(residuals)
plt.title('Linear regression residuals boxplots $\delta^{2}$H')
plt.xlabel('Model')
plt.ylabel('$\delta^{2}$H - $\delta^{2}$H,modelled (‰)')
plt.xticks([1, 2], ['ODLS', 'PWLS'])
plt.grid(True)
plt.show()











