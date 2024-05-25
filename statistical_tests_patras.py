''' In this code I apply all the statistical tests needed for my analysis (t-test to check for the equality of my slope
 against 2000-2009 period

 Kolmogorov-Smirnov test to check for a difference in the distribution of the isotopes (2000-2009 versus 2010-2022)

 Pettitt test to check for any change point of my timeseries '''

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import f
from scipy import stats
import pyhomogeneity as hg

df_full = pd.read_csv("wiser_gnip-monthly-gr-gnipmgr01_Patras.csv", usecols=[12,16,18,23,24,25])
df_full = df_full.sort_values(by='Date')  # Because Date column didn't have the right order
df_full['Date'] = pd.to_datetime(df_full['Date'])
df_full.set_index('Date', inplace=True)
df_full['D-excess'] = df_full['H2'] - 8*df_full['O18']

# Drop the rows of my dataframe where either H2 or O18 values are NaN
df = df_full.dropna(subset=['O18', 'H2'])

df_2000_2009 = df['2000-10-15':'2009-12-15'] # 2000-2009 data
df_2010_2022 = df['2010-01-15':'2022-12-15'] # 2010-2022 data

''' Here i statistically compare the slope of my PWLS regression to the equivalent slope 
of the 2000-2009 period '''

## 2010-2022 (PWLS regression)
x1 = df_2010_2022['O18']
y1 = df_2010_2022['H2']

# Put precipitation values as weights
weights1 = df_2010_2022['Precipitation']

# Perform weighted least squares regression
model1= sm.WLS(y1, sm.add_constant(x1), weights=weights1)
results1 = model1.fit()
results1.summary()  # std's of slope and constant (results.bse[1/0])


## 2000-2009 (PWLS regression)
x2 = df_2000_2009['O18']
y2 = df_2000_2009['H2']

# Put precipitation values as weights
weights2 = df_2000_2009['Precipitation']

# Perform weighted least squares regression
model2 = sm.WLS(y2, sm.add_constant(x2), weights=weights2)
results2 = model2.fit()
results2.summary()


# F-test to check if my regressions variances are equal or not
# For the first regression
observed_values1 = y1
fitted_values1= results1.fittedvalues
residuals1 = observed_values1 - fitted_values1
variance1 = np.var(residuals1)

# For the second regression
observed_values2 = y2
fitted_values2 = results2.fittedvalues
residuals2 = observed_values2 - fitted_values2
variance2 = np.var(residuals2)

print("Residual Variance - 2010-2022 Data: ", variance1)
print("Residual Variance - 2000-2009 Data: ", variance2)

F_statistic = variance1/variance2

# Degrees of freedom
df1 = len(x1) - 2
df2 = len(x2) - 2

# Set the significance level
alpha = 0.05

# Calculate the critical value
F_critical = f.ppf(1 - alpha/2 , df1, df2)

# Compare with critical value
if F_statistic > F_critical:
    print("Reject the null hypothesis. Variances are not equal.")
else:
    print("Fail to reject the null hypothesis. Variances are equal.")

# Alternatively, you can calculate the p-value
p_value = 2 * (1 - f.cdf(F_statistic, df1, df2))

# Compare with significance level
if p_value < alpha:
    print("Reject the null hypothesis. Variances are not equal.")
else:
    print("Fail to reject the null hypothesis. Variances are equal.")

# Use of the appropriate t-statistic
s2_pool = ((len(x1) - 2) * variance1 + (len(x2) - 2) * variance2)/ (len(x1) + len(x2) - 4)
ssx1 = 1/sum((df['O18']-df['O18'].mean())**2)
ssx2 = 1/sum((df_2000_2009['O18']-df_2000_2009['O18'].mean())**2)

texp = (results1.params[1] - results2.params[1])/ np.sqrt(s2_pool * (ssx1 + ssx2))

dof = df1 + df2

pvalue_t = 2 * (1 - stats.t.cdf(abs(texp), dof))

if pvalue_t < alpha:
    print("Reject the null hypothesis. Slopes are not equal.")
else:
    print("Fail to reject the null hypothesis. Slopes are equal.")


''' Here i apply a Kolmogorov-Smirnov test to compare the distributions of 2000-2009 & 2010-2022 periods '''
x_H2_00_09 = np.sort(df_2000_2009['H2'])
x_H2_10_22 = np.sort(df_2010_2022['H2'])

cdf_2000_2009 = np.arange(1, len(df_2000_2009) + 1) / len(df_2000_2009)
cdf_2010_2022 = np.arange(1, len(df_2010_2022) + 1) / len(df_2010_2022)

ks_statistic, p_value = stats.ks_2samp(df_2000_2009['H2'], df_2010_2022['H2'], alternative='two-sided')

alpha = 0.05
if p_value < alpha:
    print("The null hypothesis (that the distributions are the same) is rejected.")
else:
    print("The null hypothesis cannot be rejected.")

result_text = f'K-S Test: {"Same" if p_value >= alpha else "Different"}\np-value: {p_value:.4f}'

plt.plot(x_H2_00_09, cdf_2000_2009, label='2000-2009')
plt.plot(x_H2_10_22, cdf_2010_2022, label='2010-2022')
plt.xlabel(u'$\delta^{2}$H (‰)')
plt.ylabel('Probability')
plt.legend(loc='lower right')
plt.title('CDFs of $\\delta^{2}$H for 2000-2009 and 2010-2022')
plt.text(40, 0.2, result_text, verticalalignment='center', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8))
plt.grid()
plt.savefig('CDFS_2H_KS_test.png', format="png", dpi=150, bbox_inches="tight")
plt.close()


x_O18_00_09 = np.sort(df_2000_2009['O18'])
x_O18_10_22 = np.sort(df_2010_2022['O18'])

ks_statistic, p_value = stats.ks_2samp(df_2000_2009['O18'], df_2010_2022['O18'], alternative='two-sided')

if p_value < alpha:
    print("The null hypothesis (that the distributions are the same) is rejected.")
else:
    print("The null hypothesis cannot be rejected.")

result_text = f'K-S Test: {"Same" if p_value >= alpha else "Different"}\np-value: {p_value:.4f}'

plt.plot(x_O18_00_09, cdf_2000_2009, label='2000-2009')
plt.plot(x_O18_10_22, cdf_2010_2022, label='2010-2022')
plt.xlabel(u'$\delta^{18}$O (‰)')
plt.ylabel('Probability')
plt.legend(loc='lower right')
plt.title('CDFs of $\\delta^{18}$O for 2000-2009 and 2010-2022')
plt.text(4.4, 0.2, result_text, verticalalignment='center', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8))
plt.grid()
plt.savefig('CDFS_O18_KS_test.png', format="png", dpi=150, bbox_inches="tight")
plt.close()


''' Now I apply a Pettitt test to detect any possible change point '''

# 1st way using pyhomogeneity

pett_h2 = hg.pettitt_test(df['H2'])
pett_o18 = hg.pettitt_test(df['O18'])

# 2nd way using a function (Ioannidis M.Sc.)

def pettitt_test(x):

    T = len(x)
    U = []
    for t in range(T):  # t is used to split X into two subseries
        X_stack = np.zeros((t, len(x[t:]) + 1), dtype=int)
        X_stack[:, 0] = x[:t]  # first column is each element of the first subseries
        X_stack[:, 1:] = x[t:]  # all rows after the first element are the second subseries
        # sign test between each element of the first subseries and all elements of the second subseries, summed.
        U.append(np.sign(X_stack[:, 0] - X_stack[:, 1:].transpose()).sum())

    tau = np.argmax(np.abs(U))  # location of change (first data point of second sub-series)
    K = np.max(np.abs(U))
    p = 2 * np.exp(-6 * K ** 2 / (T ** 3 + T ** 2))

    return tau, p, K

pettitt_test(df['H2'])
pettitt_test(df['O18'])



