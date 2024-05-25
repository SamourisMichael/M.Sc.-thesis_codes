''' In this code I apply a Kolmogorov-Smirnov test to check for a difference in the distributions between Pendeli and
Thission

Then I apply a Pettitt test to Pendeli and Thission timeseries to detect for any possible change points'''

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats
import pyhomogeneity as hg

df_full_pendeli = pd.read_csv("wiser_gnip-monthly-gr-gnipmgr01_Athens_Pendeli.csv", usecols=[12,16,18,23,24,25])
df_full_pendeli = df_full_pendeli.sort_values(by='Date')  # Because Date column didn't have the right order
df_full_pendeli['Date'] = pd.to_datetime(df_full_pendeli['Date'])
df_full_pendeli.set_index('Date', inplace=True)
df_full_pendeli['D-excess'] = df_full_pendeli['H2'] - 8*df_full_pendeli['O18']

# Drop the rows of my dataframe where either H2 or O18 values are NaN
df_pendeli = df_full_pendeli.dropna(subset=['O18', 'H2'])


df_full_thission = pd.read_csv("wiser_gnip-monthly-gr-gnipmgr01_Athens_Thission.csv", usecols=[12,16,18,23,24,25])
df_full_thission = df_full_thission.sort_values(by='Date')  # Because Date column didn't have the right order
df_full_thission['Date'] = pd.to_datetime(df_full_thission['Date'])
df_full_thission.set_index('Date', inplace=True)
df_full_thission['D-excess'] = df_full_thission['H2'] - 8*df_full_thission['O18']

# Drop the rows of my dataframe where either H2 or O18 values are NaN
df_thission = df_full_thission.dropna(subset=['O18', 'H2'])


x_H2_pendeli = np.sort(df_pendeli['H2'])
x_H2_thission = np.sort(df_thission['H2'])

cdf_pendeli = np.arange(1, len(df_pendeli) + 1) / len(df_pendeli)
cdf_thission = np.arange(1, len(df_thission) + 1) / len(df_thission)

ks_statistic, p_value = stats.ks_2samp(df_pendeli['H2'], df_thission['H2'], alternative='two-sided')

alpha = 0.05
if p_value < alpha:
    print("The null hypothesis (that the distributions are the same) is rejected.")
else:
    print("The null hypothesis cannot be rejected.")

result_text = f'K-S Test: {"Same" if p_value >= alpha else "Different"}\np-value: {p_value:.4f}'

plt.plot(x_H2_pendeli, cdf_pendeli, label='Pendeli')
plt.plot(x_H2_thission, cdf_thission, label='Thission')
plt.xlabel(u'$\delta^{2}$H (‰)')
plt.ylabel('Probability')
plt.legend(loc='lower right')
plt.title('CDFs of $\\delta^{2}$H for Pendeli & Thission (2000-2020)')
plt.text(58, 0.2, result_text, verticalalignment='center', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8))
plt.grid()
plt.savefig('CDFS_2H_Pend&Thiss_KS_test.png', format="png", dpi=150, bbox_inches="tight")
plt.close()


x_O18_pendeli = np.sort(df_pendeli['O18'])
x_O18_thission = np.sort(df_thission['O18'])

ks_statistic, p_value = stats.ks_2samp(df_pendeli['O18'], df_thission['O18'], alternative='two-sided')

alpha = 0.05
if p_value < alpha:
    print("The null hypothesis (that the distributions are the same) is rejected.")
else:
    print("The null hypothesis cannot be rejected.")

result_text = f'K-S Test: {"Same" if p_value >= alpha else "Different"}\np-value: {p_value:.4f}'

plt.plot(x_O18_pendeli, cdf_pendeli, label='Pendeli')
plt.plot(x_O18_thission, cdf_thission, label='Thission')
plt.xlabel(u'$\delta^{18}$O (‰)')
plt.ylabel('Probability')
plt.legend(loc='lower right')
plt.title('CDFs of $\\delta^{18}$O for Pendeli & Thission (2000-2020)')
plt.text(5.5, 0.2, result_text, verticalalignment='center', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8))
plt.grid()
plt.savefig('CDFS_O18_Pend&Thiss_KS_test.png', format="png", dpi=150, bbox_inches="tight")
plt.close()


# Pettitt-test
pett_h2_pendeli = hg.pettitt_test(df_pendeli['H2'])
pett_o18_pendeli = hg.pettitt_test(df_pendeli['O18'])

pett_h2_thission = hg.pettitt_test(df_thission['H2'])
pett_o18_thission = hg.pettitt_test(df_thission['O18'])


# 2nd way using a function (Ioannidis Master)
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

pettitt_test(df_pendeli['H2'])
pettitt_test(df_pendeli['O18'])

pettitt_test(df_thission['H2'])
pettitt_test(df_thission['O18'])
