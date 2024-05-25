import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import odr
from scipy.stats import skew, kendalltau, kurtosis, theilslopes
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pymannkendall as mk

df_full = pd.read_csv("wiser_gnip-monthly-gr-gnipmgr01_Athens_Pendeli.csv", usecols=[12,16,18,23,24,25])
df_full = df_full.sort_values(by='Date')  # Because Date column didn't have the right order
df_full['Date'] = pd.to_datetime(df_full['Date'])
df_full.set_index('Date', inplace=True)
df_full['D-excess'] = df_full['H2'] - 8*df_full['O18']

# Drop the rows of my dataframe where either H2 or O18 values are NaN
df = df_full.dropna(subset=['O18', 'H2'])
df['Month'] = df.index.month

# Calculate the mean temperature and precipitation amount of each month
monthly_mean_temp_pendeli = df.groupby(['Month']).mean()['Air Temperature']
monthly_mean_precip_pendeli = df.groupby(['Month']).mean()['Precipitation']


# Calculation of Min. Max. Mean. Std. wMean and wStd for H2, O18 and D-ecxess
columns_to_calculate = ['H2', 'O18', 'D-excess']

for column in columns_to_calculate:
    # Calculate statistics for the current column
    min_value = df[column].min()
    max_value = df[column].max()
    mean_value = df[column].mean()
    std_deviation = df[column].std()
    std_of_mean = std_deviation / np.sqrt(len(df))

    # Calculate weighted mean and weighted standard deviation for the current column
    wMean = (df[column] * df['Precipitation']).sum() / df['Precipitation'].sum()
    wStd = np.sqrt((df['Precipitation'] * (df[column] - wMean) ** 2).sum() / (
                df['Precipitation'] - 1).sum())
    std_of_wMean = np.sqrt((df['Precipitation'] * (df[column] - wMean) ** 2).sum() /
                           ((df['Precipitation'] - 1).sum() * df['Precipitation'].sum()))

    # Print or use the calculated statistics as needed
    print(f"\nStatistics for column '{column}':")
    print(f"Min: {min_value}, Max: {max_value}, Mean: {mean_value}, Std Deviation: {std_deviation}")
    print(f"Std of mean: {std_of_mean}")
    print(f"Weighted Mean: {wMean}, Weighted Std Deviation: {wStd}")
    print(f"Std of Weighted Mean: {std_of_wMean}\n")


# ODLSR regression
x1 = df['O18']
y1 = df['H2']
# Define the model function
def linear_model(params, x1):
    a, b = params
    return a * x1 + b

# Create a Model object
odr_model = odr.Model(linear_model)

# Create a RealData object
data = odr.RealData(x1, y1)

# Set up ODR with the model and data
odr_dinstance = odr.ODR(data, odr_model, beta0=[0.2, 1.])

# Run the regression
odr_result = odr_dinstance.run()
odr_result.pprint()

# Calculate R-squared
r2 = r2_score(y1, linear_model(odr_result.beta, x1))

# Print R-squared value
print(f'R-squared: {r2}')


# PWLS
x2 = df['O18']
y2 = df['H2']

# Put precipitation values as weights
weights = df['Precipitation']

# Perform weighted least squares regression
model = sm.WLS(y2, sm.add_constant(x2), weights=weights)
results = model.fit()
results.summary()  # std's of slope and constant (results.bse[1/0])

x_extended = np.linspace(min(x1), max(x1), 100)  # Generate additional x values

plt.scatter(x1, y1, s=20, color='grey')
plt.plot(x_extended, odr_result.beta[0] * x_extended + odr_result.beta[1] , 'r', label = u'$\delta^{2}$H = 7.35(±0.15)*'u'$\delta^{18}$O + 8.94(±0.99)‰'
                                                                       u'\n ' u' R\u00b2 = 0.92, n=200, ODLS')
plt.plot(x2, results.params[1] * x2 + results.params[0] , 'b', label = u'$\delta^{2}$H = 7.26(±0.12)*'u'$\delta^{18}$O + 9.43(±0.92)‰ '
                                                                     u'\n ' u' R\u00b2 = 0.95, n=200, PWLS')
plt.plot(x1, 8 * x1 + 10, 'black', label = u'$\delta^{2}$H = 8*'u'$\delta^{18}$O + 10‰ (GMWL)')
plt.xlabel(u'$\delta^{18}$O (‰) vs VSMOW')
plt.ylabel(u'$\delta^{2}$H (‰) vs VSMOW')
plt.legend(bbox_to_anchor=(1.05, 0.6), loc='upper left', borderaxespad=0., frameon=False)
plt.title("Meteoric Lines Pendeli ODLS & PWLS regression")
plt.show()
plt.savefig("MeteoricLines_ODLS&PWLS_Pendeli.png", format="png", dpi=150, bbox_inches="tight")
plt.close()


# Temperature dependence
# Fit a second grade equation
X2 = df['Air Temperature']
Y2 = df['H2']

# Fit a second-order polynomial (degree=2)
coefficients = np.polyfit(X2, Y2, 2)

# Create a polynomial function based on the coefficients
poly_function = np.poly1d(coefficients)

# Generate x values for the line of best fit
x_fit = np.linspace(min(X2), max(X2), 100)

# Calculate corresponding y values using the polynomial function
y_fit = poly_function(x_fit)

# Calculate R-squared value
y_pred = poly_function(X2)
r_squared = r2_score(Y2, y_pred)


# PWLS  regression
Y3= df['H2']
X3= df['Air Temperature']
X3= sm.add_constant(X3)
weights3 = df['Precipitation']

model3 = sm.WLS(Y3, X3, weights=weights3)
results3 = model3.fit()
results3.summary()

highlight_dates = ['2001-09-15', '2006-07-15', '2011-09-15', '2017-08-15', '2020-07-15', '2020-08-15']
highlight_indices = [df.index.get_loc(date) for date in highlight_dates]

plt.scatter(X2, Y2, s=20)
plt.plot(x_fit, y_fit, label=u'$\delta^{2}$H = 0.008*Τ\u00b2 + 1.2*Τ - 55‰ \n R\u00b2 = 0.23, Second-order polynomial', color='red')
plt.plot(X3['Air Temperature'], results3.params[1] * X3['Air Temperature'] + results3.params[0], color='darkorange', label = u'$\delta^{2}$H = 1.28*T - 61.0‰\n'
                                                                                              u', R\u00b2 = 0.18, PWLS')
plt.scatter(X2.iloc[highlight_indices], Y2.iloc[highlight_indices], color='deeppink', s=20, label='Summer enriched points')
plt.xlabel("Air Temperature T($^{\circ}C$)")
plt.ylabel(u'$\delta^{2}$H (‰)')
plt.legend(fontsize=8)
plt.title("Temperature dependence")
plt.savefig("2Η_Temp_Pendeli.png", format="png", dpi=150, bbox_inches="tight")
plt.close()


# Fit a second grade equation
X2 = df['Air Temperature']
Y2 = df['O18']

# Fit a second-order polynomial (degree=2)
coefficients = np.polyfit(X2, Y2, 2)

# Create a polynomial function based on the coefficients
poly_function = np.poly1d(coefficients)

# Generate x values for the line of best fit
x_fit = np.linspace(min(X2), max(X2), 100)

# Calculate corresponding y values using the polynomial function
y_fit = poly_function(x_fit)

# Calculate R-squared value
y_pred = poly_function(X2)
r_squared = r2_score(Y2, y_pred)


# PWLS  regression
Y3= df['O18']
X3= df['Air Temperature']
X3= sm.add_constant(X3)
weights3 = df['Precipitation']

model3 = sm.WLS(Y3, X3, weights=weights3)
results3 = model3.fit()
results3.summary()

highlight_dates = ['2001-09-15', '2006-07-15', '2011-09-15', '2017-08-15', '2020-07-15', '2020-08-15']
highlight_indices = [df.index.get_loc(date) for date in highlight_dates]

plt.scatter(X2, Y2, s=20)
plt.plot(x_fit, y_fit, label=u'$\delta^{18}$O = 0.002*Τ\u00b2 + 0.20*Τ - 9.4‰ \n R\u00b2 = 0.35, Second-order polynomial', color='red')
plt.plot(X3['Air Temperature'], results3.params[1] * X3['Air Temperature'] + results3.params[0], color='darkorange', label = u'$\delta^{18}$O = 0.219*T - 10.20‰\n'
                                                                                              u', R\u00b2 = 0.28, PWLS')
plt.scatter(X2.iloc[highlight_indices], Y2.iloc[highlight_indices], color='deeppink', s=20, label='Summer enriched points')
plt.xlabel("Air Temperature T($^{\circ}C$)")
plt.ylabel(u'$\delta^{18}$O (‰)')
plt.legend(fontsize=8)
plt.title("Temperature dependence")
plt.savefig("O18_Temp_Pendeli.png", format="png", dpi=150, bbox_inches="tight")
plt.close()


# Precipitation amount effect
# OLS regression
Y = df['H2']
X = df['Precipitation']
X = sm.add_constant(X)
model = sm.OLS(Y,X)
results = model.fit()
print(results.summary())

# Logarithmic
fit = np.polyfit(np.log(df['Precipitation']), df['H2'], 1)
# Calculate the predicted values based on the logarithmic fit
predicted_values = np.polyval(fit, np.log(df['Precipitation']))
r_squared = r2_score(df['H2'], predicted_values)
# Generate x values for the logarithmic regression line
x_log = np.linspace(min(df['Precipitation']), max(df['Precipitation']), 100)
y_log = np.polyval(fit, np.log(x_log))

plt.scatter(X['Precipitation'],Y, s=20, marker='^', color='orange')
plt.plot(x_log, y_log, label=u'$\delta^{2}$H = -8.5*Ln(P) - 7.3‰, R\u00b2 = 0.39', color='red')
plt.plot(df['Precipitation'], results.params[1]*df['Precipitation'] + results.params[0], label=u'$\delta^{2}$H = -0.19*P - 25.6‰, '
                                                                                               u'R\u00b2 = 0.28', color='blue')
plt.xlabel("Precipitation amount (mm)")
plt.ylabel(u'$\delta^{2}$H (‰)')
plt.legend(fontsize=8)
plt.savefig("H2_precip_alldataPendeli.png", format="png", dpi=150, bbox_inches="tight")
plt.close()


# Separate winter season (DJF) and summer season (JJA) and do the same using OLS
df_summer = df[df.index.month.isin([6, 7, 8])]
df_winter = df[df.index.month.isin([12,1,2])]

# Results for logarithmic fit in summer data
fit = np.polyfit(np.log(df_summer['Precipitation']), df_summer['O18'], 1)
# Calculate the predicted values based on the logarithmic fit
predicted_values = np.polyval(fit, np.log(df_summer['Precipitation']))
r_squared = r2_score(df_summer['O18'], predicted_values)


Y_winter = df_winter['H2']
X_winter = df_winter['Precipitation']
X_winter = sm.add_constant(X_winter)
model_winter = sm.OLS(Y_winter,X_winter)
results_winter = model_winter.fit()
print(results_winter.summary())

Y_summer = df_summer['H2']
X_summer = df_summer['Precipitation']
X_summer = sm.add_constant(X_summer)
model_summer = sm.OLS(Y_summer,X_summer)
results_summer = model_summer.fit()
print(results_summer.summary())

plt.scatter(X_winter['Precipitation'],Y_winter, s=20, marker='^', color ='maroon', label='DJF data')
plt.scatter(X_summer['Precipitation'],Y_summer, s=20, marker='^', color ='lime', label='JJA data')
plt.plot(X_summer['Precipitation'], results_summer.params[1]*X_summer['Precipitation'] + results_summer.params[0],
         label=u'$\delta^{2}$H = -0.30*P + 16.5, R\u00b2 = 0.13')
plt.plot(X_winter['Precipitation'], results_winter.params[1]*X_winter['Precipitation'] + results_winter.params[0],
         label=u'$\delta^{2}$H = -0.11*P - 39.2, R\u00b2 = 0.18')
plt.xlabel("Precipitation amount (mm)")
plt.ylabel(u'$\delta^{2}$H (‰)')
plt.legend(fontsize=8)
plt.savefig("H2_precip_djf&jja_Pendeli.png", format="png", dpi=150, bbox_inches="tight")
plt.close()

Y = df['O18']
X = df['Precipitation']
X = sm.add_constant(X)
model = sm.OLS(Y,X)
results = model.fit()
print(results.summary())

# Logarithmic
fit = np.polyfit(np.log(df['Precipitation']), df['O18'], 1)
# Calculate the predicted values based on the logarithmic fit
predicted_values = np.polyval(fit, np.log(df['Precipitation']))
r_squared = r2_score(df['O18'], predicted_values)
# Generate x values for the logarithmic regression line
x_log = np.linspace(min(df['Precipitation']), max(df['Precipitation']), 100)
y_log = np.polyval(fit, np.log(x_log))


plt.scatter(X['Precipitation'],Y, s=20, marker='^', color='gray')
plt.plot(x_log, y_log, label=u'$\delta^{18}$O = -1.32*Ln(P) - 1.66‰, R\u00b2 = 0.47', color='lightcoral')
plt.plot(df['Precipitation'], results.params[1]*df['Precipitation'] + results.params[0], label=u'$\delta^{18}$O = -0.028*P - 4.54‰, '
                                                                                               u'R\u00b2 = 0.32', color='slateblue')
plt.xlabel("Precipitation amount (mm)")
plt.ylabel(u'$\delta^{18}$O (‰)')
plt.legend(fontsize=8)
plt.savefig("O18_precip_alldataPendeli.png", format="png", dpi=150, bbox_inches="tight")
plt.close()


df_summer = df[df.index.month.isin([6, 7, 8])]
df_winter = df[df.index.month.isin([12,1,2])]

Y_winter = df_winter['O18']
X_winter = df_winter['Precipitation']
X_winter = sm.add_constant(X_winter)
model_winter = sm.OLS(Y_winter,X_winter)
results_winter = model_winter.fit()
print(results_winter.summary())

Y_summer = df_summer['O18']
X_summer = df_summer['Precipitation']
X_summer = sm.add_constant(X_summer)
model_summer = sm.OLS(Y_summer,X_summer)
results_summer = model_summer.fit()
print(results_summer.summary())

plt.scatter(X_winter['Precipitation'],Y_winter, s=20, marker='^', color ='hotpink', label='DJF data')
plt.scatter(X_summer['Precipitation'],Y_summer, s=20, marker='^', color ='darkturquoise', label='JJA data')
plt.plot(X_summer['Precipitation'], results_summer.params[1]*X_summer['Precipitation'] + results_summer.params[0],
         label=u'$\delta^{2}$H = -0.044*P - 0.018, R\u00b2 = 0.16', color='turquoise')
plt.plot(X_winter['Precipitation'], results_winter.params[1]*X_winter['Precipitation'] + results_winter.params[0],
         label=u'$\delta^{2}$H = -0.013*P - 7.01, R\u00b2 = 0.19', color='deeppink')
plt.xlabel("Precipitation amount (mm)")
plt.ylabel(u'$\delta^{18}$O (‰)')
plt.legend(fontsize=8)
plt.savefig("O18_precip_djf&jja_Pendeli.png", format="png", dpi=150, bbox_inches="tight")
plt.close()


# Check for outliers (Boxplots method)
# Calculate the first and third quartiles
first_quartile = np.percentile(df['O18'], 25)
third_quartile = np.percentile(df['O18'], 75)
IQ = third_quartile-first_quartile
lower_inner_fence = first_quartile - (1.5*IQ)
upper_inner_fence = third_quartile + (1.5*IQ)
lower_outer_fence = first_quartile - (3*IQ)
upper_outer_fence = third_quartile + (3*IQ)

# Identify suspected outliers
suspected_outliers = df['O18'][(df['O18'] > upper_inner_fence) & (df['O18'] <= upper_outer_fence) | (df['O18'] < lower_inner_fence)
                              & (df['O18'] >= lower_outer_fence)]
outliers = df['O18'][(df['O18'] > upper_outer_fence) | (df['O18'] < lower_outer_fence)]

# Create a boxplot
plt.boxplot(df['O18'])

# Set y-axis limits to include both lower and upper outer fences
plt.ylim(lower_outer_fence - 2, upper_outer_fence + 3)

# Add labels and title
plt.xlabel('O18')
plt.ylabel(u'$\delta^{18}$O (‰)')
plt.title('Boxplot of O18 values Pendeli')

# Display the first and third quartiles on the plot
plt.text(0.9, first_quartile, f'Q1: {first_quartile:.2f}', va='center', ha='right', bbox=dict(boxstyle='round', fc='none', ec='none'), color='r')
plt.text(0.9, third_quartile, f'Q3: {third_quartile:.2f}', va='center', ha='right', bbox=dict(boxstyle='round', fc='none', ec='none'), color='r')

# Display additional parameters as annotations
plt.text(1.25, lower_inner_fence, f'---------------Lower Inner Fence: {lower_inner_fence:.2f}', va='center', ha='right', bbox=dict(boxstyle='round', fc='none', ec='none'), color='b')
plt.text(1.25, upper_inner_fence, f'---------------Upper Inner Fence: {upper_inner_fence:.2f}', va='center', ha='right', bbox=dict(boxstyle='round', fc='none', ec='none'), color='b')
plt.text(1.25, lower_outer_fence, f'---------------Lower Outer Fence: {lower_outer_fence:.2f}', va='center', ha='right', bbox=dict(boxstyle='round', fc='none', ec='none'), color='b')
plt.text(1.25, upper_outer_fence, f'---------------Upper Outer Fence: {upper_outer_fence:.2f}', va='center', ha='right', bbox=dict(boxstyle='round', fc='none', ec='none'), color='b')
plt.text(0.9, 2, f'suspected outliers', va='center', ha='right', bbox=dict(boxstyle='round', fc='none', ec='none'), color='green')
plt.text(0.9, -14, f'suspected outliers', va='center', ha='right', bbox=dict(boxstyle='round', fc='none', ec='none'), color='green')
plt.text(0.9, 6, f'outliers', va='center', ha='right', bbox=dict(boxstyle='round', fc='none', ec='none'), color='green')

# Show the plot
plt.show()
plt.close()

# Eischeid et al., 1995 ; Beck et al., 2005
p25 = np.percentile(df['H2'], 25)
p50 = np.percentile(df['H2'], 50)
p75 = np.percentile(df['H2'], 75)

test_value = 4*(p75-p25)
outlrs = []
for i in df['H2']:
    if abs(i-p50) >= test_value :
        outlrs.append(i)


# Histograms of data
# Functions  to calculate skewness & kurtosis and return it
def calculate_skewness(data, variable):
    return skew(data[variable])

skewness_O18 = calculate_skewness(df, 'O18')
skewness_H2 = calculate_skewness(df, 'H2')
skewness_dexcess = calculate_skewness(df, 'D-excess')

def calculate_kurtosis(data, variable):
    return kurtosis(data[variable])

kurtosis_O18 = calculate_kurtosis(df, 'O18')
kurtosis_H2 = calculate_kurtosis(df, 'H2')
kurtosis_dexcess = calculate_kurtosis(df, 'D-excess')

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 6))

# Plot histograms for 'H2' in df
axes[0].hist(df['H2'], bins=20, color='red', alpha=0.7, label=f'Skewness: {skewness_H2:.2f} \n '
                                                                 f'Kurtosis: {kurtosis_H2:.2f}')
axes[0].set_title(u'$\delta^{2}$H histogram')
axes[0].set_xlabel(u'$\delta^{2}$H (‰)')
axes[0].set_ylabel('No. of observations')
axes[0].legend()

# Plot histograms for 'O18' in df
axes[1].hist(df['O18'], bins=20, color='blue', alpha=0.7, label=f'Skewness: {skewness_O18:.2f} \n'
                                                                   f'Kurtosis: {kurtosis_O18:.2f}')
axes[1].set_title(u'$\delta^{18}$O histogram')
axes[1].set_xlabel(u'$\delta^{18}$O (‰)')
axes[1].set_ylabel('No. of observations')
axes[1].legend()

# Plot histograms for 'D-excess' in df
axes[2].hist(df['D-excess'], bins=20, color='darkviolet', alpha=0.7, label=f'Skewness: {skewness_dexcess:.2f} \n'
                                                                  f'Kurtosis: {kurtosis_dexcess:.2f}')
axes[2].set_title('D-excess histogram')
axes[2].set_xlabel('D-excess (‰)')
axes[2].set_ylabel('No. of observations')
axes[2].legend()

fig.suptitle('Histograms of Isotope Data in Pendeli (2001-2020)', fontsize=13)
# Adjust layout to avoid overlapping titles
plt.subplots_adjust(top=0.85)
plt.tight_layout()
plt.savefig('isotopes_histograms_Pendeli.png', format="png", dpi=150, bbox_inches="tight")
plt.close()


''' Check for a trend in my timeseries using Mann-Kendall test '''
# I first make the autocorrelation plot to check for serial correlation of my data
sm.graphics.tsa.plot_acf(df['H2'], lags=50)
plt.xlabel('Lags')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation of $\delta^{2}$H data')
plt.show()

# Seasonal MK test
trend, h, p, z, Tau, s, var_s, slope, intercept = mk.seasonal_test(df['H2'])

decomp_h2 = sm.tsa.seasonal_decompose(      # This model calculates the trend by using moving averages over a period of 12 values
  df['H2'], period=60)
decomp_h2_plot = decomp_h2.plot()

fig, axs = plt.subplots(2, 1, figsize=(10, 8))
# Plot original time series
label_text = f'Original Timeseries\nMann-Kendall test: False\nTheil-Sen slope : {slope:.2f}\nP-value: {p:.4f}'
axs[0].plot(df_full.index, df_full['H2'], label=label_text, color='r')
axs[0].set_ylabel(u'$\delta^{2}$H (‰)')
axs[0].legend()

# Plot trend component
axs[1].plot(decomp_h2.trend, label='Trend Component\nSeasonality removed (moving averages of 5 years period) ', color='orange')
axs[1].legend()

# Adjust layout for better visualization
plt.suptitle('Timeseries & Trend of H2 Pendeli data (2001-2020)', fontsize=14)
plt.tight_layout()
plt.savefig("2H_timeseries&trend_Pendeli.png", format="png", dpi=150, bbox_inches="tight")
plt.close()


sm.graphics.tsa.plot_acf(df['O18'], lags=50)
plt.xlabel('Lags')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation of $\delta^{18}$O data')
plt.show()

# Seasonal MK test
trend, h, p, z, Tau, s, var_s, slope, intercept = mk.seasonal_test(df['O18'])

decomp_o18 = sm.tsa.seasonal_decompose(
  df['O18'], period=60)
decomp_o18_plot = decomp_o18.plot()

fig, axs = plt.subplots(2, 1, figsize=(10, 8))
# Plot original time series
label_text = f'Original Timeseries\nMann-Kendall test: False\nTheil-Sen slope : {slope:.2f}\nP-value: {p:.4f}'
axs[0].plot(df_full.index, df_full['O18'], label=label_text)
axs[0].set_ylabel(u'$\delta^{18}$O (‰)')
axs[0].legend()

# Plot trend component
axs[1].plot(decomp_o18.trend, label='Trend Component\nSeasonality removed (moving averages of 5 years period)', color='orange')
axs[1].legend()

# Adjust layout for better visualization
plt.suptitle('Timeseries & Trend of O18 Pendeli data (2001-2020)', fontsize=14)
plt.tight_layout()
plt.savefig("O18_timeseries&trend_Pendeli.png", format="png", dpi=150, bbox_inches="tight")
plt.close()


sm.graphics.tsa.plot_acf(df['D-excess'], lags=50)
plt.xlabel('Lags')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation of D-excess data')
plt.show()

# Seasonal MK test
trend, h, p, z, Tau, s, var_s, slope, intercept = mk.seasonal_test(df['D-excess'])

decomp_dexcess = sm.tsa.seasonal_decompose(
    df['D-excess'], period=60)
decomp_dexcess_plot = decomp_dexcess.plot()

fig, axs = plt.subplots(2, 1, figsize=(10, 8))
# Plot original time series
label_text = f'Original Timeseries\nMann-Kendall test: False\nTheil-Sen slope : {slope:.2f}\nP-value: {p:.4f}'
axs[0].plot(df_full.index, df_full['D-excess'], label=label_text, color='darkviolet')
axs[0].set_ylabel('D-excess (‰)')
axs[0].legend()

# Plot trend component
axs[1].plot(decomp_dexcess.trend, label='Trend Component\nSeasonality removed (moving averages of 5 years period) ', color='orange')
axs[1].legend()

# Adjust layout for better visualization
plt.suptitle('Timeseries & Trend of D-excess Pendeli data (2001-2020)', fontsize=14)
plt.tight_layout()
plt.savefig("D-excess_timeseries&trend_Pendeli.png", format="png", dpi=150, bbox_inches="tight")
plt.close()
