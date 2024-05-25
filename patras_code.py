import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import odr
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pymannkendall as mk

df_full = pd.read_csv("wiser_gnip-monthly-gr-gnipmgr01_Patras.csv", usecols=[12,16,18,23,24,25])
df_full = df_full.sort_values(by='Date')  # Because Date column didn't have the right order
df_full['Date'] = pd.to_datetime(df_full['Date'])
df_full.set_index('Date', inplace=True)
df_full['D-excess'] = df_full['H2'] - 8*df_full['O18']

# Drop the rows of my dataframe where either H2 or O18 values are NaN
df = df_full.dropna(subset=['O18', 'H2'])

df['Month'] = df.index.month

df_2000_2009 = df['2000-10-15':'2009-12-15'] # 2000-2009 data
df_2010_2022 = df['2010-01-15':'2022-12-15'] # 2010-2022 data

# Calculate the mean temperature and precipitation amount of each month
monthly_mean_temp_patras = df.groupby(['Month']).mean()['Air Temperature']
monthly_mean_precip_patras = df.groupby(['Month']).mean()['Precipitation']


# Calculation of Min. Max. Mean. Std. wMean and wStd for H2, O18 and D-ecxess
dataframes = [df, df_2000_2009, df_2010_2022]
columns_to_calculate = ['H2', 'O18', 'D-excess']

for df_current in dataframes:
    for column in columns_to_calculate:
        # Calculate statistics for the current column
        min_value = df_current[column].min()
        max_value = df_current[column].max()
        mean_value = df_current[column].mean()
        std_deviation = df_current[column].std()
        std_of_mean = std_deviation / np.sqrt(len(df_current))

        # Calculate weighted mean and weighted standard deviation for the current column
        wMean = (df_current[column] * df_current['Precipitation']).sum() / df_current['Precipitation'].sum()
        wStd = np.sqrt((df_current['Precipitation'] * (df_current[column] - wMean) ** 2).sum() / (df_current['Precipitation']-1).sum())
        std_of_wMean = np.sqrt((df_current['Precipitation'] * (df_current[column] - wMean) ** 2).sum() /
                               ((df_current['Precipitation']-1).sum() * df_current['Precipitation'].sum()))

        # Print or use the calculated statistics as needed
        print(f"\nStatistics for column '{column}':")
        print(f"Min: {min_value}, Max: {max_value}, Mean: {mean_value}, Std Deviation: {std_deviation}")
        print(f"Std of mean: {std_of_mean}")
        print(f"Weighted Mean: {wMean}, Weighted Std Deviation: {wStd}")
        print(f"Std of Weighted Mean: {std_of_wMean}\n")


'''Here i calculate the mean temperature of each month of the year, both for 2000-2009 and 2010-2022. I do the same for the
mean precipitation amount of each month and then plot the results with barplots'''
# Calculate the mean temperature value of each month for each period
monthly_mean_temp_2000_2009 = df_2000_2009.groupby(['Month']).mean()['Air Temperature']
monthly_mean_temp_2010_2022 = df_2010_2022.groupby(['Month']).mean()['Air Temperature']

# Plot for period 2000-2009
plt.bar(monthly_mean_temp_2000_2009.index, monthly_mean_temp_2000_2009, width=0.4, label='2000-2009')

# Plot for period 2010-2022
plt.bar(monthly_mean_temp_2010_2022.index + 0.4, monthly_mean_temp_2010_2022, width=0.4, label='2010-2022')

plt.xlabel('Month')
plt.ylabel('Mean Air Temperature T($^{\circ}C$)')
plt.xticks(monthly_mean_temp_2000_2009.index + 0.2, ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend()
plt.grid(True)

# Show plot
plt.show()


# Calculate the mean precipitation amount of each month for each period
monthly_mean_precip_2000_2009 = df_2000_2009.groupby(['Month']).mean()['Precipitation']
monthly_mean_precip_2010_2022 = df_2010_2022.groupby(['Month']).mean()['Precipitation']


# Plot for period 2000-2009
plt.bar(monthly_mean_precip_2000_2009.index, monthly_mean_precip_2000_2009, width=0.4, label='2000-2009', color='blue')

# Plot for period 2010-2022
plt.bar(monthly_mean_precip_2010_2022.index + 0.4, monthly_mean_precip_2010_2022, width=0.4, label='2010-2022', color='red')

plt.bar(monthly_mean_precip_patras.index + 0.8, monthly_mean_precip_patras, width=0.4, label='Whole period', color='darkviolet' )

plt.xlabel('Month')
plt.ylabel('Precipitation amount (mm)')
plt.xticks(monthly_mean_precip_2000_2009.index + 0.2, ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend()
plt.grid(True)

# Show plot
plt.show()


# OLS regression
Y = df['H2']
X = df['O18']
X = sm.add_constant(X)
model = sm.OLS(Y,X)
results = model.fit()
print(results.params)
print(results.summary())

plt.scatter(X['O18'], Y, s=20)
plt.plot(X['O18'], results.params[1] * X['O18'] + results.params[0], 'r', label = u'$\delta^{2}$H = 6.99*'u'$\delta^{18}$O + 5.97‰'
                                                                                  u'\n R\u00b2 = 0.91, n=201' )
plt.plot(X['O18'], 8 * X['O18'] + 10, 'black', label = u'$\delta^{2}$H = 8*'u'$\delta^{18}$O + 10‰ (GMWL)')
plt.xlabel(u'$\delta^{18}$O (‰) vs VSMOW')
plt.ylabel(u'$\delta^{2}$H (‰) vs VSMOW')
plt.legend()
plt.title("Meteoric Line Patras (OLS regression)")
plt.savefig("MeteoricLine_OLS_Patras.png", format="png", dpi=150, bbox_inches="tight")
plt.close()


# ODLSR regression
x = df['O18'].values
y = df['H2'].values
# Define the model function
def linear_model(params, x):
    a, b = params
    return a * x + b

# Create a Model object
odr_model = odr.Model(linear_model)

# Create a RealData object
data = odr.RealData(x, y)

# Set up ODR with the model and data
odr_dinstance = odr.ODR(data, odr_model, beta0=[0.2, 1.])

# Run the regression
odr_result = odr_dinstance.run()
odr_result.pprint()

# Calculate R-squared
r2 = r2_score(y, linear_model(odr_result.beta, x))

# Print R-squared value
print(f'R-squared: {r2}')


plt.scatter(x, y, s=20)
plt.plot(x, odr_result.beta[0] * x + odr_result.beta[1] , 'r', label = u'$\delta^{2}$H = 7.71(±0.18)*'u'$\delta^{18}$O + 9.41(±0.93)‰'
                                                                       u'\n ' u' R\u00b2 = 0.90, n=201')
plt.plot(x, 8 * x + 10, 'black', label = u'$\delta^{2}$H = 8*'u'$\delta^{18}$O + 10‰ (GMWL)')
plt.xlabel(u'$\delta^{18}$O (‰) vs VSMOW')
plt.ylabel(u'$\delta^{2}$H (‰) vs VSMOW')
plt.legend(bbox_to_anchor=(1.05, 0.6), loc='upper left', borderaxespad=0., frameon=False)
plt.title("Meteoric Line Patras (ODLS regression)")
plt.savefig("MeteoricLine_ODLSR_Patras.png", format="png", dpi=150, bbox_inches="tight")
plt.close()

# PWLS regression
x = df['O18']
y = df['H2']

# Put precipitation values as weights
weights = df['Precipitation']

# Perform weighted least squares regression
model = sm.WLS(y, sm.add_constant(x), weights=weights)
results = model.fit()
results.summary()  # std's of slope and constant (results.bse[1/0])

plt.scatter(x, y, s=20)
plt.plot(x, results.params[1] * x + results.params[0] , 'r', label = u'$\delta^{2}$H = 7.21(±0.18)*'u'$\delta^{18}$O + 7.91(±1.09)‰ '
                                                                     u'\n ' u' R\u00b2 = 0.89, n=201')
plt.plot(x, 8 * x + 10, 'black', label = u'$\delta^{2}$H = 8*'u'$\delta^{18}$O + 10‰ (GMWL)')
plt.xlabel(u'$\delta^{18}$O (‰) vs VSMOW')
plt.ylabel(u'$\delta^{2}$H (‰) vs VSMOW')
plt.legend(bbox_to_anchor=(1.05, 0.6), loc='upper left', borderaxespad=0., frameon=False)
plt.title("Meteoric Line Patras (PWLS regression)")
plt.savefig("MeteoricLine_PWLSR_Patras.png", format="png", dpi=150, bbox_inches="tight")
plt.close()


# MWL (PWLS regression) for the 2010-2022 period, plotted against the 2000-2009 period
x = df['O18']
x1 = df_2000_2009['O18']
y1 = df_2000_2009['H2']
x2 = df_2010_2022['O18']
y2 = df_2010_2022['H2']

weights1 = df_2000_2009['Precipitation']
model1 = sm.WLS(y1, sm.add_constant(x1), weights=weights1)
results1 = model1.fit()
results1.summary()

weights2 = df_2010_2022['Precipitation']
model2 = sm.WLS(y2, sm.add_constant(x2), weights=weights2)
results2 = model2.fit()
results2.summary()

plt.scatter(x1, y1, s=20, marker='s', color='black', label='2000-2009')
plt.scatter(x2, y2, s=20, color='grey', label='2010-2022')

x_extended = np.linspace(min(x), max(x), 100)  # Generate additional x values
plt.plot(x_extended, results1.params[1] * x_extended + results1.params[0], 'purple', label=u'$\delta^{2}$H = 7.36(±0.36)*'u'$\delta^{18}$O + 8.10(±2.25)‰'
                                                                          u'\n2000-2009, R\u00b2 = 0.85, n=78')
plt.plot(x_extended, results2.params[1] * x_extended + results2.params[0], 'crimson', label=u'$\delta^{2}$H = 7.05(±0.19)*'u'$\delta^{18}$O + 7.37(±1.09)‰'
                                                                          u'\n2010-2022, R\u00b2 = 0.92, n=123')
plt.plot(x, 8 * x + 10, ls=':', color='skyblue', label = u'$\delta^{2}$H = 8*'u'$\delta^{18}$O + 10‰ (GMWL)')
plt.xlabel(u'$\delta^{18}$O (‰) vs VSMOW')
plt.ylabel(u'$\delta^{2}$H (‰) vs VSMOW')
plt.legend(bbox_to_anchor=(1.05, 0.6), loc='upper left', borderaxespad=0., frameon=False)
plt.savefig("MeteoricLine_PWLSR(00-09&10-22)_Patras.png", format="png", dpi=150, bbox_inches="tight")
plt.close()


# Temperature dependence
# Fit a second grade equation
X2 = df['Air Temperature']
Y2 = df['H2']

# Fit a second-order polynomial (degree=2)
coefficients, cov_matrix = np.polyfit(X2, Y2, 2)

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

# Summer enriched samples due to weak precipitation amount
highlight_dates = ['2005-06-15', '2006-08-15', '2008-05-15', '2015-07-15', '2018-07-15', '2020-06-15', '2020-08-15']
highlight_indices = [df.index.get_loc(date) for date in highlight_dates]

plt.scatter(X2, Y2, s=20)
plt.plot(x_fit, y_fit, label=u'$\delta^{2}$H = 0.035*Τ\u00b2 + 0.4*Τ - 45‰ \n R\u00b2 = 0.27, Second-order polynomial', color='red')
plt.plot(X3['Air Temperature'], results3.params[1] * X3['Air Temperature'] + results3.params[0], color='darkorange', label = u'$\delta^{2}$H = 1.17*T - 50.4‰\n'
                                                                                              u', R\u00b2 = 0.18, PWLS')
plt.scatter(X2.iloc[highlight_indices], Y2.iloc[highlight_indices], color='deeppink', s=20, label='Summer enriched points')
plt.xlabel("Air Temperature T($^{\circ}C$)")
plt.ylabel(u'$\delta^{2}$H (‰)')
plt.legend(fontsize=8)
plt.savefig("2Η_Temp_Patras.png", format="png", dpi=150, bbox_inches="tight")
plt.close()


# Fit a second grade equation
X2 = df['Air Temperature']
Y2 = df['O18']

# Fit a second-order polynomial (degree=2)
coefficients = np.polyfit(X2, Y2, 2, cov=True)

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

# Identify the indices of the dates with summer enriched values
highlight_dates = ['2003-08-15', '2005-06-15', '2006-08-15', '2008-05-15', '2018-07-15', '2020-06-15', '2020-08-15']
highlight_indices = [df.index.get_loc(date) for date in highlight_dates]

plt.scatter(X2, Y2, s=20)
plt.plot(x_fit, y_fit, label=u'$\delta^{18}$O = 0.005*Τ\u00b2 + 0.06*Τ - 7.3‰ \n R\u00b2 = 0.31, Second-order polynomial', color='red')
plt.plot(X3['Air Temperature'], results3.params[1] * X3['Air Temperature'] + results3.params[0], color='darkorange', label = u'$\delta^{18}$O = 0.158*T - 8.03‰\n'
                                                                                              u', R\u00b2 = 0.19, PWLS')
plt.scatter(X2.iloc[highlight_indices], Y2.iloc[highlight_indices], color='deeppink', s=20, label='Summer enriched points')
plt.xlabel("Air Temperature T($^{\circ}C$)")
plt.ylabel(u'$\delta^{18}$O (‰)')
plt.legend(fontsize=8)
plt.savefig("O18_Temp_Patras.png", format="png", dpi=150, bbox_inches="tight")
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
plt.plot(x_log, y_log, label=u'$\delta^{2}$H = -7.6*Ln(P) + 2.2‰, R\u00b2 = 0.27', color='red')
plt.plot(df['Precipitation'], results.params[1]*df['Precipitation'] + results.params[0], label=u'$\delta^{2}$H = -0.11*P - 18.9‰, '
                                                                                               u'R\u00b2 = 0.18', color='blue')
plt.xlabel("Precipitation amount (mm)")
plt.ylabel(u'$\delta^{2}$H (‰)')
plt.legend(fontsize=8)
plt.savefig("H2_precip_alldataPatras.png", format="png", dpi=150, bbox_inches="tight")
plt.close()

# Separate winter season (DJF) and summer season (JJA) and do the same using OLS
df_summer = df[df.index.month.isin([6, 7, 8])]
df_winter = df[df.index.month.isin([12,1,2])]

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
         label=u'$\delta^{2}$H = -0.76*P + 2.5, R\u00b2 = 0.25')
plt.plot(X_winter['Precipitation'], results_winter.params[1]*X_winter['Precipitation'] + results_winter.params[0],
         label=u'$\delta^{2}$H = -0.03*P - 33.7, R\u00b2 = 0.03')
plt.xlabel("Precipitation amount (mm)")
plt.ylabel(u'$\delta^{2}$H (‰)')
plt.legend(fontsize=8)
plt.savefig("H2_precip_djf&jja_Patras.png", format="png", dpi=150, bbox_inches="tight")
plt.close()


# OLS regression
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
plt.plot(x_log, y_log, label=u'$\delta^{18}$O = -1.22*Ln(P) - 0.047‰, R\u00b2 = 0.37', color='lightcoral')
plt.plot(df['Precipitation'], results.params[1]*df['Precipitation'] + results.params[0], label=u'$\delta^{18}$O = -0.017*P - 3.43‰, '
                                                                                               u'R\u00b2 = 0.24', color='slateblue')
plt.xlabel("Precipitation amount (mm)")
plt.ylabel(u'$\delta^{18}$O (‰)')
plt.legend(fontsize=8)
plt.savefig("O18_precip_alldataPatras.png", format="png", dpi=150, bbox_inches="tight")
plt.close()

# Separate winter season (DJF) and summer season (JJA) and do the same using OLS
df_summer = df[df.index.month.isin([6, 7, 8])]
df_winter = df[df.index.month.isin([12, 1, 2])]

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
         label=u'$\delta^{2}$H = -0.107*P - 0.28, R\u00b2 = 0.22', color='turquoise')
plt.plot(X_winter['Precipitation'], results_winter.params[1]*X_winter['Precipitation'] + results_winter.params[0],
         label=u'$\delta^{2}$H = -0.0045*P - 5.63, R\u00b2 = 0.06', color='deeppink')
plt.xlabel("Precipitation amount (mm)")
plt.ylabel(u'$\delta^{18}$O (‰)')
plt.legend(fontsize=8)
plt.savefig("O18_precip_djf&jja_Patras.png", format="png", dpi=150, bbox_inches="tight")
plt.close()


# Check for outliers (Boxplots method)
# Calculate the first and third quartiles
first_quartile = np.percentile(df['H2'], 25)
third_quartile = np.percentile(df['H2'], 75)
IQ = third_quartile-first_quartile
lower_inner_fence = first_quartile - (1.5*IQ)
upper_inner_fence = third_quartile + (1.5*IQ)
lower_outer_fence = first_quartile - (3*IQ)
upper_outer_fence = third_quartile + (3*IQ)

# Identify suspected outliers
suspected_outliers = df['H2'][(df['H2'] > upper_inner_fence) & (df['H2'] <= upper_outer_fence) | (df['H2'] < lower_inner_fence)
                              & (df['H2'] >= lower_outer_fence)]
outliers = df['H2'][(df['H2'] > upper_outer_fence) | (df['H2'] < lower_outer_fence)]

# Create a boxplot
plt.boxplot(df['H2'])

# Add labels and title
plt.xlabel('H2')
plt.ylabel(u'$\delta^{2}$H (‰)')
plt.title('Boxplot of H2 values Patras')

# Display the first and third quartiles on the plot
plt.text(0.9, first_quartile, f'Q1: {first_quartile:.2f}', va='center', ha='right', bbox=dict(boxstyle='round', fc='none', ec='none'), color='r')
plt.text(0.9, third_quartile, f'Q3: {third_quartile:.2f}', va='center', ha='right', bbox=dict(boxstyle='round', fc='none', ec='none'), color='r')

# Display additional parameters as annotations
plt.text(1.25, lower_inner_fence, f'---------------Lower Inner Fence: {lower_inner_fence:.2f}', va='center', ha='right', bbox=dict(boxstyle='round', fc='none', ec='none'), color='b')
plt.text(1.25, upper_inner_fence, f'---------------Upper Inner Fence: {upper_inner_fence:.2f}', va='center', ha='right', bbox=dict(boxstyle='round', fc='none', ec='none'), color='b')
plt.text(1.25, lower_outer_fence, f'---------------Lower Outer Fence: {lower_outer_fence:.2f}', va='center', ha='right', bbox=dict(boxstyle='round', fc='none', ec='none'), color='b')
plt.text(1.25, upper_outer_fence, f'---------------Upper Outer Fence: {upper_outer_fence:.2f}', va='center', ha='right', bbox=dict(boxstyle='round', fc='none', ec='none'), color='b')
plt.text(0.9, 25, f'suspected outliers', va='center', ha='right', bbox=dict(boxstyle='round', fc='none', ec='none'), color='green')
plt.text(0.9, -80, f'suspected outliers', va='center', ha='right', bbox=dict(boxstyle='round', fc='none', ec='none'), color='green')
plt.text(0.9, -100, f'outliers', va='center', ha='right', bbox=dict(boxstyle='round', fc='none', ec='none'), color='green')

# Show the plot
plt.show()
plt.close()

# Eischeid et al., 1995 ; Beck et al., 2005
p25 = np.percentile(df['O18'], 25)
p50 = np.percentile(df['O18'], 50)
p75 = np.percentile(df['O18'], 75)

test_value = 4*(p75-p25)
outlrs = []
for i in df['O18']:
    if abs(i-p50) >= test_value :
        outlrs.append(i)


# Boxplots of H2, O18 for 2000-2009 and 2010-2022 periods
h2_values = pd.merge(df_2000_2009[['H2']], df_2010_2022[['H2']], left_index=True, right_index=True, how='outer', suffixes=('_00_09', '_10_22'))
boxplot_data = [h2_values['H2_00_09'].dropna(), h2_values['H2_10_22'].dropna()]
plt.boxplot(boxplot_data)
plt.ylabel(u'$\delta^{2}$H (‰)')
plt.xticks([1, 2], ['2000-2009', '2010-2022'])
plt.grid(True)
plt.title('Boxplot of H2 values for 2000-2009 & 2010-2022 periods')

# Add median values as text annotations
for i, data in enumerate(boxplot_data, start=1):
    median_value = round(data.median(), 2)
    mean_value = round(data.mean(), 2)
    plt.text(i + 0.2, median_value, f"Mean: {mean_value}\nMedian: {median_value}", ha='center', va='bottom', color='red', fontsize=10)

plt.show()


o18_values = pd.merge(df_2000_2009[['O18']], df_2010_2022[['O18']], left_index=True, right_index=True, how='outer', suffixes=('_00_09', '_10_22'))
boxplot_data = [o18_values['O18_00_09'].dropna(), o18_values['O18_10_22'].dropna()]
plt.boxplot(boxplot_data)
plt.ylabel(u'$\delta^{18}$O (‰)')
plt.xticks([1, 2], ['2000-2009', '2010-2022'])
plt.grid(True)
plt.title('Boxplot of O18 values for 2000-2009 & 2010-2022 periods')

# Add median values as text annotations
for i, data in enumerate(boxplot_data, start=1):
    median_value = round(data.median(), 2)
    mean_value = round(data.mean(), 2)
    plt.text(i + 0.2, median_value, f"Mean: {mean_value}\nMedian: {median_value}", ha='center', va='bottom', color='red', fontsize=10)

plt.show()


# Histograms for both 2000-2022 & 2000-2009 data
# Functions  to calculate skewness & kurtosis and return it
def calculate_skewness(data, variable):
    return skew(data[variable])

skewness_O18_2010_2022 = calculate_skewness(df_2010_2022, 'O18')
skewness_O18_2000_2009 = calculate_skewness(df_2000_2009, 'O18')
skewness_H2_2010_2022 = calculate_skewness(df_2010_2022, 'H2')
skewness_H2_2000_2009 = calculate_skewness(df_2000_2009, 'H2')
skewness_dexcess_2010_2022 = calculate_skewness(df_2010_2022, 'D-excess')
skewness_dexcess_2000_2009 = calculate_skewness(df_2000_2009, 'D-excess')

def calculate_kurtosis(data, variable):
    return kurtosis(data[variable])

kurtosis_O18_2010_2022 = calculate_kurtosis(df_2010_2022, 'O18')
kurtosis_O18_2000_2009 = calculate_kurtosis(df_2000_2009, 'O18')
kurtosis_H2_2010_2022 = calculate_kurtosis(df_2010_2022, 'H2')
kurtosis_H2_2000_2009 = calculate_kurtosis(df_2000_2009, 'H2')
kurtosis_dexcess_2010_2022 = calculate_kurtosis(df_2010_2022, 'D-excess')
kurtosis_dexcess_2000_2009 = calculate_kurtosis(df_2000_2009, 'D-excess')

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))

# Plot histograms for 'H2' in df_2010_2022
axes[0,0].hist(df_2010_2022['H2'], bins=20, color='red', alpha=0.7, label=f'Skewness: {skewness_H2_2010_2022:.2f} \n '
                                                                 f'Kurtosis: {kurtosis_H2_2010_2022:.2f}')
axes[0,0].set_title(u'$\delta^{2}$H 2010-2022')
axes[0,0].set_xlabel(u'$\delta^{2}$H (‰)')
axes[0,0].set_ylabel('No. of observations')
axes[0,0].set_xlim((-100,40))
axes[0,0].legend(fontsize=7)

# Plot histograms for 'O18' in df_2010_2022
axes[0,1].hist(df_2010_2022['O18'], bins=20, color='blue', alpha=0.7, label=f'Skewness: {skewness_O18_2010_2022:.2f} \n'
                                                                   f'Kurtosis: {kurtosis_O18_2010_2022:.2f}')
axes[0,1].set_title(u'$\delta^{18}$0 2010-2022')
axes[0,1].set_xlabel(u'$\delta^{18}$O (‰)')
axes[0,1].set_ylabel('No. of observations')
axes[0,1].set_xlim((-15,5))
axes[0,1].legend(fontsize=7)

# Plot histograms for 'D-excess' in df_2010_2022
axes[0,2].hist(df_2010_2022['D-excess'], bins=20, color='darkviolet', alpha=0.7, label=f'Skewness: {skewness_dexcess_2010_2022:.2f} \n'
                                                                  f'Kurtosis: {kurtosis_dexcess_2010_2022:.2f}')
axes[0,2].set_title('D-excess 2010-2022')
axes[0,2].set_xlabel('D-excess (‰)')
axes[0,2].set_ylabel('No. of observations')
axes[0,2].set_xlim((-34.5,25))
axes[0,2].legend(fontsize=7)

# Plot histograms for 'H2' in df_2000_2009
axes[1,0].hist(df_2000_2009['H2'], bins=20, color='red', alpha=0.7, label=f'Skewness: {skewness_H2_2000_2009:.2f} \n'
                                                                           f'Kurtosis: {kurtosis_H2_2000_2009:.2f}')
axes[1,0].set_title(u'$\delta^{2}$H 2000-2009')
axes[1,0].set_xlabel(u'$\delta^{2}$H (‰)')
axes[1,0].set_ylabel('No. of observations')
axes[1,0].set_xlim((-100,40))
axes[1,0].legend(fontsize=7)

# Plot histograms for 'O18' in df_2000_2009
axes[1,1].hist(df_2000_2009['O18'], bins=20, color='blue', alpha=0.7, label=f'Skewness: {skewness_O18_2000_2009:.2f} \n'
                                                                             f'Kurtosis: {kurtosis_O18_2000_2009:.2f}')
axes[1,1].set_title(u'$\delta^{18}$O 2000-2009')
axes[1,1].set_xlabel(u'$\delta^{18}$O (‰)')
axes[1,1].set_ylabel('No. of observations')
axes[1,1].set_xlim((-15,5))
axes[1,1].legend(fontsize=7)

# Plot histograms for 'D-excess' in df_2000_2009
axes[1,2].hist(df_2000_2009['D-excess'], bins=20, color='darkviolet', alpha=0.7, label=f'Skewness: {skewness_dexcess_2000_2009:.2f} \n'
                                                                             f'Kurtosis: {kurtosis_dexcess_2000_2009:.2f}')
axes[1,2].set_title('D-excess 2000-2009')
axes[1,2].set_xlabel('D-excess (‰)')
axes[1,2].set_ylabel('No. of observations')
axes[1,2].set_xlim((-34.5,25))
axes[1,2].legend(fontsize=7)

fig.suptitle('Histograms of Isotope Data in Patras (2010-2022 & 2000-2009)', fontsize=13)
# Adjust layout
plt.tight_layout()
plt.savefig('isotopes_histograms_Patras_0009_1022.png', format="png", dpi=150, bbox_inches="tight")
plt.close()


''' Check for a trend in my timeseries using Mann-Kendall test '''
# First the autocorrelation plot to check for serial correlation of my data
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
plt.suptitle('Timeseries & Trend of H2 Patras data (2000-2022)', fontsize=14)
plt.tight_layout()
plt.savefig("2H_timeseries&trend_Patras.png", format="png", dpi=150, bbox_inches="tight")
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
plt.suptitle('Timeseries & Trend of O18 Patras data (2000-2022)', fontsize=14)
plt.tight_layout()
plt.savefig("O18_timeseries&trend_Patras.png", format="png", dpi=150, bbox_inches="tight")
plt.close()


sm.graphics.tsa.plot_acf(df['D-excess'], lags=50)
plt.xlabel('Lags')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation of D-excess data')
plt.show()

# Correlated seasonal MK test
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
plt.suptitle('Timeseries & Trend of D-excess Patras data (2000-2022)', fontsize=14)
plt.tight_layout()
plt.savefig("D-excess_timeseries&trend_Patras.png", format="png", dpi=150, bbox_inches="tight")
plt.close()



