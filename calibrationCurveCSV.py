import csv
from pylab import * #this includes numpy as np!
from scipy.optimize import leastsq
from scipy.optimize import curve_fit

def sigmoid(x, a, b):
    return 100 / (1.0 + np.exp(-a*(x-b)))

# def calibrationCurve():
csv.register_dialect('ssv', delimiter=',', skipinitialspace=True)

# pulls the data from input.txt
data = []
with open('input.txt', 'r') as f:
    reader = csv.reader(f, 'ssv')
    for row in reader:
        floats = [float(column) for column in row]
        data.append(floats)
fullData = np.array(data)

xdata = fullData[:,0]
ydata = fullData[:,1]
std = fullData[:,2]
titleText = 'Calibration curve'

# Fit a sigmoidal fit to the data
# S(x) = 1 / (1 + e^-x)
popt=[NaN, NaN]
if len(xdata) > 2:
    popt, pcov = curve_fit(sigmoid, xdata, ydata, p0 = [1, 0.1], method = 'dogbox')
    if ~isnan(popt[0]):
        xfit = np.linspace(min(xdata), max(xdata), 100)
        yfit = sigmoid(xfit, *popt)

        ss_res = np.sum((ydata - sigmoid(xdata, *popt)) ** 2)
        ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
        rSquared = 1 - (ss_res / ss_tot)

# Plot data and fit
plot(xdata, ydata, 'ko', markersize = 4)
if ~isnan(popt[0]):
    plot(xfit, yfit, 'b--', label = 'Sigmoidal fit\n(R-Squared = %.3f,\na = %.2e, b = %.2e)' % (rSquared, popt[0], popt[1]))

# Plot error bars
errLower = np.array(ydata) - np.array(std)
errUpper = np.array(ydata) + np.array(std)
for k in range(0, len(xdata)):
    plot([xdata[k], xdata[k]], [errLower[k], errUpper[k]], 'r', 'linewidth', 2)

# Figure configuration
ylim((-40, 100))
xlim((min(xdata) * .5, max(xdata) * 2))
fig = gcf()
fig.canvas.manager.set_window_title(titleText)
ax = gca()
ax.set_xscale('log')
ax.invert_yaxis()
ax.grid()
xlabel('Concentration (M)')
ylabel('Peak Suppression (%)')
title(titleText)
if ~isnan(popt[0]):
    legend(loc='upper right')

show()
# return 0

# calibrationCurve()