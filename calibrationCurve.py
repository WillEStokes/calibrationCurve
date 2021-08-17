import csv
from pylab import * #this includes numpy as np!
from scipy.optimize import leastsq
from scipy.optimize import curve_fit

def sigmoid(x, a, b):
    return 100 / (1.0 + np.exp(-a*(x - b)))

def calibrationCurve(conc, mean, std, titleText):
    # Fit a sigmoidal fit to the data
    # S(x) = 1 / (1 + e^-x)
    ppopt, pcov = curve_fit(sigmoid, conc, mean, p0 = [1, 0.1], method = 'dogbox')
    xfit = np.linspace(min(conc), max(conc), 100)
    yfit = sigmoid(xfit, *popt)

    # Plot data and fit
    plot(conc, mean, 'ko', markersize = 4)
    plot(xfit, yfit, 'b--', label = 'Sigmoidal fit')

    # Plot error bars
    errLower = np.array(mean) - np.array(std)
    errUpper = np.array(mean) + np.array(std)
    for k in range(0, len(conc)):
        plot([conc[k], conc[k]], [errLower[k], errUpper[k]], 'r', 'linewidth', 2)

    # Figure configuration
    ylim((-20, 100))
    xlim((min(conc)*.5, max(conc)*2))
    fig = gcf()
    fig.canvas.manager.set_window_title(titleText)
    ax = gca()
    ax.set_xscale('log')
    ax.invert_yaxis()
    ax.grid()
    xlabel('Concentration (M)')
    ylabel('Peak Supression (%)')
    title(titleText)
    legend(loc='upper right')

    show()
    return 0