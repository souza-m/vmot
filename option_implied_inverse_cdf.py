import csv
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as pl

# open the CSV file for reading
_dir = './data/'
with open(_dir + 'AMZN_call_price.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # skip the header row

    # initialize the arrays for strike prices and option values using NumPy
    data = np.array(list(reader), dtype=float)
    AMZNstrikes = data[:, 0]
    AMZNcallPrices = data[:, 1:4]

with open(_dir + 'AMZN_put_price.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # skip the header row

    # initialize the arrays for strike prices and option values using NumPy
    data = np.array(list(reader), dtype=float)
    AMZNstrikes = data[:, 0]
    AMZNputPrices = data[:, 1:4]

with open(_dir + 'AAPL_call_price.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # skip the header row

    # initialize the arrays for strike prices and option values using NumPy
    data = np.array(list(reader), dtype=float)
    AAPLstrikes = data[:, 0]
    AAPLcallPrices = data[:, 1:3]

with open(_dir + 'AAPL_put_price.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # skip the header row

    # initialize the arrays for strike prices and option values using NumPy
    data = np.array(list(reader), dtype=float)
    AAPLstrikes = data[:, 0]
    AAPLputPrices = data[:, 1:3]

def second_derivative(price, strike):
    firstD = np.diff(price, axis=0) / np.diff(strike).reshape(-1,1)
    firstStep = (strike[:-1] + strike[1:]) / 2
    density = np.diff(firstD, axis=0) / np.diff(firstStep).reshape(-1,1)
    newstrike = strike[1:-1]
    density = density[:, ~np.isnan(density).all(axis=0)]
    return density, newstrike

def pdf_approx(spotPrice, stepSize, callPrice, putPrice, strike):
    n, m = callPrice.shape
    startP = np.zeros((m,))
    endC = np.zeros((m,))
    strikeLength = np.zeros((m,))
    combineDensity = np.zeros((n-2 ,m))
    combineStrike = np.zeros((n-2 ,m))
    
    mid = np.argmax(strike >= spotPrice)
    densityC, newstrikeC = second_derivative(callPrice, strike)
    densityP, newstrikeP = second_derivative(putPrice, strike)
    
    for i in range(m):

        put_col = densityP[:, i]
        call_col = densityC[:, i]
        
        # Find the portion of the option implied density calculated from put options
        zero_indices = np.where(densityP[:mid, i] == 0)[0]
        if len(zero_indices) > 0:
            start = zero_indices[-1] + 1
        else:
            start = 0
        
        # Find the portion of the option implied density calculated from call options
        end = mid + np.argmax(call_col[mid:] == 0)
        if end == mid:
            end = n
            
        slenght = end - start

        startP[i] = start
        endC[i] = end
        strikeLength[i] = slenght
    
        # copy those data and store in the output array
        combineDensity[:mid-start, i] = put_col[start:mid]
        combineDensity[mid-start:end-start, i] = call_col[mid:end]
        combineStrike[:end-start, i] = newstrikeC[start:end]
    
    # normalize density integral (sum * step) to 1
    # ...
    
    return combineStrike, combineDensity, strikeLength


def create_inv_cdf(density, position):
    # compute step size
    step_size = position[1] - position[0]

    # create evenly-spaced distribution
    density = np.append(density, 0)
    num_points = len(density)
    x_vals = np.linspace(position[0] -0.5 * step_size, position[0] + (num_points + 0.5) * step_size, num_points)

    # interpolate density function
    f = interp1d(x_vals, density)


    # create evenly-spaced distribution for CDF
    num_cdf_points = 1000
    cdf_vals = np.linspace(position[0] -0.5 * step_size, position[0] + (num_points + 0.5) * step_size, num_cdf_points)

    # print(cdf_vals)

    # compute CDF
    cdf = np.zeros(num_cdf_points)
    for i in range(num_cdf_points):
        cdf[i] = np.trapz(f(cdf_vals[cdf_vals <= cdf_vals[i]]), cdf_vals[cdf_vals <= cdf_vals[i]])

    # print(cdf)

    # compute inverse CDF
    inv_cdf = interp1d(cdf, cdf_vals)

    #print(inv_cdf.x)

    # return desired inverse CDF value
    return inv_cdf

AMZNSpot = 95.0
AMZNStep = 5.0   # mkt practice

AAPLSpot = 151.07
AAPLStep = 5.0   # mkt practice

AMZNcombineStrike, AMZNcombineDensity, AMZNstrikeLength = pdf_approx(AMZNSpot, AMZNStep, AMZNcallPrices, AMZNputPrices, AMZNstrikes)
AAPLcombineStrike, AAPLcombineDensity, AAPLstrikeLength = pdf_approx(AAPLSpot, AAPLStep, AAPLcallPrices, AAPLputPrices, AAPLstrikes)
AMZNstrikeLength = AMZNstrikeLength.astype(int)
AAPLstrikeLength = AAPLstrikeLength.astype(int)

# check sum to 1
AMZNcombineDensity.sum(axis=0)
AAPLcombineDensity.sum(axis=0)

# show pdf
fig, (ax1, ax2) = pl.subplots(nrows=2, ncols=1, figsize=(8, 8))
[ax1.bar(AMZNcombineStrike[:,i]+0.25*i, AMZNcombineDensity[:,i], width = 0.25) for i in range(2)]
[ax2.bar(AAPLcombineStrike[:,i][AAPLcombineStrike[:,i] > 0]+0.25*i, AAPLcombineDensity[:,i][AAPLcombineStrike[:,i] > 0], width = 0.25) for i in range(AAPLcombineStrike.shape[1])]
pl.show()

fig, (ax1, ax2) = pl.subplots(nrows=2, ncols=1, figsize=(8, 8))
[ax1.bar(AMZNcombineStrike[:,i]+0.25*i, AMZNcombineDensity[:,i], width = 0.25) for i in range(AMZNcombineStrike.shape[1])]
[ax2.bar(AAPLcombineStrike[:,i][AAPLcombineStrike[:,i] > 0]+0.25*i, AAPLcombineDensity[:,i][AAPLcombineStrike[:,i] > 0], width = 0.25) for i in range(AAPLcombineStrike.shape[1])]
pl.show()

# # check
# x1.shape
# x2.shape
# y1.shape
# y2.shape
# x1_pdf.shape
# x2_pdf.shape
# y1_pdf.shape
# y2_pdf.shape


# these are the inverse cdf functions
AMZN_inv_cdf = create_inv_cdf(AMZNcombineDensity[:AMZNstrikeLength[1],1], AMZNcombineStrike[:AMZNstrikeLength[1],1])
AAPL_inv_cdf = create_inv_cdf(AAPLcombineDensity[:AAPLstrikeLength[1],1], AAPLcombineStrike[:AAPLstrikeLength[1],1])

random_numbers = np.random.rand(100)
AMZN_sample = AMZN_inv_cdf(random_numbers)
AAPL_sample = AAPL_inv_cdf(random_numbers)

fig, (ax1, ax2) = pl.subplots(nrows=2, ncols=1, figsize=(8, 8))

# Plot the density function
AMZN_sorted_indices = np.argsort(AMZN_sample)
ax1.plot(AMZN_sample[AMZN_sorted_indices])
ax1.set_title('AMZN sample')

# Plot the two sets of random points
AAPL_sorted_indices = np.argsort(AAPL_sample)
ax2.plot(AAPL_sample[AAPL_sorted_indices])
ax2.set_title('AAPL sample')

pl.show()
