
''' Things you will need for this tutorial:

    -an ipython terminal
    -numpy
    -matplotlib
'''
# why ipython
# -function?
# -debugging
# -better autocomplete
# -magic
# -timing

# -----------------------------------------------------------------------------
# basic syntax: data types, lists and dictionaries
# -----------------------------------------------------------------------------

# basic data types
type(3)
type(3.0)
type('three')
type(True)

# variable assignment and operations
num = 3
num2 = 5
num + num2
num / num2
num * num2
num ** num2    # exponents
num2 % num     # modulus

# strings
string = 'hello'
string2 = 'Jason'
string + string2
string + " " + string2    # single or double quotes both work

# indexing lists
b = [1, 2, 3]
c = [num, 5, string2]
c[0]
d = c[0]
c[0] = 25

# other list operations
c
c.append(15)
c
len(c)


# multidimensional lists
multi_list = [[1, 2, 3], ['hello']]
multi_list[1]
multi_list[0][1]
multi_list[1][1]  # error

# slicing lists
longer_list = [[100, 3, 2, 18, 25], [5, 4, 22, 55, 31]]
longer_list[0][:]
longer_list[1][2:4]
longer_list[0][1:]
longer_list[0][:3]
longer_list[:][2]      # error

# slicing strings
string2
string2[0]
string2[3:]

# dictionaries
mydict = {'name': 'kevin', 'age': 25}  # or mydict = dict()
mydict['name']
mydict['status'] = 'student'
mydict[15] = 14
mydict[name]  # error

control   = [1,2,3,4,5]
dataset1  = [2,4,6,8,10]
dataset2  = [3,6,9,12,15]
data_dict = {'control data': control, 'set 1':dataset1, 'set 2':dataset2}
data_dict['set 1']


# tuples
mytuple = (3, 'hello', 7.5)
(int_fromtup, string_fromtup, float_fromtup) = mytuple

# -----------------------------------------------------------------------------
# loops and ranges
# -----------------------------------------------------------------------------

# python separates blocks of code using indentation!

for i in range(10):  # don't forget colon
    print(i)

# range starts with 0, unless you specify. Like list operations, the first
# number is included, but the last one IS NOT
for i in range(1, 11):
    print(i)

# Can specify step
for i in range(1,20,2):
    print(i)

numbers = [3, 2, 6, 5, 1]
for i in numbers:
    print(i)

# while loops
i = 0
while i < 10:
    print(i)
    i = i + 1

# -----------------------------------------------------------------------------
# booleans and conditionals
# -----------------------------------------------------------------------------
a = True
b = False

# boolean comparisions
a and b
a or  b
a &   b   # same as and for this use case
a |   b   # same as or  for this use case

num1 = 5
num2 = 3

# boolean operations
num1 == num2
num1 >  num2
num1 <  num2
num1 != num2

# greater/equals
num3 = 5
num1 > num3
num1 >= num3

# conditionals and control flow
a = True
if a:   # don't forget colon
    print('true')

a = False
if a:
    print('true')

# else statement
if a:
    print('true')
else:
    print('false')

a = 6
if a < 10:
    print('a is less than 10')

if a < 10 and a > 4:
    print('a is less than 10 and greater than 4')
if a < 10 and a > 7:
    print('a is less than 10 and greater than 7')


if a < 3:
    print('a is less than 3')
elif a >= 3 and a <= 10:
    print('a is between 3 and 10 (inclusive)')
else:
    print('a is greater than 10')

# can't do operations on a list
b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
b > 5  # error

# can check contents of a list if boolean
bool_list = [True, False, True]
alltrue   = [True, True,  True]
any(bool_list)
all(bool_list)
any(alltrue)
all(alltrue)

# boolean as numbers
True + False
True + True

# example - checking values in standard curve
vals     = [0.8,1.5,0.723,0.05,0.4]
low_lim  = 0.1
high_lim = 1.0

inrange = []
for absorbance in vals:
    if absorbance < low_lim or absorbance > high_lim:
        inrange.append(False)
    else:
        inrange.append(True)

if all(inrange):
    print('Every value falls within standard curve')
else:
    print('Some values fall out of standard curve')

# ------------------------------------------------------------------------------
# functions
# ------------------------------------------------------------------------------

def average(num1, num2):  # don't forget colon
    num_sum = num1 + num2
    num_average = num_sum / 2
    return num_average

myavg = average(5, 6)

# better
def average(num1, num2):
    return (num1 + num2) / 2

# note: not all functions return stuff, in which case no return statement
# is necessary and the function just ends when the indentation ends
def printme(print_string):
    print(print_string)

printme('This function has no return value')

# examples --------------------------------------------------------------------
def calc_Kav(V_elute,Vo,Vt):
    return (V_elute - Vo) / (Vt - Vo)

def percent_change(data,original):
    return (data - original) / original

# keyword arguments
def percent_change(data,original,return_type='fraction',abs_val=False):
    if return_type == 'fraction':
        mult_factor = 1
    elif return_type == 'percentage':
        mult_factor = 100

    if abs_val:
        return mult_factor * abs(data - original) / original
    else:
        return mult_factor * (data - original) / original


# -----------------------------------------------------------------------------
# modules and importing
# -----------------------------------------------------------------------------

# import entire module
import time
time.localtime()
time.time()

# import one thing from a module
localtime()  # error
from time import localtime  # notice no time. prefix needed
localtime()

# import and rename
import numpy as np
numpy.pi   # error
np.pi


# -----------------------------------------------------------------------------
# numpy
# -----------------------------------------------------------------------------
import numpy as np

# index slicing
longer_list = [[100, 3, 2, 18, 25], [5, 4, 22, 55, 31]]
longer_list[:][2]   # error
np_list = np.array(longer_list)
np_list[:][2]  # error
np_list[:, 2]

# combination of arrays
a = [1, 2, 3]
b = [4, 5, 6]
a + b
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
a + b
a * b
a ** 2

# built in functions
np.mean(np_list)
np.mean(np_list, axis=0)
np.mean(np_list, axis=1)
np_list.shape
np_list.shape[0]
np.mean(np_list).shape

# boolean slicing: more advanced, super useful
np_list
boolean_list = np_list < 5
np_list.shape
boolean_list.shape
np_list[boolean_list]

# finding the indices that satisfy a requirement
np_list = np.array([0,1,2,4,5,6,7,8])
np.where(np_list > 4)
# -----------------------------------------------------------------------------
# introduction to plotting
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt  # plt is by convention.

xdata = np.arange(0, 2 * np.pi, 0.1)
ydata = np.sin(xdata)

# line plot
plt.plot(xdata, ydata)
plt.show()

# scatter plot
plt.scatter(xdata, ydata)
plt.show()

# specify color
plt.scatter(xdata, ydata, c='r')
plt.show()

# specify marker
plt.scatter(xdata, ydata, c='r', marker='+')
plt.show()

# multiple data sets
ydata2 = np.cos(xdata) + 1
plt.scatter(xdata, ydata, c='r', marker='+')
plt.scatter(xdata, ydata2, c='b', marker='*')
plt.show()

# labeling
plt.scatter(xdata, ydata, c='r', marker='+')
plt.scatter(xdata, ydata2, c='b', marker='*')
plt.title('Example plot title')
plt.xlabel('X coordinate (radians)')
plt.ylabel('Dependent variable')
plt.legend(['Sine dataset', 'Cosine dataset'])  # entered as list of legend labels
plt.show()

# bar plot
x = [1, 2, 3, 4]
y = [10, 4, 3, 6]
plt.bar(x, y)
plt.show()

# xtick labels
plt.bar(x, y)
plt.xticks(x, ('First data', 'second', 'third', 'fourth'))
plt.show()

# errorbars
errors = [0.5, 0.5, 1, 2]
plt.bar(x, y, yerr=errors)
plt.xticks(x, ('First data', 'second', 'third', 'fourth'))
plt.show()


# ---------------------------------------------------------------------
# data analysis
# ---------------------------------------------------------------------

a = np.loadtxt('/home/kevin/git_repos/python_tutorial/fluorescence_data.csv', delimiter=',', skiprows=1)
a.shape
wavelengths = a[:, 0]
data = a[:, 1:]
wavelengths
data
data[:, 0]
plt.plot(wavelengths, data)
plt.show()

# check max wavelength
np.argmax(data, axis=0)
wavelengths[np.argmax(data, axis=0)]

# isolate peak data
baseline_peaks = data[60, 0:3]
data_peaks = data[60, 3:]

# take control average, and normalize data set
baseline_average = np.mean(baseline_peaks)
normalized_peaks = (data_peaks - baseline_average) / baseline_average

avg_pt1uM = np.mean(normalized_peaks[0:3]);  std_pt1uM = np.std(normalized_peaks[0:3])
avg_1uM = np.mean(normalized_peaks[3:6]);    std_1uM = np.std(normalized_peaks[3:6])
avg_10uM = np.mean(normalized_peaks[6:9]);   std_10uM = np.std(normalized_peaks[6:9])
avg_100uM = np.mean(normalized_peaks[9:12]); std_100uM = np.std(normalized_peaks[9:12])
avg_500uM = np.mean(normalized_peaks[12:]);  std_500uM = np.std(normalized_peaks[12:])

concentrations = [0.1, 1, 10, 100, 500]

plt.errorbar(concentrations,
             [avg_pt1uM, avg_1uM, avg_10uM, avg_100uM, avg_500uM],
             yerr=[std_pt1uM, std_1uM, std_10uM, std_100uM, std_500uM],
             fmt='o', capsize=2)
plt.show()


def process_fluorescence_data(infile, concentrations, n_trials, peak_max):
    '''
    Processes fluorescence data csv. Format of csv should be first column=wavelength, then
    groups of n_trial columns for each concentration, with the 0-concentration baseline
    being the first n_trials columns. The first row of the csv will be skipped

    Returns normalized average and standard deviation at each concentration
    '''

    # load file
    initial_data = np.loadtxt(infile, delimiter=',', skiprows=1)

    # split input into wavelengths, baseline data, sample data
    wavelengths = initial_data[:, 0]
    baseline_data = initial_data[:, 1:1 + n_trials]
    sample_data = initial_data[:, 1 + n_trials:]

    # get index of peak location
    peak_index = np.where(wavelengths == peak_max)[0]

    # calculate baseline average
    baseline_peak = np.mean(baseline_data[peak_index, :])

    n_samples = len(concentrations)
    averages = []
    stds = []

    count = 0
    while count < n_samples * n_trials:
        normalized_data = (sample_data[peak_index, count:count + n_trials] - baseline_peak) / baseline_peak
        averages.append(np.mean(normalized_data))
        stds.append(np.std(normalized_data))
        count += n_trials
    return averages, stds


# cleaning up and making this reproducible
n_trials = 3
concentrations = [0.1, 1, 10, 100, 500]
peak_max = 360
avg1, std1 = process_fluorescence_data('dataset1.csv', concentrations, n_trials, peak_max)
avg2, std2 = process_fluorescence_data('dataset2.csv', concentrations, n_trials, peak_max)

plt.errorbar(concentrations, avg1, yerr=std1, c='r', fmt='o', capsize=2)
plt.errorbar(concentrations, avg2, yerr=std2, c='b', fmt='o', capsize=2)
plt.xlabel('Concentration (uM)')
plt.ylabel('Normalized intensity')
plt.legend(['Data set 1', 'Data set 2'])
plt.show()

# Resources
# - youtube tutorials
# - jupyter
