

# ------------------------------------------------------------------------------
# Section 1: file input/output, string formatting
#------------------------------------------------------------------------------

# basic reading and writing
myfile = open('input_file.txt','r') # r stands for write
first_line = myfile.readline()
second_line = myfile.readline()
myfile.close()

# get whole text as a string
myfile = open('input_file.txt','r')
wholetext_str = myfile.read()
myfile.close()

# get line separated list
myfile = open('input_file.txt','r')
wholetext_list = myfile.readlines()
myfile.close()

# different way of iterating through
myfile = open('input_file.txt','r')
for line in myfile:
    print(line)
myfile.close()

# using "with": safer and easier to read than open and close
with open('input_file.txt','r') as myfile:
    first_line = myfile.readline()
    second_line = myfile.readline()

# file OUTPUT
with open('output_file.txt','w') as outfile: # notice the w
    for i in range(10):
        outfile.write(str(i))

with open('output_file_newline.txt','w') as outfile:
    for i in range(10):
        outfile.write(str(i) + '\n')

# numpy options
import numpy as np
sample_data = np.loadtxt('dataset1.csv', delimiter=',', skiprows=1)

np.savetxt('output_data_spaceformatted.csv',
            sample_data,
            fmt='%.4f',
            delimiter=' ')
np.savetxt('output_data_commaformatted.csv',
            sample_data,
            fmt='%.4e',
            delimiter=',')

# pickle
import pickle
pickle.dump(sample_data,open('storage.pkl','wb'))
%reset
import pickle
sample_reloaded = pickle.load(open('storage.pkl','rb'))


# sys argvals
import sys # access input variables with sys.argv
sys.path

# string manipulating and formatting in python 3!
def add_and_print(num1,num2):
    print('The sum of ' + str(num1) + ' and ' + str(num2) + ' is ' + str(num1 + num2))

def add_and_print_formatted(num1,num2):
    print('The sum of {:d} and {:d} is {:d}'.format(num1, num2, num1 + num2))

'The sum of {:d} and {:d} is {:d}'.format(3,5,8)
myname = 'kevin'
print('my name is {:s}'.format(myname))
print('my name is {}'.format(myname))
print('my name is {}'.format(2))

myfloat = 43.321864
print('truncated number:  {:.3f}'.format(myfloat))
print('truncated and buffered number:{:10.3f}'.format(myfloat))
print('{:5d}{:5d}{:5d}\n{:5d}{:5d}{:5d}'.format(200,3,17,4000,18,2))
# this formatting can be used in numpy savetxt as well

# strip and split
'Hello, my name is'.split(' ')
'Firstline\nsecondline\nthirdline'.splitlines()
'Hello, my name is     '.strip(' ')

# A LOT more functionality exists to manipulate and format strings - I only covered the basics

# -----------------------------------------------------------------------------
# Section 2: Data manipulation and slicing - numpy and pandas
# -----------------------------------------------------------------------------
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

# pandas
import pandas as pd
my_df = pd.read_csv('pd_input.csv')
my_df['concentration']
my_df['control']
my_df.loc[3,'concentration':'set1']
my_df.iloc[2,:]

# dealing with incomplete data
a = np.array([[1,np.NaN,3],[4,5,67]])
np.mean(a)    # can't deal with nan
my_df.iloc[2,3] = np.NaN
my_df.mean()

my_df.dropna('index')
my_df.dropna('columns') # inplace=True for changing actual dataframe

my_df.fillna(0) # still not inplace, so doesn't change
my_df.fillna(method='pad')

# -----------------------------------------------------------------------------
# Section 3: curve fitting and statistics with scipy
# -----------------------------------------------------------------------------
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

substrate_C = [0,1,2,5,8,12,30,50] # mM
Vo = [0,11.1,25.4,44.8,54.5,58.2,72.0,60.1]
plt.scatter(substrate_C,Vo)
plt.show()
def mich_menton(x,Km,Vmax):
    return x * Vmax / (x + Km)

myfit, covar = curve_fit(mich_menton,substrate_C,Vo,p0 = [3,60])
fit_error = np.sqrt(np.diag(covar))

xdata_fit = np.linspace(0,50,1000)
ydata_fit = mich_menton(xdata_fit,myfit[0],myfit[1]) # fancier: mich_menton(xdata_fit,*myfit)

plt.scatter(substrate_C,Vo)
plt.plot(xdata_fit,ydata_fit,c='r')
plt.show()


# statistical tests
from scipy.stats  import ttest_ind

set1 = np.random.normal(10,2,60)
set2 = np.random.normal(11,2,60)

plt.hist(set1,normed=True)
plt.hist(set2,normed=True)
plt.show()

Tval,pval = ttest_ind(set1,set2)

# -----------------------------------------------------------------------------
# Section 4: plotting with matplotlib
# -----------------------------------------------------------------------------
%matplotlib

# getting started - initiating a figure, and using figure handles
plt.figure()
plt.figure(2) # make a second figure
plt.figure(1) # switch back to first figure
plt.scatter([1,2,4],[3,2,1])
plt.figure(2) # switch back to second figure
plt.plot([1,2,3,4],[1,2,3,4])
plt.figure(1) # back to first
plt.xticks([1,3])

# selecting size of figure on startup
plt.figure?
plt.figure(1,figsize=(12,8),dpi=80) #,edgecolor='r',facecolor='b')


# OK, we're gonna go from simple to complicated. First lets generate some data
xdata = np.arange(5,100,0.1)
yfit =  np.log(xdata)
ydata_noisy = yfit + np.random.rand(yfit.size) - 0.5

#basics
plt.scatter(xdata,ydata_noisy) # basic scatter
plt.plot(xdata,yfit)
plt.plot(xdata,yfit,c='k')
plt.clf()

plt.scatter(xdata,ydata_noisy)
plt.plot(xdata,yfit,c='k',linestyle='--',linewidth=5)

# setting a line/scatter to a variable
fit_plot,  = plt.plot(xdata,yfit)
fit_plot.set_color('k')
fit_plot.set_linestyle(':')
fit_plot.set_linewidth(3)

# setting marker type
plt.scatter(xdata,ydata_noisy,marker='+')

# setting transparency
plt.scatter(xdata,ydata_noisy,alpha=0.5)

# setting marker size
random_sizes = np.random.rand(xdata.size)* 40

plt.scatter(xdata,ydata_noisy,s=60)   # fixed
plt.scatter(xdata,ydata_noisy,s=random_sizes)
plt.scatter(xdata,ydata_noisy,s=xdata,alpha=0.5)

# colors
plt.scatter(xdata,ydata_noisy,c='r')    # fixed
plt.scatter(xdata,ydata_noisy,c='r',edgecolor='b')

plt.scatter(xdata,ydata_noisy,c=random_sizes)
plt.scatter(xdata,ydata_noisy,c=xdata)
plt.scatter(xdata,ydata_noisy,c=np.abs(ydata_noisy - yfit))

# colormaps - https://matplotlib.org/examples/color/colormaps_reference.html
plt.scatter(xdata,ydata_noisy,c=np.abs(ydata_noisy - yfit),cmap='binary')
plt.scatter(xdata,ydata_noisy,c=np.abs(ydata_noisy - yfit),cmap='seismic')
plt.scatter(xdata,ydata_noisy,c=np.abs(ydata_noisy - yfit),cmap='flag')

plt.scatter(xdata,ydata_noisy,
            c=np.abs(ydata_noisy - yfit),
            cmap='seismic',
            vmin = -0.5, vmax = 0.5) # can change normalization method

# peripheral things
plt.xlabel('Independent variable (some unit)')
plt.ylabel('Dependent variable (some other unit)')
plt.colorbar(label='distance (unit)')  # check options
plt.xlim(-10,110)
plt.ylim(0.8,5.5)
plt.xticks([0,10,30,87])
plt.yticks([1,2,3,4,5],['first tick','second','third','fourth','last'],fontsize=30)

# legend
plt.scatter(xdata,ydata_noisy) # can change normalization method
plt.plot(xdata,yfit,c='k',linestyle='--',linewidth=3)
plt.legend(['actual data','fit'])
plt.legend(['actual data','fit'],loc='lower right')

# alternate legend
plt.scatter(xdata,ydata_noisy,label='data')
plt.plot(xdata,yfit,c='k',linestyle='--',linewidth=3,label='fit')
plt.legend(loc='center right')

# plt annotate
plt.scatter(xdata,ydata_noisy,label='data')
plt.plot(xdata,yfit,c='k',linestyle='--',linewidth=3,label='fit')
plt.annotate('Text at this point',xy=(50,5.0),xytext=(55,5.3),arrowprops=dict(arrowstyle="->"))

# axes and subplots
fig = plt.figure()
ax  = plt.gca()  # "Get Current Axis"  gcf() is "Get Current Figure"
plt.scatter(xdata,ydata_noisy,label='data')
plt.plot(xdata,yfit,c='k',linestyle='--',linewidth=3,label='fit')
ax.set_xscale('log')

# second plot on same axis
ax2 = ax.twinx()
ax2.plot(np.arange(100),np.arange(100),c='r',linewidth=4)
ax2.set_ylim(20,80)
ax.set_ylim(20,80)
ax2.set_ylabel('right side data')

# second axis internal
fig = plt.figure()
main_ax = fig.add_axes([0.1,0.1,0.9,0.9]) #[left,bottom,width,height]
internal_ax = fig.add_axes([0.15,0.65,0.3,0.3])

main_ax.scatter(xdata,ydata_noisy,label='data')
main_ax.plot(xdata,yfit,c='k',linestyle='--',linewidth=3,label='fit')
internal_ax.scatter(xdata,ydata_noisy,label='data')
internal_ax.plot(xdata,yfit,c='k',linestyle='--',linewidth=3,label='fit')
main_ax.set_xscale('log')
main_ax.set_xlabel('main x axis')
internal_ax.set_xlabel('internal x axis')

# subplots
fig1, axarray_2by2   = plt.subplots(2,2)
axarray_2by2[0,1].plot([1,2,3],[4,5,6])
axarray_2by2[1,1].plot([1,2,3],[6,4,2])

fig2, axarray_sharex = plt.subplots(2,1,sharex=True)

# heatmaps, contour plots
z = np.empty((20,20))
for i in range(z.shape[0]):
    for j in range(z.shape[1]):
        z[i,j] = i + j

plt.imshow(z,interpolation='none')
plt.imshow(z,interpolation='bilinear')
plt.contourf(z); plt.colorbar()


# animation
from matplotlib import animation

anim_data = np.loadtxt('./animation_data.csv',delimiter=',')
anim_data[:,1] = anim_data[:,1] / np.max(anim_data[:,1])
adp_points = [300,500,700]

anim_fig = plt.figure()
plt.xlim(-10,1000)
plt.ylim(0.5,1.1)
plt.xlabel('time (s)')
plt.ylabel('Relative fluorescence intensity')
plt.plot([100,100],[0.5,1],'k--',alpha=0.3)
plt.plot([300,300],[0.5,1],'k--',alpha=0.3)
plt.plot([500,500],[0.5,1],'k--',alpha=0.3)
plt.plot([700,700],[0.5,1],'k--',alpha=0.3)

data = plt.scatter([],[],5)

def anim_init():
    pass

def anim_update(i):
    data.set_offsets(anim_data[:i,:])

ani = animation.FuncAnimation(anim_fig,anim_update,frames=anim_data.shape[0],interval=15)
