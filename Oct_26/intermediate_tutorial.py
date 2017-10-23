

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
import sys



# string manipulating and formatting in python 3!

# strip and split

# printing with {} formatting

# -----------------------------------------------------------------------------
# Section 2: Data manipulation and slicing - numpy and pandas
# -----------------------------------------------------------------------------

# rehashing past stuff

# import a pandas dataframe

# make your own pandas dataframe

# -----------------------------------------------------------------------------
# Section 3: curve fitting and statistics with scipy
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Section 4: plotting with matplotlib
# -----------------------------------------------------------------------------
%matplotlib
# figsize (inches)?,dpi (dots per inch), legendbyplot
# plt annotate, frameon

# give a decent example of a fancy plotting

# show subplots, sharing axes

# heatmap style, imshow and pcolor (look at interpolation)

# animation


# -----------------------------------------------------------------------------
# Section 5 - miscellaneous stuff
# -----------------------------------------------------------------------------
