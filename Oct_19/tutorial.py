
''' Things you will need for this tutorial:

    -an ipython terminal
    -numpy
    -mdtraj

'''
# why ipython
# function?
# debugging
# better autocomplete
# magic
# -   timing

# -----------------------------------------------------------------------------
# basic syntax: lists and dictionaries
# -----------------------------------------------------------------------------
num = 3;
mystring  = 'hello'
num2 = 5
num + num2
num / num2
num * num2
num ** num2    # exponents


# indexing lists
b = [1,2,3]
c = [num,5,mystring]
d = c[0]
c[0] = 25

# multidimensional lists
multi_list = [ [1, 2, 3] , ['hello']  ]
multi_list[1]
multi_list[0][1]
multi_list[1][1]  # error

# slicing lists
longer_list = [ [100,3,2,18,25], [5,4,22,55,31] ]
longer_list[0][:]
longer_list[1][2:4]
longer_list[0][1:]
longer_list[0][:3]
longer_list[:][2]      # error

# dictionaries
mydict = { 'name':'kevin', 'age':25  } # or mydict = dict()
mydict['name']
mydict['status'] = 'student'
mydict[15] = 14
mydict[name] # error

# -----------------------------------------------------------------------------
# loops and ranges
# -----------------------------------------------------------------------------
for i in range(10): # don't forget colon
    print(i)

for i in range(1,11):
    print(i)

numbers = [3,2,6,5,1]
for i in numbers:
    print(i)


# list comprehension
mylist = [i for i in range(10)]
mylist = [i for i in range(10) if i % 2  == 0]


# -----------------------------------------------------------------------------
# conditionals
# -----------------------------------------------------------------------------
a = True
if a:   # don't forget colon
    print('true')

a = False
if a:
    print('true')

if a:
    print('true')
else:
    print('false')



a = 5
a < 10

if a < 10:
    print('true')

b = [1,2,3,4,5,6,7,8,9,10]
b > 5
b >= 5
b > 5 and b < 8
any(b > 5)
all(b > 5)

# ------------------------------------------------------------------------------
# functions
# ------------------------------------------------------------------------------

def average(num1,num2): # don't forget colon
    num_sum = num1 + num2
    num_average = num_sum / 2
    return num_average

myavg = average(5,6)

# better
def average(num1,num2):
    return (num1 + num2) / 2

# note: not all functions return stuff, in which case no return statement
# is necessary and the function just ends when the indentation ends


# -----------------------------------------------------------------------------
# numpy
# -----------------------------------------------------------------------------
import numpy as np

# index slicing
longer_list = [ [100,3,2,18,25], [5,4,22,55,31]]
longer_list[:][2]   # error
np_list = np.array(longer_list)
np_list[:][2] # error
np_list[:,2]

# combination of arrays
a = [1,2,3]; b = [4,5,6]
a + b
a = np.array([1,2,3]); b = np.array([4,5,6])
a + b
a * b
a ** 2

# built in functions
np.mean(np_list)
np.mean(np_list,axis=0)
np.mean(np_list,axis=1)
np_list.shape
np_list.shape[0]
np.mean(np_list).shape

# boolean slicing: more advanced, super useful
np_list
boolean_list = np_list < 5
np_list.shape;
boolean_list.shape
np_list[boolean_list]

# finding the indices that satisfy a requirement
np.where(np_list < 5)

'''in general, try to use numpy without looping as much as possible, it's built
   for large array-scale actions, rather than lots of individual procedures
'''
my_array = np.random.rand(1000,1000)* 10       # 1 million numbers ranging from 0 to 10

# count the number of instances that this array is less than 2:
# bad
count = 0
for i in range(my_array.shape[0]):
    for j in range(my_array.shape[1]):
        if my_array[i,j] < 2:
            count += 1

# better
boolean_lessthan2 = my_array < 2
count = np.sum(boolean_lessthan2)

# or on one line
count = np.sum(my_array < 2)


# -----------------------------------------------------------------------------
# example 1: topology manipulation
# -----------------------------------------------------------------------------
''' The problem: My new topology for a lipid has a deleted hydrogen. This throws
off the numbering in the topology file. Easy enough to change in the [atoms]
directory, but bonds,angles,dihedrals would take days, and you're likely to make
mistakes

The hydrogen being deleted is number 21 in the sequence of 248 atoms
'''

# first, we set up a dictionary, in which the keys are the old indices,
# and the values are the new indices


number_mapping_dict = {}
# first 20 don't change
for i in range(1,21):                                          # remember, does not go to last number
    number_mapping_dict[i] = i                                 # 1:1, 2:2
# we'll set the one we're deleting to 0, will see why momentarily
number_mapping_dict[21] = 0                                    # 21:0
# for rest of them, the new index is one less than the old index
for i in range(22,249):
    number_mapping_dict[i] = i - 1                             # 22:21 , 23:22, ....

in_angles =  np.loadtxt('example_angles.txt',dtype=int)        # input, 'r' is for read
out_angles = np.copy(in_angles)                                # we'll modify this one for output

# get data dimensions
n_lines = in_angles.shape[0]
columns = [0,1,2]                                              # We don't want to modify column 4 (index 3)

write = np.ones(n_lines)                                       # ones correspond to True in np, zeros to false

# main loop, change numbers
for i in range(n_lines):                                       # go over each line
    for j in columns:                                          # go over selected columns in line
        out_angles[i,j] = number_mapping_dict[in_angles[i,j]]  # Substitute numbers using mapping dictionary
        if out_angles[i,j] == 0:
            write[i] = 0

np.savetxt('example_out.txt',out_angles[np.where(write > 0)[0],:],delimiter=" ",fmt='%5i')
# -----------------------------------------------------------------------------
# making things nice and reusable
# -----------------------------------------------------------------------------
''' Ok so that was nice. But we also have to do this for bonds, pairs, dihedrals
ect. We could just copy and paste that whole block of code and change a few things
each time, but that's inefficient. It's also not helpful if you come back to the same
problem in a few months and want to reuse your code instead of writing it all new.
SO....., lets clean up that above stuff and turn it into a function


Functions need inputs - what were the things I needed to know to change my files?
    - the mapping dictionary
    - the specific columns to write out
    - the input file to read
    - the output file to print
'''

def replace_column_indices(in_file,out_file,index_map,columns):
    in_angles = np.loadtxt(in_file,dtype=int)
    out_angles = in_angles

    n_lines = in_angles.shape[0]
    #columns = [1,2,3]     # now we don't need to manually set columns
    write = np.ones(n_lines)

    for i in range(n_lines):
        for j in columns:
            out_angles[i,j] = index_map[in_angles[i,j]]
            if out_angles[i,j] == 0:
                write[i] = 0
    np.savetxt(out_file, out_angles[np.where(write > 0)[0],:],delimiter=" ",fmt='%5i')

# Great! Now we can use this function for all of our sections!
replace_column_indices('example_bonds.txt'    , 'pairs_out.txt'    , number_mapping_dict, [1,2])
replace_column_indices('example_pairs.txt'    , 'bonds_out.txt'    , number_mapping_dict, [1,2])
replace_column_indices('example_angles.txt'   , 'angles_out.txt'   , number_mapping_dict, [1,2,3])
replace_column_indices('example_dihedrals.txt', 'dihedrals_out.txt', number_mapping_dict, [1,2,3,4])

''' GO BACK AND DOCUMENT WHAT YOU DID'''

# ------------------------------------------------------------------------------
# mdtraj
# ------------------------------------------------------------------------------
import mdtraj as md

traj = md.load_xtc('example.xtc',top='example.gro')
topology = traj.topology
traj.xyz
traj.xyz.shape

# selection
topology.select("protein")
protein_indices = topology.select('protein')
protein_coords  = traj.xyz[:,protein_indices,:]

# trajectory atom properties
topology.residue(0)
# getting indices of a specific residue
res_zero_indices = [atom.index for atom in topology.residue(0).atoms]
res_one_indices =  [atom.index for atom in topology.residue(1).atoms]



# -----------------------------------------------------------------------------
# example 2 - geometric analyses
# -----------------------------------------------------------------------------

''' Shivangi has a trajectory containing proteins, lipids, water, and ions.
She wants to run analysis on a subset of the system that lies with in a z-slice
(ie z > 1 nm, z < 3 nm). The problem is, this subset is dynamic - it changes over time
as things diffuse in and out of the z block. An additional complication is that
molecules can be partially in the block: if only part of the molecule is in this
z slab, we still need to run the analysis on the WHOLE molecule'''

z_min = 1
z_max = 3

z_coordinates = traj.xyz[:,:,2]
z_coordinates.shape

# first lets make a boolean array of what's in and out of the slab
in_z_slab = (z_coordinates > z_min) & (z_coordinates < z_max)
in_z_slab.shape

# Simplistic analysis will be just count number of atoms selected
def DO_ANALYSIS(trajectory):
    return trajectory.shape[0]

# if we ignore the whole molecule problem
result = np.zeros(traj.n_frames)
for i in range(traj.n_frames):
    analysis_indices = in_z_slab[i,:]
    result[i] = DO_ANALYSIS(traj.xyz[i,analysis_indices,:])

# now lets implemement the whole molecule issue with a function
def make_molecules_whole(indices,top):
    '''Indices is an n_frames * n_atoms boolean array, top is the mdtraj topology
        The goal is to return an array similar to indices, but for every 1 in
        indices, all other atoms in the residue of that index needs to also be 1
    '''
    out_indices = indices                                      # initialize output
    for res in top.residues:                                   # iterate over all residues
        residue_indices = [ atom.index for atom in res.atoms ] # get residue indices
        for frame in range(indices.shape[0]):                  # iterate over frames
            if any(indices[frame,residue_indices]):
                out_indices[frame,residue_indices] = True      # if any are true, set all true
    return out_indices


# lets try again
result_corrected = np.zeros(traj.n_frames)
in_z_slab_corrected = make_molecules_whole(in_z_slab,topology) # correction
for i in range(traj.n_frames):
    analysis_indices = in_z_slab_corrected[i,:]
    result_corrected[i] = DO_ANALYSIS(traj.xyz[i,analysis_indices,:])

# plotting
import matplotlib.pyplot as plt
plt.plot(result)
plt.plot(result_corrected)
plt.legend(["Only atoms in slab","same residue as atoms in slab"])
plt.xlabel("Frame"); plt.ylabel("Number of atoms")
plt.show()
