import mdtraj as md
import numpy as np

traj = md.load_xtc('example.xtc',top='example.gro')
traj.n_atoms
traj.n_residues
dir(traj)                      # "vars" gives variables in object

# 2 main parts to a trajectory, coordinates and topology
traj.xyz                       # units in nm, not Angstroms
traj.xyz.shape                 # nframes * nparticles * xyz
topology = traj.topology
dir(topology)

# selection
topology.select("protein")
topology.select("resid 1 to 10")
topology.select("element H")
topology.select("resname ALA")
topology.select("resname ALA or resname GLY")
topology.select("resname ALA and element H")
protein_indices = topology.select('protein')

# can slice just the coordinates, or make a whole new trajectory
protein_coords  = traj.xyz[:,protein_indices,:]
protein_traj    = traj.atom_slice(protein_indices)

# selection based on geometric constraints
bottom_half_indices = np.where(traj.xyz[0,:,2] < np.mean(traj.xyz[0,:,2]))[0]
bottom_half_traj    = traj.atom_slice(bottom_half_indices)

# combinations of selections by index comparison
bottom_half_and_protein_indices = np.intersect1d(protein_indices,bottom_half_indices)

# combinations of selections by boolean arrays
sys_mean = traj.xyz[0,:,2].mean()
sys_min  = traj.xyz[0,:,2].min()

bottom_half_boolean     = traj.xyz[0,:,2] < sys_mean
not_bot_quarter_boolean = traj.xyz[0,:,2] > ((sys_mean + sys_min) / 2)
bottom_half_boolean & not_bot_quarter_boolean


# trajectory atom properties
topology.residue
topology.residue(0)
topology.residue(0).atoms
topology.residues
topology.atom
topology.atoms

# getting indices of a specific residue
res_zero_indices    = [atom.index for atom in topology.residue(0).atoms]
res_zero_indices_v2 = [atom.index for atom in topology.atoms if atom.residue.index == 0]
protein_indices_v2  = [atom.index for atom in topology.atoms if atom.residue.is_protein]
# -----------------------------------------------------------------------------
# example 2 - geometric analyses
# -----------------------------------------------------------------------------

''' Shivangi has a trajectory containing proteins, lipids, water, and ions.
She wants to run analysis on a subset of the system that lies with in a z-slice
(ie z > 1 nm, z < 3 nm). The problem is, this subset is dynamic - it changes
over time as things diffuse in and out of the z block. An additional
complication is that molecules can be partially in the block: if only part of
the molecule is in this z slab, we still need to run the analysis on the WHOLE
molecule'''

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
for i in range(traj.n_frames):                       # frame by frame analysis
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


# -----------------------------------------------------------------------------
# analysis tools
# ----------------------------------------------------------------------------

dssp = md.compute_dssp(protein_traj)
fract_helical = np.sum(dssp == 'H' ) / dssp.size
fract_helical_byframe = np.sum(dssp == 'H',axis = 1) / protein_traj.n_residues
