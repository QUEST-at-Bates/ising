import numpy as np
import pandas as pd
import os 
pd.options.mode.chained_assignment = None  # default='warn'

#note: python's numbering means that i actually corresponds to y and j corresponds to x... 
#due to symmetry, this doesn't matter, but should your symmetry change, you will need to change this

def spin():
    s = 2*(np.random.randint(2))-1
    return s
    
def generate_lattice(N, bias = "random"):
    L = np.ones(shape=(N,N), dtype=int)
    if bias == "random":
        #generates random lattice
        for i,row in enumerate(L):
            for j,entry in enumerate(row):
                L[i,j] = spin()
    elif bias == "up":
        #generates random lattice
        for i,row in enumerate(L):
            for j,entry in enumerate(row):
                L[i,j] = 1
    elif bias == "down":
        #generates random lattice
        for i,row in enumerate(L):
            for j,entry in enumerate(row):
                L[i,j] = -1
    return L

def nn(i,j):
    jplusone = (i, periodic(j+1))
    jminusone = (i, periodic(j-1))
    iplusone = (periodic(i+1),j)
    iminusone = (periodic(i-1),j)
    return [jplusone,jminusone,iplusone,iminusone]
    
def periodic(i):
    if i==-1:
        return N-1
    if i == N:
        return 0
    else:
        return i

def calc_Eij(L,i,j):
    sum_nn = 0.
    nns = nn(i,j)
    for neighbor in nns:
        sum_nn += L[neighbor]
    Eij = -J*L[i,j]*sum_nn
    return Eij

def calc_E(L):
    E = 0.
    for i,row in enumerate(L):
        for j,entry in enumerate(row):
            E += calc_Eij(L,i,j)
    return E

def flip_spin(L,i,j):
    metadata = {}
    E0 = calc_Eij(L,i,j)
    Ef = -1.*E0
    dE = Ef - E0
    r = np.random.rand()
    metadata["dE"] = [dE]
    metadata["r"] = [r]
    metadata["exp(-dE/T)"] = [np.exp(-1.*dE/T)]
    if dE < 0 or r <= np.exp(-1.*dE/T):
        L[i,j] = -1*L[i,j]
        metadata["Status"] = ["Accept"]
    else:
        metadata["Status"] = ["Reject"]
    return L, pd.DataFrame(data = metadata)

def pick_site(N, touched):
    i = np.random.randint(N)
    j = np.random.randint(N)
    while (i,j) in touched:
        i = np.random.randint(N)
        j = np.random.randint(N)
    return (i,j)

def Metropolis(L,T,metadata,phase):
    N = len(L[0])
    touched = []
    count = 1
    while count <= N*N: #visit every site in random order
        i,j = pick_site(N, touched)
        touched.append((i,j))
        L, trial = flip_spin(L,i,j)
        trial["step"] = count
        trial["Phase"] = phase
        trial["T"] = T
        metadata = pd.concat([metadata, trial], ignore_index=True)
        count += 1
    return L, metadata

def calc_acc_rate(df):
    df = df.dropna()
    conditions = [ df['Status'] == "Accept", df['Status'] == "Reject",]
    outputs = [ 1.,0.]
    df["Accept"] = np.select(conditions, outputs, 'Other')
    df["Accept"] = pd.to_numeric(df["Accept"])
    return df
    
def ising_MC(T, N, ntherm, nMC):
    L = generate_lattice(N)
    metadata = pd.DataFrame(data = {"T": [np.nan], "Phase": [np.nan],"step": [np.nan], "dE": [np.nan], "r": [np.nan], "exp(-dE/T)": [np.nan], "Status": ["N/A"]})
    therm_df = metadata
    sim_df = metadata
    #thermalize
    for n in range(ntherm):
        L, therm_df = Metropolis(L,T,therm_df,"Thermalization")
    therm_df = calc_acc_rate(therm_df)
    #simulate
    E_array = [calc_E(L)]
    for n in range(nMC):
        L, sim_df = Metropolis(L,T,sim_df,"Simulation")
        E_array.append(calc_E(L))
    sim_df = calc_acc_rate(sim_df)
    E_tot = np.mean(E_array)
    metadata = pd.concat([therm_df, sim_df], ignore_index=True)
    acc_rate = sim_df["Accept"].mean()
    return L, E_tot, metadata, acc_rate


if __name__ == "__main__":
    N = 10
    J = 1.
    ntherm = 10000
    nMC = 100000
    Tmax = 5.
    dT = 0.2
    
    T = Tmax
    metadata = pd.DataFrame(data = {"T": [np.nan], "Phase": [np.nan],"step": [np.nan], "dE": [np.nan], "r": [np.nan], "exp(-dE/T)": [np.nan], "Status": ["N/A"]})
    mc_data = pd.DataFrame(data = {"E": [np.nan], "T/J": [np.nan], "acc rate": [np.nan]})
    while T > 0:
        L, E_tot, temp_metadata, acc_rate = ising_MC(T, N, ntherm, nMC)
        temp = pd.DataFrame(data = {"E": [E_tot], "T/J": [1.0*T/J], "acc rate": [acc_rate]})
        mc_data = pd.concat([mc_data,temp], ignore_index = True)
        metadata = pd.concat([metadata,temp_metadata], ignore_index = True)
        T -= dT
    metadata = metadata.dropna()
    mc_data = mc_data.dropna()
    mc_data = mc_data.set_index('T/J')
    mc_data.to_csv('ising_python.csv', index=True) 
