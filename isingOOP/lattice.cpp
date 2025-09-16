// Casey Berger
// Created: May 24 2023
// Last edited: May 24, 2023
#include <iostream> //cout, endl
#include <iomanip> //setw
#include <vector> //vector
#include <cmath> //exp

#include "lattice.h"

namespace ising {
    //public functions
    Lattice::Lattice(int length, double J, double kBT){
        Lattice::setLength(length); //set length
        interactionJ_ = J; //set J (do I need to do this the same way I do length?)
        kBT_ = kBT;//see above note
    }
    void Lattice::setLength(int length){
        length_ = length;
    }
    int Lattice::getLength(){
        return length_;
    }
    void Lattice::initialize(){
        int len = Lattice::getLength();
        int ** grid = new int*[len];
        for(int i = 0; i < len; i++){
            grid[i] = new int[len];
            for (int j = 0; j<len; j++){
                double r = ((double) std::rand())/((double) RAND_MAX);
                if (r<0.5)
                    {grid[i][j] = -1;}
                else
                    {grid[i][j] = 1;}
            }
        }
        grid_ = grid;
    }
    void Lattice::printLattice(){
        int len = Lattice::getLength();
        for(int i = 0; i < len; i++){
            for (int j = 0; j<len; j++){
                std::cout << std::setw(4) << grid_[i][j] << ",";
            }
            std::cout << std::endl;
        }
    }
    
    int* Lattice::getNeighbors(int i, int j){
        static int nn[8];
        nn[0] = Lattice::plusOne_(i);
        nn[1] = j;
        nn[2] = i;
        nn[3] = Lattice::plusOne_(j);
        nn[4] = Lattice::minusOne_(i);
        nn[5] = j;
        nn[6] = i;
        nn[7] = Lattice::minusOne_(j);
        return nn;
    }
    
    int* Lattice::getNeighborSpins(int i, int j){
        static int nnSpins[4];
        nnSpins[0] = grid_[Lattice::plusOne_(i)][j];
        nnSpins[1] = grid_[i][Lattice::plusOne_(j)];
        nnSpins[2] = grid_[Lattice::minusOne_(i)][j];
        nnSpins[3] = grid_[i][Lattice::minusOne_(j)];
        return nnSpins;
    }
    
    double Lattice::localEnergy(int i, int j){
        int *nnSpins = Lattice::getNeighborSpins(i,j);
        double nnsum = nnSpins[0] + nnSpins[1] + nnSpins[2] + nnSpins[3];
        double Eloc = (-1.*interactionJ_ * grid_[i][j] * nnsum);
        //std::cout << "Eloc = " << Eloc <<  std::endl;
        return Eloc;
    }
    
    double Lattice::Energy(){
        double Energy;
        for(int i = 0; i < length_; i++){
            for (int j = 0; j<length_; j++){
                Energy += Lattice::localEnergy(i,j);
            }
        }
        return Energy;
    }
    
    void Lattice::metropolisLoop(int nMC, std::vector<double> &mc_E){
        //adds energy for individual configurations to a vector so we can do stats for that temperature
        for (int n = 0; n < nMC; n++){
            Lattice::sweepLattice_(); //iterate over the whole lattice, flipping spins if favorable
            mc_E.push_back(Lattice::Energy());
        }
    }
    
    //private functions
    int Lattice::plusOne_(int i){
        //int len = Lattice::getLength();
        if(i+1 == length_){ return 0;}
        else{return i+1;}
    }
    
    int Lattice::minusOne_(int i){
        if(i == 0){ return length_-1;}
        else{return i-1;}
    }
        
    void Lattice::flipSpin_(int i, int j){
        double E_init = Lattice::localEnergy(i, j);
        double dE = -2.*E_init;
        double r = ((double) std::rand())/((double) RAND_MAX);
        if (dE < 0){
            grid_[i][j] = -1 * grid_[i][j];
        }
        else{
            if (r<=std::exp(-1.0*dE/kBT_)){
                grid_[i][j] = -1*grid_[i][j];
            }
            else{
                dE = 0.0;
            }
        }
    }
    
    void Lattice::sweepLattice_(){
        //consider making two arrays for i and j, which are shuffled orders of the indices
        for(int i = 0; i < length_; i++){
            for (int j = 0; j<length_; j++){
                Lattice::flipSpin_(i, j);
            }
        }
    }
}