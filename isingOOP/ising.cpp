// Casey Berger
// Created: May 7 2014
// Last edited: May 24, 2023 (OOP)

// include files
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <stdio.h>
#include <sstream>
#include <time.h>

//custom headers
#include "lattice.h"

using namespace std;
using ising::Lattice;

//global variables
//const int len = 10; //length of lattice
//const int l_end = len-1; //last spot on lattice
//const int num = len*len;  //total number of spins
//double Tmax = 4.0;  //max temp
//double Tmin = 0.0; //min temp
//double dT = 0.2; //temperature iterator
//const int nMC = 10000; //number of monte carlo iterations
//const int eq_iter = 1000; //number of iterations for equilibration
//double J = 1.0; //interaction strength

//function declaration
/*double calc_E(int Lattice[len][len], int i, int j);
void flip_spin(int (&Lattice)[len][len], int i, int j, double T, double &dE);
double tot_E(int Lattice[len][len]);
void mc_sol(vector<double> &mc_E, int (&Lattice)[len][len], vector<double> &temp_vec);
double avg(vector<double> sample_vec);
void print_lattice(int Lattice[len][len]);
void exact_sol(vector<double> &exact_E);
void equilibrate(int (&Lattice)[len][len], double T);*/
void write_to_file(vector<double> &mc_E, double T, double J, int len);
int main ()
{
    int len = 10; //length of lattice
    double J = 1.0; //interaction strength
    int nMC = 10000; //number of monte carlo iterations
    double Tmax = 4.0;  //max temp
    double Tmin = 0.0; //min temp
    double dT = 0.2; //temperature iterator
    
    Lattice L(len, J, Tmax);
    cout << "Length of lattice L = " << L.getLength() << endl;
    srand(7); //seed random number
    L.initialize();
    cout << "Initialized"<<endl;
    L.printLattice();
    double T = Tmax; //current temp
    //temperature loop
    while (T>=Tmin){
        vector<double> mc_E; //stores energies for monte carlo loop at one T
        L.metropolisLoop(nMC, mc_E);
        write_to_file(mc_E,T,J,len);
        T -= dT;
    }
    return 0;
}

void write_to_file(vector<double> &mc_E, double T, double J, int len)
{
    //cout << "write_to_file" << endl;
    //output both solutions to a .csv
    
    std::stringstream T_stream, J_stream, L_stream;
    T_stream << std::fixed << std::setprecision(3) << T; //truncate T for filename
    J_stream << std::fixed << std::setprecision(3) << J; //truncate J for filename
    L_stream << std::fixed << std::setprecision(1) << len; //truncate len for filename
    std::string str_T = T_stream.str();
    std::string str_J = J_stream.str();
    std::string str_L = L_stream.str();
    std::string fname = "data/ising_MC_T_" + str_T + "_J_" + str_J +"_L_" + str_L + ".csv";//output filename
    ofstream fout; //output stream
    fout.open(fname.c_str(),ios::out);
    
    // check if files are open
    if (!fout.is_open())
    {
        cerr << "Unable to open file " << fname <<"." << endl;
        exit(10);
    }
    fout.setf(ios::fixed);
    fout << "# E, T, J, L" << endl;
    for (unsigned int n = 0; n<mc_E.size(); n++)
    {
        fout.setf(ios::fixed);
        fout << mc_E[n] << "," << T << "," << J << "," << len << endl;
    }
    fout.close();
    mc_E.clear();
}
/*
void equilibrate(int (&Lattice)[len][len], double T)
{
    //cout << "equilibrate" << endl;
    double dE = 0.0;
    int count = 0;
    while (count < eq_iter)
    {
        for (int i = 0; i < len; i++)
        {
            for (int j = 0; j < len; j++)
            {
                flip_spin(Lattice, i, j, T, dE);
                count++;
            }
        }
    }
}
*/