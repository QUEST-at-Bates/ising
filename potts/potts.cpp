// Casey Berger
// Created: May 30 2014
// Last edited: June 2, 2014
//
//

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

using namespace std;

//global variables
const int len = 10; //length of lattice
const double q = 10; //number of spin states
const int l_end = len-1; //last spot on lattice
const int num = len*len;  //total number of spins
double Tmax = 2.5;  //max temp
double Tmin = 1.5; //min temp
double dT = 0.01; //temperature iterator
const int mc_iter = 1000; //number of monte carlo iterations
const int eq_iter = 10000; //number of iterations for equilibration
double J = 1.0; //interaction strength


//function declaration
void make_lattice(int (&Lattice)[len][len][len]);
int kronecker_delta(int Lattice[len][len][len], int i, int j, int k, int i2, int j2, int k2);
double calc_E(int Lattice[len][len][len], int i, int j, int k);
void flip_spin(int (&Lattice)[len][len][len], int i, int j, int k, double T, double &dE);
double tot_E(int Lattice[len][len][len]);
void mc_sol(vector<double> &mc_E, int (&Lattice)[len][len][len], vector<double> &temp_vec);
double avg(vector<double> sample_vec);
void print_lattice(int Lattice[len][len][len]);
void write_to_file(vector<double> &mc_E, vector<double> &temp_vec);
void equilibrate(int (&Lattice)[len][len][len], double T);

int main ()
{
    srand(time(NULL)); //seed random number
    vector<double> mc_E; //stores monte carlo solution
    vector<double> temp_vec; //stores temperatures
    int Lattice[len][len][len]; //stores lattice configuration    
    mc_sol(mc_E, Lattice, temp_vec);
    write_to_file(mc_E, temp_vec);
    return 0;
}

void make_lattice(int (&Lattice)[len][len][len])
{
    //cout << "make_lattice" << endl;
    //generates random configuration
    for (int i = 0; i<len; i++)
    {
        for (int j = 0; j<len; j++)
        {
            for (int k = 0; k<len; k++)
            {
                double r = q*((double)rand())/((double)RAND_MAX)+1.0; //generate random number between 1 and q+1
                int spin = (int)r; //converts rand to an integer between 1 and q
                Lattice[i][j][k] = spin; //assigns that spin to the lattice
            }
        }
    }
}

int kronecker_delta(int Lattice[len][len][len], int i, int j, int k, int i2, int j2, int k2)
{
    if (Lattice[i][j][k] == Lattice[i2][j2][k2])
    {
        return 1;
    }
    else
    {
        return (-1);
    }
}

double calc_E(int Lattice[len][len][len], int i, int j, int k)
{
    //cout << "calc_E" << endl;
    //calculates energy at a particular point (i,j) on the lattice
    double Energy = 0.0;
    int up; //pos y dir (j)
    int down; //neg y dir (j)
    int left; //neg x dir (i)
    int right; //pos x dir (i)
    int towards; //pos z dir (k)
    int away; //neg z dir (k)
    
    if (i==0)
    {
        left = l_end;
        right = i+1;
    }
    else if (i==l_end)
    {
        left = i-1;
        right = 0;
    }
    else
    {
        left = i-1;
        right = i+1;
    }
    if (j==0)
    {
        up = j+1;
        down = l_end;
    }
    else if (j==l_end)
    {
        up = 0;
        down = j-1;
    }
    else
    {
        up = j+1;
        down = j-1;
    }
    if (k==0)
    {
        towards = k+1;
        away = l_end;
    }
    else if (k==l_end)
    {
        towards = 0;
        away = k-1;
    }
    else
    {
        towards = k+1;
        away = k-1;
    }
    
    int nnleft = kronecker_delta(Lattice,i,j,k,left,j,k);
    int nnright = kronecker_delta(Lattice,i,j,k,right,j,k);
    int nnup = kronecker_delta(Lattice,i,j,k,i,up,k);
    int nndown = kronecker_delta(Lattice,i,j,k,i,down,k);
    int nntowards = kronecker_delta(Lattice,i,j,k,i,j,towards);
    int nnaway = kronecker_delta(Lattice,i,j,k,i,j,away);
    
    Energy = -1.0*J*(nnleft+nnright+nnup+nndown+nntowards+nnaway);
    
    if (Energy == -0)
    {return 0;}
    else
    {return Energy;}
}

void flip_spin(int (&Lattice)[len][len][len], int i, int j, int k, double T, double &dE)
{
    //cout << "flip_spin" << endl;
    //flips spin if correct conditions are met
    double E0 = calc_E(Lattice, i, j, k); //calculate initial energy at that site
    int oldspin = Lattice[i][j][k];
    int newspin = Lattice[i][j][k];
    while (newspin == oldspin) //make a new, random spin that is not the same as the old one
    {
        double r = q*((double)rand())/((double)RAND_MAX)+1.0; //generate random number between 1 and q+1
        newspin = (int)r; //converts rand to an integer between 1 and q
    }
    
    Lattice[i][j][k] = newspin; //change spin
    double Ef = calc_E(Lattice, i, j, k); //calculate new energy from new spin
    dE = Ef-E0;
    //cout << "dE = " << dE << endl;
    double r = ((double)rand())/((double)RAND_MAX);
    if (Ef <= E0)
    {Lattice[i][j][k] = newspin;}
    else
    {
        if (r<=exp(-1.0*dE/T))
        {Lattice[i][j][k] = newspin;}
        else
        {
            dE = 0.0;
            Lattice[i][j][k] = oldspin;
        }
    }
}

double tot_E(int Lattice[len][len][len])
{
    //cout << "tot_E" << endl;
    //calculates energy of entire configuration
    double Energy = 0.0;
    //J = -1.0*J;
    double dE = 0.0;
    for (int i = 0; i < len; i++)
    {
        for (int j = 0; j < len; j++)
        {
            for (int k = 0; k<len; k++)
            {
                dE  = calc_E(Lattice, i, j, k);
                Energy = Energy+dE;
            }
        }
    }
    return 0.5*Energy;
}

void mc_sol(vector<double> &mc_E, int (&Lattice)[len][len][len], vector<double> &temp_vec)
{
    //cout << "mc_sol" << endl;
    //calculates average energy (over many lattice configs) at each temperature interval
    double T=Tmax; //set initial temp
    make_lattice(Lattice); //initialize the lattice
    //print_lattice(Lattice); //print the lattice
    
    //temperature loop
    while (T>=Tmin)
    {
        double Eavg; //Energy expectation value variable
        vector<double> E_config; //stores energy of each configuration within one temp iteration
        equilibrate(Lattice, T); //let the lattice equilibrate after each temperature change
        
        //monte carlo loop
        for (int n = 0; n < mc_iter; n++)
        {
            //pick a random spot on the lattice
            int i = len*((double)rand())/((double)RAND_MAX); //random integer
            int j = len*((double)rand())/((double)RAND_MAX); //random integer
            int k = len*((double)rand())/((double)RAND_MAX); //random integer
            int count = 0;
            double Energy = tot_E(Lattice); //initial energy
            
            //metropolis loop
            while (count < num)
            {
                double dE = 0.0; //declare and initialize the change in energy variable
                flip_spin(Lattice, i, j, k, T, dE); //flip spin if conditions are right
                
                //if (i==l_end && count%len == 0)
                if(i==l_end)
                {i = 0;}
                else
                {i++;}
                if (j == l_end)
                {j = 0;}
                else
                {j++;}
                
                count++;
            }//close metropolis loop
            Energy = tot_E(Lattice); //update energy with new config
            //cout << Energy << endl;
            E_config.push_back(Energy); //add the energy for this configuration
        }//close mc loop
        Eavg = avg(E_config); //average the energies of all configurations
        mc_E.push_back(Eavg); //add this temperature's energy to vector
        temp_vec.push_back(T); //add this temperature to vector
        E_config.clear();
        T=T-dT; //iterate temperature
    }//close temperature loop
}

double avg(vector<double> sample_vec)
{
    //cout << "avg" << endl;
    //calculates average energy of many configurations
    double avg = 0;
    int elements = sample_vec.size();
    for (int n = 0; n<elements; n++)
    {
        avg=avg+sample_vec[n];
    }
    avg = (1.0/(1.0*elements))*avg;
    //cout << avg/(1.0*num) << endl;
    return avg;
}

void print_lattice(int Lattice[len][len][len])
{
    //cout << "print_lattice" << endl;
    //prints lattice to screen
    for (int i = 0; i<len; i++)
    {
        for (int j = 0; j<len; j++)
        {
            for (int k = 0; k<len; k++)
            {
                cout << Lattice[i][j][k] << " ";
            }
            cout << " | ";
        }
        cout << endl;
    }
    cout << endl;
}

void write_to_file(vector<double> &mc_E, vector<double> &temp_vec)
{
    //cout << "write_to_file" << endl;
    //output both solutions to a .txt file to open in gnuplot
    string fname = "/Users/Casey/Dropbox/REU Project 2014 Casey/Codes/mc_potts_data.txt";
    ofstream fout; //output stream
    fout.open(fname.c_str(),ios::out);
    
    // check if files are open
    if (!fout.is_open())
    {
        cerr << "Unable to open file " << fname <<"." << endl;
        exit(10);
    }
    fout.setf(ios::fixed);
    fout << setw(20) << "# E/N" << setw(20) << "T/J" << endl;
    for (unsigned int n = 0; n<mc_E.size(); n++)
    {
        fout.setf(ios::fixed);
        fout << setw(20) << 1.0*mc_E[n]/(1.0*J*num) << setw(20) << temp_vec[n]/(abs(J)) << endl;
    }
    fout.close();
    mc_E.clear();
    temp_vec.clear();
}

void equilibrate(int (&Lattice)[len][len][len], double T)
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
                for (int k = 0; k < len; k++)
                {
                    flip_spin(Lattice, i, j, k, T, dE);
                    count++;
                }
            }
        }
    }
}

