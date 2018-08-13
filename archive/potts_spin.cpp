/* Trials with Potts_Spin based optimization.
 *
 * Copyright (C) 2014 Jennifer David. All rights reserved.
 *
 * BSD license:
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of
 *    contributors to this software may be used to endorse or promote
 *    products derived from this software without specific prior written
 *    permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR THE CONTRIBUTORS TO THIS SOFTWARE BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
   \file potts_spin.cpp
   
   Trials with running neural network based optimization method for task assignment. 
   
*/
#include <fstream>
#include <math.h>
#include <iomanip> // needed for setw(int)
#include <string>
#include "stdio.h"
#include <iostream>
#include <cstdlib>
#include <Eigen/Dense>
#include <Eigen/LU> 
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <time.h>
#include <limits>
#include "gnuplot_iostream.h"

using namespace std;
using namespace Eigen;

int nVehicles = 2;
int nTasks = 4;
int nDim = 2*nVehicles + nTasks;
int nComp = nDim*nDim;
int rDim = nTasks + nVehicles;
double kT_start = 100;
double kT_stop  = 0.001;
//double kT_swfac = 0.00015;
//double kT_fac = exp( log(kT_swfac) / (nVehicles * nDim) ); // lower T after every neuron update
double kT = kT_start;// * kT_fac;  // * to give the ini conf a own datapoint in sat plot
double kT_fac = 0.99;
static const double small = 1e-15;
static const double onemsmall = 1 - small;
static const double lk0 = 1/small - 1; 
double lk;
double g = 100;
double psi = 1;
double kappa;

std::string veh_alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
std::string task_alpha = "abcdefghijklmnopqrstuvwxyz";
std::string solStrA;
std::string solStrB;

int indx = 0;
int indxB = 0;
int checkTime;

////////////////////////////////////////////////////////////////////////////////////////

Eigen::MatrixXd VMatrix = MatrixXd::Zero(nDim,nDim);
Eigen::MatrixXd UpdatedVMatrix = MatrixXd::Zero(nDim,nDim);
Eigen::MatrixXd NonNormUpdatedVMatrix = MatrixXd::Zero(nDim,nDim);
Eigen::MatrixXd DeltaVMatrix = MatrixXd::Zero(nDim,nDim);
Eigen::MatrixXd DeltaPMatrix = MatrixXd::Zero(nDim,nDim);

Eigen::MatrixXd PMatrix = MatrixXd::Zero(nDim,nDim);
Eigen::MatrixXd P = MatrixXd::Zero(nDim,nDim);
Eigen::MatrixXd PP = MatrixXd::Zero(nDim,nDim);
Eigen::MatrixXd PPP = MatrixXd::Zero(nDim,nDim);
Eigen::MatrixXd PPP2 = MatrixXd::Zero(nDim,nDim);
Eigen::MatrixXd UpdatedPMatrix = MatrixXd::Zero(nDim,nDim);

Eigen::MatrixXd DeltaMatrix = MatrixXd::Ones(nDim,nDim);
Eigen::MatrixXd TauMatrix = MatrixXd::Zero(nDim,nDim);
Eigen::MatrixXd I = MatrixXd::Identity(nDim,nDim); //identity matrix
Eigen::MatrixXd dQ = MatrixXd::Zero(nDim,nDim);

Eigen::MatrixXd E_local = MatrixXd::Zero(nDim,nDim);
Eigen::MatrixXd E_loop = MatrixXd::Zero(nDim,nDim);
Eigen::MatrixXd E_task = MatrixXd::Zero(nDim,nDim);
Eigen::MatrixXd E_norm = MatrixXd::Zero(nDim,nDim);

Eigen::MatrixXd Et = MatrixXd::Zero(nDim,nDim);
Eigen::MatrixXd E = MatrixXd::Zero(nDim,nDim);

Eigen::MatrixXd initialE_local = MatrixXd::Zero(nDim,nDim);
Eigen::MatrixXd initialE_loop = MatrixXd::Zero(nDim,nDim);
Eigen::MatrixXd initialE_task = MatrixXd::Zero(nDim,nDim);
Eigen::MatrixXd initialE = MatrixXd::Zero(nDim,nDim);

Eigen::MatrixXd AMatrix = MatrixXd::Zero(nDim,nDim);
Eigen::MatrixXd BMatrix = MatrixXd::Zero(nDim,nDim);

Eigen::MatrixXd vDeltaR;
Eigen::MatrixXd vDeltaL;
Eigen::MatrixXd ivDeltaR;
Eigen::MatrixXd ivDeltaL;

Eigen::VectorXd ivdVecL(nDim);
Eigen::VectorXd ivdVecR(nDim);
Eigen::VectorXd irightVec(nDim);
Eigen::VectorXd ileftVec(nDim);
Eigen::VectorXd sumv; 
Eigen::VectorXd sumw; 
Eigen::VectorXd sumr; 
Eigen::VectorXd sumc;
Eigen::VectorXd dv(nDim);
Eigen::VectorXd cTime(nVehicles);

Eigen::VectorXd TVec(nDim);
Eigen::VectorXd vdVecL(nDim);
Eigen::VectorXd vdVecR(nDim);
Eigen::VectorXd rightVec(nDim);
Eigen::VectorXd leftVec(nDim);

////////////////////////////////////////////////////////////////////////////////////////////////

  
void getVMatrix ()
  {
      cout << "\n ///////////////////////////////////////////////////////////////////// " << endl;
      cout << "\n No. of Vehicles M = " << nVehicles << endl;
      cout << "\n No. of Tasks N = " << nTasks << endl;
      cout << "\n Dimension of VMatrix is: " << nDim << endl;
      cout << "\n Total no of Vij: " << nDim*nDim << endl;
        
      // double r = ((double) rand() / (RAND_MAX))/80;
      // cout << "\n r is: " << r << endl;
      double tmp = 1./(rDim-1);
      cout << "\n tmp is: " << tmp << endl;

      // intitalising VMatrix with 1/n values
      for (int i = 0; i < nDim; i++)
        {
            for (int j = 0; j < nDim; j++)
            {
                VMatrix(i,j) = tmp;// + (((double) rand() / (RAND_MAX))/80) ;// + 0.02*(rand() - 0.5);	// +-1 % noise
            }
        }
    
      //adding all the constraints for the vehicles and tasks
      VMatrix.diagonal().array() = 0;
      VMatrix.leftCols(nVehicles) *= 0;
      VMatrix.bottomRows(nVehicles) *= 0;
      VMatrix.topRightCorner(nVehicles,nVehicles) *= 0;  
      int y = 0;
      LABEL0:
      sumr = VMatrix.rowwise().sum();
      for (int i = 0; i < nDim; i++)
                {
                    if (sumr(i) == 0.000)
                        { 
                            //cout << "\n Row " << k << " is with constraints, so skipping" << endl;
                        }
                    else if (sumr(i) == 1.000)
                        { 
                            //cout << "\n Row " << k << " is already normalised" << endl;
                        }
                    else 
                        {   // cout << "\n Row Normalising " << endl;
                            for (int j = 0; j < nDim; j++)
                            {VMatrix(i,j) = VMatrix(i,j)/sumr(i);}
                        }
                 }                  
            //cout << "\n So finally, the Row Normalised UpdatedVMatrix is \n" << UpdatedVMatrix << endl;
            //cout << "\n row sum after row normalisation is \n" << UpdatedVMatrix.rowwise().sum() << endl;
   
            //Normalising columns of VMatrix
            sumc = VMatrix.colwise().sum();
            for (int i = 0; i < nDim; i++)
                {
                    if (sumc(i) == 0)
                        {//cout << "\n Col " << k << " is with constraints, so skipping" << endl;
                        }
                    else if (sumc(i) == 1)
                        {//cout << "\n Col " << k << " is already normalised" << endl;
                        }   
                    else 
                        {//cout << "\n Column Normalising \n" << endl;
                            for (int j = 0; j < nDim; j++)
                                {VMatrix(j,i) = VMatrix(j,i)/sumc(i);}
                        }
                 }
        y++;
        if (y != 200)
            goto LABEL0;
        else
            cout << "\n VMatrix is: \n" << VMatrix << endl;
            cout << "Sum of VMatrix along the row \n" << VMatrix.rowwise().sum() << endl;
            cout << "Sum of VMatrix along the col \n" << VMatrix.colwise().sum() << endl;
            PMatrix = (I - VMatrix).inverse();
            cout << "\n PMatrix is as in (I - V)^(-1): \n" << PMatrix << endl;         
  }
    
Eigen::MatrixXd asyncRandomUpdate(Eigen::MatrixXd UpdatedVMatrix)
{
            E = ((-1*E_local)/kT);
            int ran = rand() % 2;
            cout << "\n ran is: " << ran << endl;         

            if (ran == 1)
            {
                int RowUp = rDim;
                int RowLo = 0;
                int RowRange = abs(RowUp - RowLo);
                int RowRan = rand()% (RowRange) + RowLo;
                cout << "\n RowRan is: " << RowRan << endl;         
                
                cout << "\n Updating Row" << endl;
                for (int j = nVehicles; j < nDim; j++)
                    {UpdatedVMatrix(RowRan,j) = std::exp (E(RowRan,j));}  
                cout << "\n Updating Row VMatrix is: (the numerator) \n" << UpdatedVMatrix << endl;
            
                sumv = UpdatedVMatrix.rowwise().sum();
                cout << "\n sumv is (my ref): \n" << sumv << endl;
            
                for (int j = nVehicles; j < nDim; j++)
                    {UpdatedVMatrix(RowRan,j) = UpdatedVMatrix(RowRan,j)/sumv(RowRan);}  
                cout << "\n Updating Row VMatrix is (complete update):  \n" << UpdatedVMatrix << endl;
                
            }
            
            else
            {    
                int ColUp = nDim;
                int ColLo = nVehicles;
                int ColRange = abs(ColUp - ColLo);
                int ColRan = rand()% (ColRange) + ColLo;
                cout << "\n ColRan is: " << ColRan << endl;         
                             
                cout << "\n Updating Col" << endl;
                for (int j = 0; j < rDim; j++)
                    {UpdatedVMatrix(j,ColRan) = std::exp (E(j,ColRan));}  
                cout << "\n Updating Column VMatrix is: (the numerator) \n" << UpdatedVMatrix << endl;
            
                sumw = UpdatedVMatrix.colwise().sum();
                cout << "\n sumw is (my ref): \n" << sumw << endl;
            
                for (int j = 0; j < rDim; j++)
                    {UpdatedVMatrix(j,ColRan) = UpdatedVMatrix(j,ColRan)/sumw(ColRan);}  
                cout << "\n Updating Column VMatrix is (complete update):  \n" << UpdatedVMatrix << endl;
                
            }
                return UpdatedVMatrix;

}

Eigen::MatrixXd asyncAvgUpdate(Eigen::MatrixXd UpdatedVMatrix)
{
    //////////////////////////////////updating VMatrix///////////////////////////////////////////////////////////
            E = ((-1*E_local)/kT);
            for (int i = 0; i < rDim; i++)
                {
                    for (int j=nVehicles; j< nDim; j++)
                    {
                        AMatrix(i,j) = std::exp (E(i,j));
                    }  
                }
            cout << "\n Updating AMatrix is: \n" << AMatrix << endl;
            sumv = AMatrix.rowwise().sum();
            
            cout << "sumv of AMatrix is (my ref): \n " << sumv << endl;
            for (int i = 0; i < rDim; i++)
                {                    
                    for (int j=nVehicles; j< nDim; j++)
                    {
                        AMatrix(i,j) /= sumv(i);
                    }  
                }          
            AMatrix.topRightCorner(nVehicles,nVehicles) *= 0;  
            cout << "\n Updating AMatrix is (complete update):  \n" << AMatrix << endl;
   
            for (int i = 0; i < rDim; i++)
                {
                    for (int j=nVehicles; j< nDim; j++)
                    {
                        BMatrix(i,j) = std::exp (E(i,j));
                    }  
                }
            cout << "\n Updating BMatrix is: (the numerator) \n" << BMatrix << endl;
            sumw = BMatrix.colwise().sum();
            
            cout << "sumw of BMatrix is (my ref): \n " << sumw << endl;
            for (int i = 0; i < rDim; i++)
                {                    
                    for (int j=nVehicles; j< nDim; j++)
                    {
                        BMatrix(j,i) /= sumw(j);
                    }  
                }          
            BMatrix.topRightCorner(nVehicles,nVehicles) *= 0;  
            cout << "\n Updating BMatrix is (complete update):  \n" << BMatrix << endl;
            
            UpdatedVMatrix = 0.5 *(AMatrix + BMatrix);
            cout << "\n Updated VMatrix:  \n" << UpdatedVMatrix << endl;
            
            return UpdatedVMatrix;
            
}
    
Eigen::MatrixXd syncUpdate(Eigen::MatrixXd UpdatedVMatrix)
{
             ofstream outfile12("hist", std::ios_base::app);
            //////////////////////////////////updating VMatrix///////////////////////////////////////////////////////////
            E = ((-1*E_local)/kT);
            for (int i = 0; i < rDim; i++)
                {
                    for (int j=nVehicles; j< nDim; j++)
                    {
                        //if (VMatrix(i,j) == 0)
                        //    { UpdatedVMatrix(i,j) = 0;}
                        //else 
                            { UpdatedVMatrix(i,j) = std::exp (E(i,j));
                            }
                    }  
                }
            cout << "\n Updating VMatrix is: (the numerator) \n" << UpdatedVMatrix << endl;
            sumv = UpdatedVMatrix.rowwise().sum();
            cout << "sumv is (my ref): \n " << sumv << endl;
            for (int i = 0; i < rDim; i++)
                {                    
                    for (int j=nVehicles; j< nDim; j++)
                    {
                        //if (VMatrix(i,j) == 0)
                        //    UpdatedVMatrix(i,j) = 0;
                        //else
                            {UpdatedVMatrix(i,j) /= sumv(i);}
                    }  
                }          
            UpdatedVMatrix.topRightCorner(nVehicles,nVehicles) *= 0;  
            cout << "\n Updating Row VMatrix is (complete update):  \n" << UpdatedVMatrix << endl;
            return UpdatedVMatrix;
}

void sumCost(int iteration)
{
    //Updating the Energy 
             ofstream outfile12("hist", std::ios_base::app);

                for (int i = 0; i < nDim; i++)
                {
                    for (int j=0; j< nDim; j++)
                    {
                        double X = 0; double Y = 0;
                        double sumaa = 0; double sumbb=0;
                        
                        for (int l=(nVehicles+nTasks); l<(nDim) ; l++)
                        {
                            sumaa = sumaa + PPP(j,l);
                        }
                        
                        for (int l=0; l< nVehicles; l++)
                        {
                            sumbb = sumbb + PPP(l,i);
                        }
                                                
                        X = ((leftVec(i) + DeltaMatrix(i,j)) * sumaa);
                        Y = ((rightVec(j) + DeltaMatrix(i,j)) * sumbb);
                        
                        E_task(i,j) = (0.5/nVehicles)*(X + Y);
                        
                        
                outfile12 << "\n" << iteration << "\t";
                for (int i=0; i<nDim; i++)
                    {
                        outfile12 << sumaa << "\t" << sumbb << "\t";  
                    }
                        
                        //Eloop calculation
                        double lk = PPP(j,i) / PPP(i,i);	
                            if ( lk < onemsmall )
                                lk = lk/(1-lk); 
                            else
                                lk = lk0;  
                        E_loop(i,j) = lk;
                        //E_loop(i,j) = PPP2(j,i);
                        
                        double normsum = 0;
                        for (int l=rDim; l<nDim ; l++)
                        {
                            normsum = normsum + UpdatedVMatrix(l,j);
                        }
                       //Enorm calculation
                       E_norm(i,j) = 2*(normsum-1);
                    }
                }
}

void maxCost(int iteration,MatrixXf::Index maxl,MatrixXf::Index maxr)
{
    //Updating the Energy 
                for (int i = 0; i < nDim; i++)
                {
                    for (int j=0; j< nDim; j++)
                    {
                        double X = 0; double Y = 0;
                        
                        X = ((leftVec(i) + DeltaMatrix(i,j)) * PPP(j,maxl));
                        Y = ((rightVec(j) + DeltaMatrix(i,j)) * PPP(maxr,i));
                        E_task(i,j) = (0.5/nVehicles)*(X + Y);
                            
                        //Eloop calculation
                        double lk = PPP(j,i) / PPP(i,i);	
                            if ( lk < onemsmall )
                                lk = lk/(1-lk); 
                            else
                                lk = lk0;  
                        E_loop(i,j) = lk;
                        //E_loop(i,j) = PPP2(j,i);
                        
                        double normsum = 0;
                        for (int l=0; l<nDim ; l++)
                        {
                            normsum = normsum + UpdatedVMatrix(l,j);
                        }
                       //Enorm calculation
                       E_norm(i,j) = 2*(normsum-1);
                    }
                }
}
  
///////////////////////////////////////////////////////////////////////////////////////////////////
  
void NN_algo()
    {             
            //calculating initial values L,R, energy for V and P
            ivDeltaL = VMatrix.transpose() * DeltaMatrix;
            cout << "\n ivDeltaL is: \n" << ivDeltaL << endl;
            ivdVecL = ivDeltaL.diagonal();
            cout << "\n ivdVecL is: \n" << ivdVecL << endl;
            ileftVec = PMatrix.transpose() * (TVec + ivdVecL);
            cout << "\n ileftVec is: \n" << ileftVec << endl;


            ivDeltaR = VMatrix * DeltaMatrix.transpose();
            cout << "\n ivDeltaR is: \n" << ivDeltaR << endl;
            ivdVecR = ivDeltaR.diagonal();
            cout << "\n ivdVecR is: \n" << ivdVecR << endl;
            irightVec = PMatrix * (TVec + ivdVecR);
            cout << "\n irightVec is: \n" << irightVec << endl;

            MatrixXf::Index imaxl, imaxr;
            double imaxleftVecInd, imaxrightVecInd;
            imaxleftVecInd = ileftVec.maxCoeff(&imaxl);
            imaxrightVecInd = irightVec.maxCoeff(&imaxr);
            /*cout << "imaxl is: " << imaxl << endl;
            cout << "imaxr is: " << imaxr << endl;*/
            kappa = 0.5 * (ileftVec(imaxl) + irightVec(imaxr));
            cout << "\n Kappa is: \n" << kappa << endl;
            
            cout << "\n ///////////////////////////////////////////////////////////////////// " << endl;

            for (int i = 0; i < nDim; i++)
                {
                    for (int j=0; j< nDim; j++)
                    {
                        double iX = 0; double iY = 0;
                        
                        double isumaa = 0; double isumbb=0;
                        for (int l=rDim; l<nDim ; l++)
                        {
                            isumaa = isumaa + PMatrix(j,l);
                        }
                        for (int l=0; l< nVehicles; l++)
                        {
                            isumbb = isumbb + PMatrix(l,i);
                        }
                        
                        /*cout << "\n leftVec(i) is: " << leftVec(i) << endl;
                        cout << "\n PPP(j,maxl) is: " << PPP(j,maxl) << endl;
                        cout << "\n righttVec(j) is: " << rightVec(j) << endl;
                        cout << "\n PPP(maxr,i) is: " << PPP(maxr,i) << endl;
                        
                        iX = ((ileftVec(i) + DeltaMatrix(i,j)) * PMatrix(j,imaxl));
                        iY = ((irightVec(j) + DeltaMatrix(i,j)) * PMatrix(imaxr,i));
                        /*cout << "\n X is: " << X << endl;
                        cout << "\n Y is: " << Y << endl;*/
                        
                        iX = (ileftVec(i) + DeltaMatrix(i,j)) * isumaa;
                        iY = (irightVec(i) + DeltaMatrix(i,j)) * isumbb;
                        initialE_task(i,j) = (0.5/nVehicles)*(iX + iY);
                        
                        double initial_lk = PMatrix(j,i) / PMatrix(i,i);	// the "zeroed" Pji
                            if ( initial_lk < onemsmall )
                                initial_lk = initial_lk/(1-initial_lk); // => the resulting Pji for choice j
                            else
                                initial_lk = lk0;  
                        initialE_loop(i,j) = initial_lk;
                    }
                }
                
         initialE_task.leftCols(nVehicles) *= 10000000;
         initialE_task.bottomRows(nVehicles) *= 10000000;
         initialE_task.topRightCorner(nVehicles,nVehicles) *= 10000000;  //adding all the constraints for the vehicles and tasks  
         initialE_task.topRightCorner(nVehicles,nVehicles) = initialE_task.bottomLeftCorner(nVehicles,nVehicles).eval();       
         initialE_task.topLeftCorner(nVehicles,nVehicles) = initialE_task.bottomLeftCorner(nVehicles,nVehicles).eval();  
         initialE_task = initialE_task/kappa;
         initialE_loop.diagonal().array() = 10000000;
                
         initialE_local = (g * initialE_loop) + (psi * initialE_task);
         E_local = initialE_local;      
              
         cout << "\n initial E_task is: \n" << initialE_task << endl;
         cout << "\n initial E_loop is: \n" << initialE_loop << endl;
         cout << "\n initial E_local is: \n" << initialE_local << endl;
         E_local = initialE_local;
         cout << "\n Initial VMatrix is: \n" << VMatrix << endl;
         cout << "\n Initial PMatrix is: \n" << PMatrix << endl;
         cout << "\n TauMatrix is: \n" << TauMatrix << endl;
         cout << "\n E_local is: \n" << E_local << endl;
         cout << "\n ////////////////////////////////////////////////////////////////////////// " << endl;
        
         int iteration = 1;
         int FLAG = 1;
         ofstream outfile4("VMatrix", std::ios_base::app);
         ofstream outfile5("results/PMatrix.txt", std::ios_base::app);
         ofstream outfile6("results/DeltaVMatrix.txt", std::ios_base::app);
         ofstream outfile7("results/leftVec.txt", std::ios_base::app);
         ofstream outfile8("results/rightVec.txt", std::ios_base::app);
         ofstream outfile9("results/Elocal.txt", std::ios_base::app);
         ofstream outfile10("results/Eloop.txt", std::ios_base::app);
         ofstream outfile11("results/Etask.txt", std::ios_base::app);
         ofstream outfile13("V", std::ios_base::app);

         while (FLAG != 0)
         {	
            cout << "\n" << iteration << " ITERATION STARTING" << endl;
            cout << "\n kT is " << kT << endl;
            cout << "\n PMatrix before is \n" << PMatrix << endl;
            cout << "\n VMatrix before is \n" << VMatrix << endl;
            
            UpdatedVMatrix = syncUpdate(VMatrix);
            //UpdatedVMatrix = asyncRandomUpdate(VMatrix);
            //UpdatedVMatrix = asyncAvgUpdate(VMatrix);

            cout << "\n VMatrix now is \n" << UpdatedVMatrix << endl;

            outfile13 << "\n" << iteration << "\t";
            for (int i=0; i<nDim; i++)
                    {
                        for (int j = 0; j <nDim; j++)
                        {
                           outfile13 << UpdatedVMatrix(i,j) << "\t";
                        }
                    }
                    
            for (int i = 0; i < nDim; i++)
                {                    
                    for (int j = 0; j< nDim; j++)
                    {   
                        if ( std::isnan(UpdatedVMatrix(i,j)) )
                        {
                          cout << "\n The matrix has Nan" << endl;
                          return;
                        }
                    }
                }
            NonNormUpdatedVMatrix = UpdatedVMatrix;
                
             /*else
             {
                for (int i = 0; i < nDim; i++)
                 {
                     for (int j=0; j< nDim; j++)
                     {
                         if (VMatrix(i,j) == 0)
                             { UpdatedVMatrix(i,j) = 0;}
                         else 
                             { UpdatedVMatrix(i,j) = std::exp (E(i,j));}
                     }  
                }                 
             sumw = UpdatedVMatrix.colwise().sum();
             cout << "sumv is (my ref): \n " << sumv << endl;             
             for (int i = 0; i < nDim; i++)
                 {                    
                     for (int j=0; j< nDim; j++)
                     {
                         if (VMatrix(i,j) == 0)
                             UpdatedVMatrix(i,j) = 0;
                         else
                         UpdatedVMatrix(j,i) /= sumw(i);
                     }  
                 }  
            cout << "\n After Col Updation, UpdatedVMatrix is: \n" << UpdatedVMatrix << endl;
            } */
            
            //Normalising till the values along the row/columns is zero
            cout << "\n ///////////////////////////////////////////////////////////////////// " << endl;
            cout << "\n NORMALISATION BEGINS \n" << endl;
            int x = 0;
            LABEL1:
            Eigen::VectorXd sum_row; 
            sum_row = UpdatedVMatrix.rowwise().sum();
            //cout << "\n row sum before row normalisation is \n" << sum_row << endl;
            for (int k = 0; k < sum_row.size(); k++)
                {
                    if (sum_row(k) == 0.000)
                        {
                         // cout << "\n Row " << k << " is with constraints, so skipping" << endl;
                        }
                    else if (sum_row(k) == 1.000)
                        {
                         // cout << "\n Row " << k << " is already normalised" << endl;
                        }
                    else 
                        {
                         // cout << "\n Row Normalising " << endl;
                            for (int i = 0; i < nDim; i++)
                                {
                                    UpdatedVMatrix(k,i) = UpdatedVMatrix(k,i)/sum_row(k);
                                }
                        }
                 }                  
            //cout << "\n So finally, the Row Normalised UpdatedVMatrix is \n" << UpdatedVMatrix << endl;
            //cout << "\n row sum after row normalisation is \n" << UpdatedVMatrix.rowwise().sum() << endl;
   
            //Normalising columns of VMatrix
            Eigen::VectorXd sum_col; 
            sum_col = UpdatedVMatrix.colwise().sum();
            //cout << "\n col sum before column normalisation is \n" << sum_col.transpose() << endl;
            for (int k = 0; k < sum_col.size(); k++)
                {
                    if (sum_col(k) == 0)
                        {
                         // cout << "\n Col " << k << " is with constraints, so skipping" << endl;
                        }
                    else if (sum_col(k) == 1)
                        {
                         // cout << "\n Col " << k << " is already normalised" << endl;
                        }   
                    else 
                        {
                         // cout << "\n Column Normalising \n" << endl;
                            for (int j = 0; j < nDim; j++)
                                {
                                    UpdatedVMatrix(j,k) = UpdatedVMatrix(j,k)/sum_col(k);
                                }
                        }
                 }
            //cout << "\n Col Normalised UpdatedVMatrix is \n" << UpdatedVMatrix << endl;
            //cout << "\n col sum after column normalisation is \n" << UpdatedVMatrix.colwise().sum() << endl;
            //cout << "\n row sum after column normalisation is \n" << UpdatedVMatrix.rowwise().sum() << endl;
            x++;
            if (x != 200)
              goto LABEL1;
            else
            {
                cout << "\n col sum after final normalisation is \n" << UpdatedVMatrix.colwise().sum() << endl;
                cout << "\n row sum after final normalisation is \n" << UpdatedVMatrix.rowwise().sum() << endl;
                cout << "\n Final row and column normalised UpdatedVMatrix is \n" << UpdatedVMatrix << endl;
                cout << "\n NORMALISATION ENDS \n" << endl;
                cout << "\n /////////////////////////////////////////////////////////////////////////////////////// " << endl;
                
                outfile4 << "\n" << iteration << "\t";
                for (int i=0; i<nDim; i++)
                    {
                        for (int j = 0; j <nDim; j++)
                        {
                           outfile4 << UpdatedVMatrix(i,j) << "\t";
                        }
                    }
    
                cout << "\n  Previous VMatrix is \n" << VMatrix << endl;
                DeltaVMatrix = (UpdatedVMatrix - VMatrix);
                cout << "\n Delta V Matrix is \n" << DeltaVMatrix << endl;
                outfile6 << "\n" << iteration << "\t";
                for (int i=0; i<nDim; i++)
                    {
                        for (int j = 0; j <nDim; j++)
                        {
                           outfile6 << DeltaVMatrix(i,j) << "\t"; 
                        }
                    }
                 
//                 Eigen::ArrayXXf dv(nDim,nDim);
//                 for (int i=0; i<nDim; i++)
//                     {
//                         for (int j = 0; j <nDim; j++)
//                         {
//                            dv(i,j) = DeltaVMatrix(i,j); 
//                         }
//                     }
//                 if ((dv < 0.005).all())
//                 {   cout << "\n SOLUTIONS SATURATED. STOPPING ITERATION........" << endl;
//                     return;
//                 }
                    
                Eigen::ColPivHouseholderQR<MatrixXd >lu_decomp(DeltaVMatrix);
                int rank = lu_decomp.rank();
                cout << "\n Rank of Delta V Matrix is: " << rank << endl;
                dQ = DeltaVMatrix * PMatrix;    
            
                cout << "\n dQ:\n" << dQ << endl;
                cout << "\n trace of dQ:\n" << dQ.trace() << endl;
                cout << "\n 1 - (trace of dQ) :\n" << (1 - dQ.trace()) << endl;

                cout << "\n PMatrix old is \n" << PMatrix << endl;             
                DeltaPMatrix = (PMatrix * dQ)/(1-(dQ.trace()));
                cout << "\n Delta PMatrix:\n" << DeltaPMatrix << endl;
             
                UpdatedPMatrix = PMatrix + DeltaPMatrix;
                cout << "\n Updated PMatrix using SM method is:\n" << UpdatedPMatrix << endl;
                
                PP = (I-UpdatedVMatrix);
                PPP = PP.inverse();
                PPP2 = PPP*PPP;
                cout << "\n Updated PMatrix using exact inverse is:\n" << PPP << endl;
                
                outfile5 << "\n" << iteration << "\t";
                for (int i=0; i<nDim; i++)
                    {
                        for (int j = 0; j <nDim; j++)
                        {
                           outfile5 << PPP(i,j) << "\t";  
                        }
                    }
                    
                cout << "\n ///////////////////////////////////////////////////////////////////// " << endl;
                //  Computing L and R
                vDeltaL = UpdatedVMatrix.transpose() * DeltaMatrix;
                cout << "\n vDeltaL is: \n" << vDeltaL << endl;
                vdVecL = vDeltaL.diagonal();
                cout << "\n vdVecL is: \n" << vdVecL << endl;
                leftVec = PPP.transpose() * (TVec + vdVecL);
                cout << "\n leftVec is: \n" << leftVec << endl;

                vDeltaR = UpdatedVMatrix * DeltaMatrix.transpose();
                cout << "\n vDeltaR is: \n" << vDeltaR << endl;
                vdVecR = vDeltaR.diagonal();
                cout << "\n vdVecR is: \n" << vdVecR << endl;
                rightVec = PPP * (TVec + vdVecR);
                cout << "\n rightVec is: \n" << rightVec << endl;
       
                MatrixXf::Index maxl, maxr;
                double maxleftVecInd = leftVec.maxCoeff(&maxl);
                double maxrightVecInd = rightVec.maxCoeff(&maxr);
                
                outfile7 << "\n" << iteration << "\t";
                for (int i=0; i<nDim; i++)
                    {
                        outfile7 << leftVec(i) << "\t";  
                    }
                
                outfile8 << "\n" << iteration << "\t";
                for (int i=0; i<nDim; i++)
                    {
                        outfile8 << rightVec(i) << "\t";  
                    }

                sumCost(iteration);
                //maxCost(iteration,maxl,maxr);
                
                E_task.leftCols(nVehicles) *= 10000000000;
                E_task.bottomRows(nVehicles) *= 10000000000;
                E_task.topRightCorner(nVehicles,nVehicles) *= 10000000000;  //adding all the constraints for the vehicles and tasks  
                E_task.topRightCorner(nVehicles,nVehicles) = E_task.bottomLeftCorner(nVehicles,nVehicles).eval();
                E_task.topLeftCorner(nVehicles,nVehicles) = E_task.bottomLeftCorner(nVehicles,nVehicles).eval();       
                E_loop.diagonal().array() = 1000000000000;

                E_task = E_task/kappa;
                E_local = (g * E_loop) + (psi * E_task);// + (E_norm); 
            
                cout << "\n E_local is: \n" << E_local << endl;
                cout << "\n E_task is: \n" << E_task << endl;
                cout << "\n E_loop is: \n" << E_loop << endl;
                
                 for (int i=0; i<nDim; i++)
                    {
                        for (int j = 0; j <nDim; j++)
                            {
                            if (E_task(i,j) > 100000)
                                Et(i,j) = 0;
                            else
                                Et(i,j) = E_task(i,j);
                            }
                     }
                     
                
                outfile11 << "\n" << iteration << "\t";
                for (int i=0; i<nDim; i++)
                    {
                        for (int j = 0; j <nDim; j++)
                        {
                           outfile11 << E_local(i,j) << "\t";  
                        }
                    }
                
                outfile10 << "\n" << iteration << "\t";
                for (int i=0; i<nDim; i++)
                    {
                        for (int j = 0; j <nDim; j++)
                        {
                           outfile10 << Et(i,j) << "\t";  
                        }
                    }
                
                outfile9 << "\n" << iteration << "\t";
                for (int i=0; i<nDim; i++)
                    {
                        for (int j = 0; j <nDim; j++)
                        {
                           outfile9 << E_loop(i,j) << "\t";  
                        }
                    }
                
                kT *= kT_fac;
                cout << "\n new kT is: " << kT << endl;
                cout << "\n" << iteration << " ITERATION DONE" << endl;
                iteration = iteration + 1;
                VMatrix = UpdatedVMatrix;
                PMatrix = PPP;
                cout << "\n /*/*/*/*/*/*/*/**/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/ " << endl;
                
                if (kT < kT_stop) 
                FLAG = 0;
            }
    }
}  


void displaySolution()
{
    
    for (int i = 0; i < nVehicles; i++)
    {
            for (int j = 0; j < nDim; j++)
            {if (VMatrix(i,j)==1)
                {indx = j;}
            }
            if (i == 0)
            {
                solStrA = std::string("S") + veh_alpha[i] + std::string(" -> ") + task_alpha[indx-nVehicles];
                solStrB = "max(" + std::to_string(DeltaMatrix(i,indx)) + std::string(" + ") + std::to_string(TVec(indx));
            }
            else
            {
                solStrA = solStrA + std::string(" & S") + veh_alpha[i] + std::string(" -> ") + task_alpha[indx-nVehicles];
                solStrB = solStrB + std::string(", ") + std::to_string(DeltaMatrix(i,indx)) + std::string(" + ") + std::to_string(TVec(indx));
            }
            cTime[i] = DeltaMatrix(i,indx) + TVec(indx);
            
            while (indx <= (nDim-nVehicles-1))
            {
                 for (int j = 0; j < nDim; j++)
                    {if (VMatrix(indx,j)==1)
                        {indxB = j;}
                    }
               
                if (indxB > (nVehicles+nTasks-1))
                {
                    solStrA = solStrA + std::string(" -> E") + veh_alpha[indxB-nVehicles-nTasks];
                    solStrB = solStrB + std::string(" + ") + std::to_string(DeltaMatrix(indx,indxB));
                    cTime[i] = cTime[i] + DeltaMatrix(indx,indxB);
                    solStrB = solStrB + std::string(" = ") + std::to_string(cTime[i]);            
                }
                else
                { 
                     solStrA = solStrA + std::string(" -> ") + task_alpha[indxB-nVehicles];
                     solStrB = solStrB + std::string(" + ") + std::to_string(DeltaMatrix(indx,indxB)) + std::string(" + ") + std::to_string(TVec(indxB));
                     cTime[i] = cTime[i] + DeltaMatrix(indx,indxB) + TVec(indxB);
                }
                indx = indxB;
            }
    }
solStrB = solStrB + std::string(")");
//checkTime = *max_element(cTime , cTime + nVehicles);

cout << "This corresponds to the following routing:\n";
cout << "\n" <<solStrA << endl;
cout << "\n" <<solStrB << endl;
//cout << "\n" <<checkTime << endl;
    
}

void readFile()
{
    ifstream file("tVec.txt");
    if (file.is_open())
       {
           for (int i=0; i<nDim; i++)
           {
               double item;
               file >> item;
               TVec(i) = item;
           }
       }
    else
       cout <<"file not open"<<endl;
    //TVec = VectorXd::LinSpaced(nDim,1,10);
    
    ifstream file2("deltaMat.txt");
    if (file2.is_open())
       {
            for (int i = 0; i < nDim; i++)
                for (int j = 0; j < nDim; j++)
                    {
                        double item2;
                        file2 >> item2;
                        DeltaMatrix(i,j) = item2;
                    }
       } 
    else
       cout <<"file not open"<<endl;
    cout << "\n DeltaMatrix is: \n" << DeltaMatrix <<endl;
    
    DeltaMatrix.diagonal().array() = 10000000000;
    DeltaMatrix.leftCols(nVehicles) *= 10000000000;
    DeltaMatrix.bottomRows(nVehicles) *= 10000000000;
    DeltaMatrix.topRightCorner(nVehicles,nVehicles) = DeltaMatrix.bottomLeftCorner(nVehicles,nVehicles).eval();       
    //DeltaMatrix.row(1) += 100* DeltaMatrix.row(0);
    cout << "\n Updated DeltaMatrix is: \n" << DeltaMatrix << endl;   
    
}

void generateRandom()
{
    for (int i=0; i<nDim; i++)
           {
               TVec(i) = 1;
           }
    cout << "\n TVec is: \n" << TVec <<endl;
    
    std::srand((unsigned int) time(0));
    DeltaMatrix = MatrixXd::Random(nDim,nDim);
    double HI = 100; // set HI and LO according to your problem.
    double LO = 1;
    double range= HI-LO;
    DeltaMatrix = (DeltaMatrix + MatrixXd::Constant(nDim,nDim,1.)) * range/2;
    DeltaMatrix = (DeltaMatrix + MatrixXd::Constant(nDim,nDim,LO));
    cout << "\n DeltaMatrix is: \n" << DeltaMatrix <<endl;
    
    DeltaMatrix.diagonal().array() = 10000000000;
    DeltaMatrix.leftCols(nVehicles) *= 10000000000;
    DeltaMatrix.bottomRows(nVehicles) *= 10000000000;
    DeltaMatrix.topRightCorner(nVehicles,nVehicles) = DeltaMatrix.bottomLeftCorner(nVehicles,nVehicles).eval();       
    //DeltaMatrix.row(1) += 100* DeltaMatrix.row(0);
    cout << "\n Updated DeltaMatrix is: \n" << DeltaMatrix << endl;   
    
  
    
}

/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc,char* argv[])
{    
    remove("results/Eloop.txt");
    remove("results/Elocal.txt");
    remove("results/Etask.txt");
    remove("VMatrix");
    remove("results/PMatrix.txt");
    remove("results/DeltaVMatrix.txt");
    remove("results/tauMat.txt");
    remove("results/leftVec.txt");
    remove("results/rightVec.txt");
    remove("hist");
    remove("V");
    
   
         
    Gnuplot gp;
    Gnuplot gp2;
        
    clock_t tStart = clock();
    //readInput(); //Read the inputs - nVehicles,nTasks,DeltaMatrix,TVec from file/cmd input/generate random
    getVMatrix(); //Initialize the VMatrix
    //generateRandom();
    readFile();

    
    std::ofstream outfile3 ("results/tauMat.txt");
    TauMatrix = (DeltaMatrix).colwise() + TVec;
    outfile3 << TauMatrix << std::endl;
    cout << "\n TauMatrix is: \n" << TauMatrix << endl;
    
    cout << "\n kT_start is "<< kT_start << endl;
    cout << "\n kT_stop is "<< kT_stop << endl;
    //cout << "\n kT_swfac is "<< kT_swfac << endl;
    cout << "\n kT_fac is "<< kT_fac << endl;
    cout << "\n gamma is "<< g << endl;
    
    cout << "\n ///////////////////////////////////////////////////////////////////// " << endl;

    NN_algo();    //NN_algo - updating equations
    
    //thresholding the solution
    for (int i = 0; i < nDim; i++)
        for (int j = 0; j < nDim; j++)
        {
            if(NonNormUpdatedVMatrix(i,j) > 0.9)
                NonNormUpdatedVMatrix(i,j) = 1;
            else if (NonNormUpdatedVMatrix(i,j) < 0.1)
                NonNormUpdatedVMatrix(i,j) = 0;
            else (NonNormUpdatedVMatrix(i,j) = NonNormUpdatedVMatrix(i,j));
        }
    cout << "\n The final solution without normalizing V is: \n" << endl;
    cout << NonNormUpdatedVMatrix;
    for (int i = 0; i < nDim; i++)
        for (int j = 0; j < nDim; j++)
        {
            if(VMatrix(i,j) > 0.8)
                VMatrix(i,j) = 1;
            else if (VMatrix(i,j) < 0.2)
                VMatrix(i,j) = 0;
            else (VMatrix(i,j) = VMatrix(i,j));
        }
    cout << "\n The final solution after V normalisation is: \n" << endl;
    cout << VMatrix;
    cout << "\n DetlaMatrix is: \n" << endl;
    cout << DeltaMatrix;
    
    gp << "N = `awk 'NR==2 {print NF}' VMatrix` \n";
    gp << "unset key \n";
    gp << "plot for [i=2:N] 'VMatrix' using 1:i with linespoints" << endl;
   
//     gp2 << "N = `awk 'NR==2 {print NF}' V` \n";
//     gp2 << "unset key \n";
//     gp2 << "plot for [i=2:N] 'V' using 1:i with linespoints" << endl;
   std::ofstream outfile1 ("tVec.txt");
    std::ofstream outfile2 ("deltaMat.txt");
    
    outfile1 << TVec << std::endl;
    outfile1.close();
    outfile2 << DeltaMatrix << std::endl;
    outfile2.close();
    cout << "\n Annealing done \n" << endl;
    //compare with solutions.cpp
    displaySolution();//Printing out the solution
    printf("\n Total computational time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
}
