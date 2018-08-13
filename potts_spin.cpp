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

int nVehicles;
int nTasks;
int nDim;
int rDim;
double kT_start;
double kT_stop;
double kT_fac;
double g;
double kT;
double lk;
double kappa;
double beta = 1;

static const double small = 1e-15;
static const double onemsmall = 1 - small;
static const double lk0 = 1/small - 1; 

////////////////////////////////////////////////////////////////////////////////////////

Eigen::MatrixXd VMatrix;
Eigen::MatrixXd DeltaMatrix;
Eigen::MatrixXd PMatrix;
Eigen::MatrixXd I = MatrixXd::Identity(nDim,nDim); 
Eigen::MatrixXd P;
Eigen::MatrixXd NonNormUpdatedVMatrix;
Eigen::MatrixXd UpdatedVMatrix;
Eigen::MatrixXd UpdatedPMatrix;
Eigen::MatrixXd E_task; 
Eigen::MatrixXd E_loop; 
Eigen::MatrixXd E_local; 
Eigen::MatrixXd vDeltaR;
Eigen::MatrixXd vDeltaL;
    
Eigen::VectorXd TVec; 
Eigen::VectorXd vdVecL(nDim);
Eigen::VectorXd vdVecR(nDim);
Eigen::VectorXd rightVec(nDim);
Eigen::VectorXd leftVec(nDim);
Eigen::VectorXd sumr; 
Eigen::VectorXd sumc;
Eigen::VectorXd cTime;
Eigen::VectorXd checkTimeVec;
Eigen::VectorXd sub;

////////////////////////////////////////////////////////////////////////////////////////////////
  
  Eigen::MatrixXd normalisation(Eigen::MatrixXd VMatrix) //this function normalises a matrix - makes into doubly stochastic matrix
  {
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
            {goto LABEL0;}
       else
            {cout << "\n Normalised VMatrix is: \n" << VMatrix << endl;
            cout << "Sum of VMatrix after row normalisation \n" << VMatrix.rowwise().sum() << endl;
            cout << "Sum of VMatrix after col normalisation \n" << VMatrix.colwise().sum() << endl;}
       return VMatrix;
  }
  
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  void getVMatrix () //initialize VMatrix
  {
      cout << "\n ///////////////////////////////////////////////////////////////////// " << endl;
      cout << "\n No. of Vehicles M = " << nVehicles << endl;
      cout << "\n No. of Tasks N = " << nTasks << endl;
      cout << "\n Dimension of VMatrix is: " << nDim << endl;
      cout << "\n Total no of Vij: " << nDim*nDim << endl;
        
      VMatrix = MatrixXd::Zero(nDim,nDim);
      
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
      
      VMatrix = normalisation(VMatrix);
  }
  
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
/*Eigen::MatrixXd asyncRandomUpdate(Eigen::MatrixXd UpdatedVMatrix)
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
            
}*/
    
  Eigen::MatrixXd syncUpdate(Eigen::MatrixXd E_local, Eigen::MatrixXd UpdatedVMatrix) //Updating mean field equations (VMatrix) along row-wise
  {
        Eigen::MatrixXd E = MatrixXd::Zero(nDim,nDim);
       // E = ((-1*E_local)/kT);

        for (int i = 0; i < rDim; i++)
             {
               for (int j=nVehicles; j< nDim; j++)
                    {
                        E(i,j) = ((-1*(E_local(i,j) - (beta*(UpdatedVMatrix(i,j)))))/kT);
                        UpdatedVMatrix(i,j) = std::exp (E(i,j));
                    }  
             }
        cout << "\n Updating VMatrix is: (the numerator) \n" << UpdatedVMatrix << endl;
        
        Eigen::VectorXd sumv = UpdatedVMatrix.rowwise().sum();
        cout << "sumv is (my ref): \n " << sumv << endl;
        for (int i = 0; i < rDim; i++)
             {                    
               for (int j=nVehicles; j< nDim; j++)
                    {
                       UpdatedVMatrix(i,j) /= sumv(i);
                    }  
             }          
        UpdatedVMatrix.topRightCorner(nVehicles,nVehicles) *= 0;  
        cout << "\n Updating Row VMatrix is (complete update):  \n" << UpdatedVMatrix << endl;
        return UpdatedVMatrix;
    }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

   std::tuple<Eigen::VectorXd, Eigen::VectorXd> calculateLR(Eigen::MatrixXd VMatrix, Eigen::MatrixXd PMatrix) //calculate LR from V and PMatrix
   {
        vDeltaL = VMatrix.transpose() * DeltaMatrix;
        cout << "\n vDeltaL is: \n" << vDeltaL << endl;
        vdVecL = vDeltaL.diagonal();
        cout << "\n vdVecL is: \n" << vdVecL << endl;
        leftVec = PMatrix.transpose() * (TVec + vdVecL);
        cout << "\n leftVec is: \n" << leftVec << endl;

        vDeltaR = VMatrix * DeltaMatrix.transpose();
        cout << "\n vDeltaR is: \n" << vDeltaR << endl;
        vdVecR = vDeltaR.diagonal();
        cout << "\n vdVecR is: \n" << vdVecR << endl;
        rightVec = PMatrix * (TVec +vdVecR);
        cout << "\n rightVec is: \n" << rightVec << endl;

        return std::make_tuple(leftVec, rightVec);
   }

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

   std::tuple<int, int, float> maxLR(Eigen::VectorXd left, Eigen::VectorXd right) //calculate initial maximum of L and R
   {
        MatrixXf::Index imaxl, imaxr;
        double imaxleftVecInd, imaxrightVecInd;
        imaxleftVecInd = left.maxCoeff(&imaxl);
        imaxrightVecInd = right.maxCoeff(&imaxr);
        kappa = 0.5 * (left(imaxl) + right(imaxr));
        cout << "\n Kappa is: \n" << kappa << endl;
        cout << "\n ///////////////////////////////////////////////////////////////////// " << endl;
        return std::make_tuple(imaxl, imaxr, kappa);
    }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    Eigen::MatrixXd calculateE(Eigen::VectorXd left, Eigen::VectorXd right, Eigen::MatrixXd PMatrix) //calculate ELocal using L,R and PMatrix
    {
    for (int i = 0; i < nDim; i++)
        {
            for (int j=0; j< nDim; j++)
                {
                 double X = 0; double Y = 0;
                 double sumaa = 0; double sumbb=0;
                        for (int l=(nVehicles+nTasks); l<nDim ; l++)
                        {sumaa = sumaa + PMatrix(j,l);}
                        for (int l=0; l< nVehicles; l++)
                        {sumbb = sumbb + PMatrix(l,i);}
                        
                        /*cout << "\n leftVec(i) is: " << leftVec(i) << endl;
                        cout << "\n PPP(j,maxl) is: " << PPP(j,maxl) << endl;
                        cout << "\n righttVec(j) is: " << rightVec(j) << endl;
                        cout << "\n PPP(maxr,i) is: " << PPP(maxr,i) << endl;
                        
                        iX = ((ileftVec(i) + DeltaMatrix(i,j)) * PMatrix(j,imaxl));
                        iY = ((irightVec(j) + DeltaMatrix(i,j)) * PMatrix(imaxr,i));
                        /*cout << "\n X is: " << X << endl;
                        cout << "\n Y is: " << Y << endl;*/
                        
                        X = (left(i) + DeltaMatrix(i,j)) * sumaa;
                        Y = (right(i) + DeltaMatrix(i,j)) * sumbb;
                        E_task(i,j) = (0.5/nVehicles)*(X + Y);
                        
                        double lk = PMatrix(j,i) / PMatrix(i,i);	// the "zeroed" Pji
                            if (lk < onemsmall )
                                lk = lk/(1-lk); // => the resulting Pji for choice j
                            else
                                lk = lk0;  
                        E_loop(i,j) = lk;
                    }
         }
    E_task.leftCols(nVehicles) *= 10000000;
    E_task.bottomRows(nVehicles) *= 10000000;
    E_task.topRightCorner(nVehicles,nVehicles) *= 10000000;  //adding all the constraints for the vehicles and tasks  
    E_task.topRightCorner(nVehicles,nVehicles) = E_task.bottomLeftCorner(nVehicles,nVehicles).eval();       
    E_task.topLeftCorner(nVehicles,nVehicles) = E_task.bottomLeftCorner(nVehicles,nVehicles).eval();  
    E_task = E_task/kappa;
    E_loop.diagonal().array() = 10000000;
    cout << "\nE_loop is: \n" << E_loop << endl;
    cout << "\nE_task is: \n" << E_task << endl;

    E_local = (g * E_loop) + E_task;
    cout << "\n E_local is: \n" << E_local << endl;
    return E_local;
    }
    
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    Eigen::MatrixXd UpdateP(Eigen::MatrixXd UpdatedVMatrix, Eigen::MatrixXd VMatrix, Eigen::MatrixXd PMatrix, bool Pstate) //calculate Updated P
    {
        Eigen::MatrixXd P = MatrixXd::Zero(nDim,nDim);
        if (Pstate == true)
        {
            Eigen::MatrixXd dQ = MatrixXd::Zero(nDim,nDim);    
            Eigen::MatrixXd DeltaVMatrix = MatrixXd::Zero(nDim,nDim);
            Eigen::MatrixXd DeltaPMatrix = MatrixXd::Zero(nDim,nDim);
            
            cout << "\n  Previous VMatrix is \n" << VMatrix << endl;
            DeltaVMatrix = (UpdatedVMatrix - VMatrix);
            cout << "\n Delta V Matrix is \n" << DeltaVMatrix << endl;
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
        }
    else 
        {
            P = (I-UpdatedVMatrix);
            UpdatedPMatrix = P.inverse();
            cout << "\n Updated PMatrix using exact inverse is:\n" << UpdatedPMatrix << endl;
        }           
    return UpdatedPMatrix;
    }

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void NN_algo()
    {             
        //calculating initial values L,R, energy for V and P
         Eigen::MatrixXd E_local = MatrixXd::Zero(nDim,nDim);
         Eigen::MatrixXd UpdatedVMatrix = MatrixXd::Zero(nDim,nDim);
         Eigen::MatrixXd UpdatedPMatrix = MatrixXd::Zero(nDim,nDim); 
         E_task = MatrixXd::Zero(nDim,nDim);
         E_loop = MatrixXd::Zero(nDim,nDim);
         std::ofstream outfile3 ("VMatrix");
         std::ofstream outfile4 ("NormVMatrix");
            
         PMatrix = (I - VMatrix).inverse(); //calculate initial P
         calculateLR(VMatrix, PMatrix); // calculate LR for initial values
         maxLR(leftVec,rightVec); //calculate kappa from max of LR
         E_local = calculateE(leftVec,rightVec,PMatrix); //calculate initial local energy E
         
         cout << "\n initial E_local is: \n" << E_local << endl;
         cout << "\n Initial VMatrix is: \n" << VMatrix << endl;
         cout << "\n Initial PMatrix is: \n" << PMatrix << endl;
         cout << "\n ////////////////////////////////////////////////////////////////////////// " << endl;
        
         int iteration = 1;
         int FLAG = 1;
         kT = kT_start;

         while (FLAG != 0) //iteration starting 
         {	
            cout << "\n" << iteration << " ITERATION STARTING" << endl;
            cout << "\n kT is " << kT << endl;
            cout << "\n PMatrix before is \n" << PMatrix << endl;
            cout << "\n VMatrix before is \n" << VMatrix << endl;
            
            UpdatedVMatrix = syncUpdate(E_local, VMatrix); //calculate updatedVMatrix using ELocal
            //UpdatedVMatrix = asyncRandomUpdate(VMatrix);
            //UpdatedVMatrix = asyncAvgUpdate(VMatrix);

            cout << "\n VMatrix now is \n" << UpdatedVMatrix << endl;
            outfile3 << "\n" << iteration << "\t";
            outfile4 << "\n" << iteration << "\t";

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
            for (int i = 0; i < nDim; i++)
                {                    
                    for (int j = 0; j< nDim; j++)
                    {   
                        outfile3 << UpdatedVMatrix(i,j) << "\t";
                    }
                }
            
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
            UpdatedVMatrix = normalisation(UpdatedVMatrix);
            cout << "\n col sum after final normalisation is \n" << UpdatedVMatrix.colwise().sum() << endl;
            cout << "\n row sum after final normalisation is \n" << UpdatedVMatrix.rowwise().sum() << endl;
            cout << "\n Final row and column normalised UpdatedVMatrix is \n" << UpdatedVMatrix << endl;
            cout << "\n NORMALISATION ENDS \n" << endl;
            cout << "\n /////////////////////////////////////////////////////////////////////////////////////// " << endl;
            
            for (int i = 0; i < nDim; i++)
                {                    
                    for (int j = 0; j< nDim; j++)
                    {   
                        outfile4 << UpdatedVMatrix(i,j) << "\t";
                    }
                }
            
            UpdatedPMatrix = UpdateP(UpdatedVMatrix, VMatrix, PMatrix, false);
            calculateLR(UpdatedVMatrix, UpdatedPMatrix);                
            E_local = calculateE(leftVec,rightVec,UpdatedPMatrix);
            cout << "\n E_local is: \n " << E_local << endl;
                
            kT *= kT_fac;
            cout << "\n new kT is: " << kT << endl;
            cout << "\n" << iteration << " ITERATION DONE" << endl;
            iteration = iteration + 1;
            VMatrix = UpdatedVMatrix;
            PMatrix = UpdatedPMatrix;
            cout << "\n /*/*/*/*/*/*/*/**/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/ " << endl;
            if (kT < kT_stop) 
            FLAG = 0;
            cout << "\n ///////////////////////////////////////////////////////////////////// " << endl;
         }
 }  
 
    void displaySolution() //parses the solution
    {
        std::string veh_alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        std::string task_alpha = "abcdefghijklmnopqrstuvwxyz";
        std::string solStrA;
        std::string solStrB;

        int indx = 0;
        int indxB = 0;
        int checkTime;
        cTime = Eigen::VectorXd(nVehicles);
        sub = VectorXd::Ones(nVehicles);

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
        checkTimeVec = VectorXd(2*nVehicles);
        checkTimeVec << cTime, cTime + (nVehicles * sub);
        VectorXf::Index maxE;
        checkTime = checkTimeVec.maxCoeff(&maxE);

        cout << "This corresponds to the following routing:\n";
        cout << "\n" <<solStrA << endl;
        cout << "\n" <<solStrB << endl;
        cout << "\n" <<checkTime << endl;
    }
    
/////////////////////////////////////////////////////////////////////////////////////////////////

    int main(int argc, const char* argv[])
    {
    remove("results/Eloop.txt");
    remove("results/Elocal.txt");
    remove("results/Etask.txt");
    remove("VMatrix");
    remove("results/PMatrix.txt");
    remove("results/DeltaVMatrix.txt");
    remove("results/leftVec.txt");
    remove("results/rightVec.txt");
    remove("hist");
    remove("V");
    
    nVehicles = atoi(argv[1]);
    nTasks = atoi(argv[2]);
    kT_start = atoi(argv[5]);
    kT_stop = atof(argv[6]);
    kT_fac = atof(argv[7]);
    g = atoi(argv[8]);
    
    if (argc != 9)
    {
        std::cout<<"\n Less input options... exiting "<<endl;
        return(0);
    }

    std::cout<<"\n nVehicles is "<<nVehicles<<endl;
    std::cout<<"\n nTasks is "<<nTasks<<endl;

    nDim = 2*nVehicles + nTasks;; 
    rDim = nVehicles + nTasks;

    std::cout<<"\n nDim is "<<nDim<<endl;
    std::cout<<"\n rDim is "<<rDim<<endl;

    DeltaMatrix = MatrixXd::Ones(nDim,nDim);
    I = MatrixXd::Identity(nDim,nDim); //identity matrix
    TVec = VectorXd(nDim);
           
    if (!strcmp(argv[3],"-random"))
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
     }
     
    else if (!strcmp(argv[3],"-read"))
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
        {cout <<"file not open"<<endl;}
    cout << "\n TVec is: \n" << TVec <<endl;
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
        {cout <<"file not open"<<endl;}
    cout << "\n DeltaMatrix is: \n" << DeltaMatrix <<endl;
    }
    
    else 
    {cout << "\n Invalid option: " << argv[3] << "      exiting....\n";
                return(0);
    }
    
    DeltaMatrix.diagonal().array() = 10000000000;
    DeltaMatrix.leftCols(nVehicles) *= 10000000000;
    DeltaMatrix.bottomRows(nVehicles) *= 10000000000;
    DeltaMatrix.topRightCorner(nVehicles,nVehicles) = DeltaMatrix.bottomLeftCorner(nVehicles,nVehicles).eval();       
    //DeltaMatrix.row(1) += 100* DeltaMatrix.row(0);
    
    cout << "\n Updated DeltaMatrix is: \n" << DeltaMatrix << endl;    
    std::ofstream outfile1 ("tVec.txt");
    std::ofstream outfile2 ("deltaMat.txt");
    outfile1 << TVec << std::endl;
    outfile2 << DeltaMatrix << std::endl;
    outfile1.close();
    outfile2.close();
    
    cout << "\n kT_start is "<< kT_start << endl;
    cout << "\n kT_stop is "<< kT_stop << endl;
    cout << "\n kT_fac is "<< kT_fac << endl;
    cout << "\n gamma is "<< g << endl;
    cout << "\n ///////////////////////////////////////////////////////////////////// " << endl;
    
    Gnuplot gp;
    Gnuplot gp2;
        
    clock_t tStart = clock();
    getVMatrix(); 
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
            if(VMatrix(i,j) > 0.7)
                VMatrix(i,j) = 1;
            else if (VMatrix(i,j) < 0.3)
                VMatrix(i,j) = 0;
            else (VMatrix(i,j) = VMatrix(i,j));
        }
    cout << "\n For the given DetlaMatrix: \n" << endl;
    cout << DeltaMatrix;
    cout << "\n The final solution after V normalisation is: \n" << endl;
    cout << VMatrix;

    gp << "N = `awk 'NR==2 {print NF}' VMatrix` \n";
    gp << "unset key \n";
    gp << "plot for [i=2:N] 'VMatrix' using 1:i with linespoints" << endl;
    gp2 << "N = `awk 'NR==2 {print NF}' NormVMatrix` \n";
    gp2 << "unset key \n";
    gp2 << "plot for [i=2:N] 'NormVMatrix' using 1:i with linespoints" << endl;

    cout << "\n Annealing done \n" << endl;
    //compare with solutions.cpp
    displaySolution();//Printing out the solution
    printf("\n Total computational time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
}
