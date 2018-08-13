#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <stdio.h>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <bitset>
#include <limits>
#include <fstream>
#include <math.h>
#include <iomanip> // needed for setw(int)
#include <cstdlib>
#include <Eigen/Dense>
#include <Eigen/LU> 
#include <cmath>
#include <time.h>
#include <limits>

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
double costValue;

Eigen::VectorXd cTime;
Eigen::VectorXd vdVecL;
Eigen::VectorXd vdVecR; 
Eigen::VectorXd rightVec;
Eigen::VectorXd leftVec;
Eigen::MatrixXd vDeltaR;
Eigen::MatrixXd vDeltaL;
Eigen::VectorXd valVec;
Eigen::MatrixXd DeltaMatrix;
Eigen::VectorXd TVec;
Eigen::VectorXd sub;
Eigen::MatrixXd I;
Eigen::VectorXd checkTimeVec;

void displaySolution(Eigen::MatrixXd vMat)
{
    std::string veh_alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    std::string task_alpha = "abcdefghijklmnopqrstuvwxyz";
    std::string solStrA;
    std::string solStrB;
    
    int indx = 0;
    int indxB = 0;
    int checkTime;
    cTime = VectorXd(nVehicles);
    sub = VectorXd::Ones(nVehicles);

    for (int i = 0; i < nVehicles; i++)
        {
            for (int j = 0; j < nDim; j++)
            {if (vMat(i,j)==1)
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
                    {if (vMat(indx,j)==1)
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
    
    cout << "\nThis corresponds to the following routing:\n";
    cout << "\n" << solStrA << endl;
    cout << "\n" << solStrB << endl;
    cout << "\n" <<checkTime << endl;
}
/////////////////////////////////////////////////////////////////////////////////////////////////

bool isValid(Eigen::MatrixXd x)
{
     Eigen::VectorXd sumr = VectorXd::Zero(nDim); 
     sumr = x.rowwise().sum();
     std::cout<<"\n"<<sumr<<endl;
     if (std::isnan(sumr(1)))
             {  
                 std::cout<<"\n has loops so invalid"<<endl;
                 return false;  
             }
    else
             {  
                 std::cout<<"\n no loops so valid"<<endl;
                 return true;
             }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

float calculateLR(Eigen::MatrixXd VMatrix, Eigen::MatrixXd PMatrix) //calculate LR from V and PMatrix
   {
        vDeltaL = VMatrix.transpose() * DeltaMatrix;
        //cout << "\n vDeltaL is: \n" << vDeltaL << endl;
        vdVecL = vDeltaL.diagonal();
        //cout << "\n vdVecL is: \n" << vdVecL << endl;
        leftVec = PMatrix.transpose() * (TVec + vdVecL);
        //cout << "\n leftVec is: \n" << leftVec << endl;

        vDeltaR = VMatrix * DeltaMatrix.transpose();
        //cout << "\n vDeltaR is: \n" << vDeltaR << endl;
        vdVecR = vDeltaR.diagonal();
        //cout << "\n vdVecR is: \n" << vdVecR << endl;
        rightVec = PMatrix * (TVec +vdVecR);
        //cout << "\n rightVec is: \n" << rightVec << endl;

        MatrixXf::Index imaxl, imaxr;
        double maxleftVec, maxrightVec;
        maxleftVec = leftVec.maxCoeff(&imaxl);
        maxrightVec = rightVec.maxCoeff(&imaxr);


        costValue = 0.5 * (leftVec(imaxl) + rightVec(imaxr));
        cout << "\n ///////////////////////////////////////////////////////////////////// " << endl;
        
        return costValue;
   }


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char* argv[])
{
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
    
    std::vector<int> v; 
    double costValueBest = 10000000000; 
   
    for(int i=0;i<rDim;++i)
        {v.push_back(i*1);}              
    
    Eigen::MatrixXd subMat = MatrixXd::Zero(rDim,rDim);
    Eigen::MatrixXd vMat = MatrixXd::Zero(nDim,nDim);
    Eigen::MatrixXd vMatBest = MatrixXd::Zero(nDim,nDim);
    Eigen::MatrixXd vMatInv = MatrixXd::Zero(nDim,nDim);
    Eigen::MatrixXd I = MatrixXd::Identity(nDim,nDim);
    Eigen::VectorXd L(nDim);
    Eigen::VectorXd R(nDim);

    for (int j=0;j<v.size();++j)
        {
            for (int i=0;i<rDim;++i)
                {
                    subMat(i,v[j,i]) = 1;
                }
        }
                
    std::cout<<"\n"<<subMat<<endl;
    vMat.block(0,nVehicles,rDim,rDim) = subMat;
    vMat.topRightCorner(nVehicles,nVehicles) *=0;
    vMat.diagonal().array() = 0;
    std::cout<<"\n"<<vMat<<endl;

    vMatInv = (I - vMat).inverse();
    std::cout<<"\n vMatInv is \n"<< vMatInv << endl;
    if (isValid(vMatInv))
        {
            costValue = calculateLR(vMat, vMatInv);
            if (costValue < costValueBest)
               {vMatBest = vMat;
               costValueBest = costValue;}
        }
    int iteration = 1;
    int indexA =0; int indexB = 0;
    kT = kT_start;
         int FLAG = 1;

    while(FLAG != 0)
      {
          cout << "\n" << iteration << " ITERATION STARTING" << endl;
          cout << "\n kT is " << kT << endl;
          cout<<"\n vMat is \n"<< vMat << endl;
              STEP: 
                  int ran = rand() % 2;
                 if (ran == 1) //change rows to create newvMat
                 {
                     cout << "Row swap" << endl;
                     std::vector<int> rowVector;
                     for (int i=0; i<(nVehicles+nTasks); ++i) 
                     {rowVector.push_back(i);}
                     std::random_shuffle (rowVector.begin(), rowVector.end() );
                     for (std::vector<int>::iterator it=rowVector.begin(); it!=rowVector.end(); ++it)

                     for (int j=0;j<nDim;j++)
                     {
                     if (vMat(rowVector[0],j)==1)
                        {indexA = j;
                        }
                     }
                     vMat(rowVector[0],indexA) = 0;   

                     for (int j=0;j<nDim;j++)
                     {
                     if (vMat(rowVector[1],j)==1)
                        {indexB = j;
                        }
                     }
                     vMat(rowVector[1],indexB) = 0;

                     vMat(rowVector[1],indexA) = 1;
                     vMat(rowVector[0],indexB) = 1;
                 }
                 else
                 {
                     cout << "Column swap" << endl; //change rows to create newvMat
                     std::vector<int> colVector;
                     for (int i=nVehicles; i<nDim; ++i) 
                     {colVector.push_back(i);}
                     std::random_shuffle (colVector.begin(), colVector.end() );
                     for (std::vector<int>::iterator it=colVector.begin(); it!=colVector.end(); ++it)

                     for (int j=0;j<nDim;j++)
                     {
                     if (vMat(j,colVector[0])==1)
                        {indexA = j;
                        }
                     }
                     vMat(indexA,colVector[0]) = 0;   

                     for (int j=0;j<nDim;j++)
                     {
                     if (vMat(j,colVector[1])==1)
                        {indexB = j;
                        }
                     }
                     vMat(indexB,colVector[1]) = 0;

                     vMat(indexA,colVector[1]) = 1;
                     vMat(indexB,colVector[0]) = 1;
                 }
            std::cout<<"\n new vMat is \n"<< vMat << endl;
            vMatInv = (I - vMat).inverse();
            std::cout<<"\n vMatInv is \n"<< vMatInv << endl;
            if (isValid(vMatInv))
                {
                costValue = calculateLR(vMat, vMatInv);
                cout << "\n new costValue is: " << costValue << endl;
                cout << "\n Best costValue is: " << costValueBest << endl;
                if (costValue < costValueBest)
                    {vMatBest = vMat;
                    costValueBest = costValue;
                    cout << "\n better solution found " << endl;
                    }
                kT *= kT_fac;
                cout << "\n new kT is: " << kT << endl;
                cout << "\n" << iteration << " ITERATION DONE" << endl;
                iteration = iteration + 1;
                if (kT < kT_stop) 
                FLAG = 0;
                }
            else
            {   cout << "\n Invalid solution found.. Rechecking again.. " << endl;
                goto STEP;}
     }    
     cout << costValueBest << endl;
     cout << vMatBest << endl;
     displaySolution(vMatBest);//find solutions for each matrix...
     cout << "\nDeltaMatrix is: \n" << DeltaMatrix << endl;    
     cout << "\nTVec is: \n" << TVec << endl;    
}
