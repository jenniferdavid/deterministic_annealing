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
//int perm = fact(rDim);
int validMat;
bool sol;

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

/*
int fact(int n)
{
    if (n>1)
        return n*fact(n-1);
    else
        return 1;
}*/

float solutionCheck(Eigen::MatrixXd vMat, Eigen::MatrixXd vMatInv, bool sol)
{
     
    valVec = VectorXd(2*nVehicles);
    vdVecL = VectorXd(nDim);
    vdVecR = VectorXd(nDim);
    rightVec = VectorXd(nDim);
    leftVec = VectorXd(nDim);

 //  Computing L and R  
    
        vDeltaL = vMat.transpose() * DeltaMatrix;
        vdVecL = vDeltaL.diagonal();
        leftVec = vMatInv.transpose() * (TVec + vdVecL);
                
        vDeltaR = vMat * DeltaMatrix.transpose();
        vdVecR = vDeltaR.diagonal();
        rightVec = vMatInv * (TVec + vdVecR);
                
        valVec << leftVec.tail(nVehicles),rightVec.head(nVehicles);
        cout << "\nvalVec is: \n" << valVec << endl;    

    if (sol == true)
    {
        VectorXf::Index maxV;
        float maxValue = valVec.maxCoeff(&maxV);
        cout << "\nmaxValue is: \n" << maxValue << endl;    
        return maxValue;
    }
    
    else
    {
        float meanValue = valVec.mean();
        cout << "\nmeanValue is: \n" << meanValue << endl;    
        return meanValue;
    }
                
}

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

bool isValid(Eigen::MatrixXd x)
{
     Eigen::VectorXd sumr = VectorXd::Zero(nDim); 
     sumr = x.rowwise().sum();
     //std::cout<<"\n"<<sumr<<endl;
     if (std::isnan(sumr(1)))
             {  
                // std::cout<<"\n has loops so invalid"<<endl;
                 return false;  
             }
    else
             {  
               // std::cout<<"\n no loops so valid"<<endl;
                 return true;
             }
}

int main(int argc, const char* argv[])
{
    nVehicles = atoi(argv[1]);
    nTasks = atoi(argv[2]);
    //std::cout<<"\n argc is "<<argc<<endl;

    if (argc != 5)
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
    
    std::vector<int> v; 
    std::vector<Eigen::MatrixXd> MVector;
    std::vector <float> costValue; 
    std::vector<Eigen::MatrixXd> PVector; 
    validMat = 0;
   
    for(int i=0;i<rDim;++i)
        {v.push_back(i*1);}
        do  {
                Eigen::MatrixXd subMat = MatrixXd::Zero(rDim,rDim);
                Eigen::MatrixXd vMat = MatrixXd::Zero(nDim,nDim);
                Eigen::MatrixXd vMatInv = MatrixXd::Zero(nDim,nDim);

                for (int i=0;i<rDim;++i)
                    {std::cout<<"\n"<<v[i]<<endl;}
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

                int check = 0;
                for (int i = 0; i < nDim; i++)
                {                    
                    for (int j = 0; j< nDim; j++)
                    {   
                        if (vMat(i,j)==1)
                        {
                          check = check+1;
                        }
                    }
                }
                
                if (check == rDim)
                {
                vMatInv = (I - vMat).inverse();
                std::cout<<"\n vMatInv is \n"<< vMatInv << endl;
                     if (isValid(vMatInv))
                        {
                        validMat = validMat+1;
                        MVector.push_back(vMat);
                        PVector.push_back(vMatInv);
                        }
                }
          }
       while(std::next_permutation(v.begin(),v.end()));
          
       std::cout<<"\n********************************************************************************"<<endl;
       if (!strcmp(argv[4],"-max"))
       {sol = true;}
       else if (!strcmp(argv[4],"-avg"))
       {sol = false;}
       else
       {return(0);}
        
       for (int i = 0;i <validMat;i++)
       {
       std::cout<<"\nValid solution no "<< i+1 << endl;
       std::cout<<"\n"<<MVector[i]<<endl;
       std::cout<<"\nCorresponding Propagator is " << endl;
       std::cout<<"\n"<<PVector[i]<<endl;
       displaySolution(MVector[i]);//find solutions for each matrix...
       float maxValue = solutionCheck(MVector[i],PVector[i],sol);//finds lowest and highest in each vector...
       costValue.push_back(maxValue);
       std::cout<<"\n********************************************************************************"<<endl;
       }

       std::cout<<"********************************************************************************"<<endl;
       std::cout << "\nList of all costs for valid solutions: \n";
       for (int i = 0;i < costValue.size();i++)
       {std::cout << costValue[i] << endl;}      
       std::cout << "\nNo of valid solutions for " <<nVehicles<< " vehicles and "<<nTasks<<" tasks is " << costValue.size() << std::endl;
       auto biggest = std::max_element(std::begin(costValue), std::end(costValue));
       std::cout << "\nMax cost is " << *biggest << std::endl;
       auto smallest = std::min_element(std::begin(costValue), std::end(costValue));
       std::cout << "\nMin cost is " << *smallest << " which is solution no " << std::distance(std::begin(costValue), smallest+1) << std::endl;
       
       displaySolution(MVector[std::distance(std::begin(costValue), smallest)]);//find solutions for each matrix...

       cout << "\nDeltaMatrix is: \n" << DeltaMatrix << endl;    
       cout << "\nTVec is: \n" << TVec << endl;    
       std::cout<<"\n*****************************************************************************"<<endl;
       std::cout<<"*****************************************************************************"<<endl;
}
