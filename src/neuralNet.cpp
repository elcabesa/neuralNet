#include <iostream>
#include <random>
#include <vector>


#include "labeledExample.h"
#include "model.h"
#include "linear.h"
#include "relu.h"
#include "denseLayer.h"
#include "sparse.h"
#include "dense.h"
#include "parallelDenseLayer.h"
#include "inputSet.h"
#include "gradientDescend.h"


int main() {
    std::cout << "NeuralNET" << std::endl;
    InputSet inSet;
    inSet.generate();

    Model m;
    m.addLayer(std::make_unique<ParallelDenseLayer>(2, 2, 5, ActivationFactory::create(ActivationFactory::type::linear)));
    m.addLayer(std::make_unique<DenseLayer>(10,10, ActivationFactory::create(ActivationFactory::type::relu)));
    m.addLayer(std::make_unique<DenseLayer>(10,10, ActivationFactory::create(ActivationFactory::type::relu)));
    m.addLayer(std::make_unique<DenseLayer>(10, 1, ActivationFactory::create(ActivationFactory::type::linear)));
    m.randomizeParams();
    std::cout<<"randomized params"<<std::endl;
    
    GradientDescend gd(m, inSet, 1e6, 1e-4, 1.0, 1e4);
    
    gd.train();
    

    std::cout<<"-------------------------"<<std::endl;

    {
        std::cout<<"serialize"<<std::endl;
        std::ofstream nnFile;
        nnFile.open ("nn.txt");
        m.serialize(nnFile);
        nnFile.close();
        std::cout<<"done"<<std::endl;
    }
    std::cout<<"randomize Params"<<std::endl;
    m.randomizeParams();
    std::cout<<"final total loss: " <<m.calcTotalLoss(inSet.trainSet())<<" "<<m.calcTotalLoss(inSet.validationSet())<<std::endl;
    std::cout<<"reload"<<std::endl;
    {
        std::cout<<"deserialize"<<std::endl;
        std::ifstream nnFile;
        nnFile.open ("nn.txt");
        if(m.deserialize(nnFile)){
             std::cout<<"done"<<std::endl;
        }else {
             std::cout<<"FAIL"<<std::endl;
        }
        nnFile.close();
    }
    std::cout<<"final total loss: " <<m.calcTotalLoss(inSet.trainSet())<<" "<<m.calcTotalLoss(inSet.validationSet())<<std::endl;
    
}


