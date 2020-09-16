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
#include "diskInputSet.h"
#include "memoryInputSet.h"
#include "gradientDescend.h"


int main(int argc, char*argv[]) {
    double learnRate = 1e-4;
    double regularization = 1.0;
    double beta = 0.9;
    unsigned int passes = 30000;
    
    if(argc == 5) {
        passes = atoi(argv[1]);
        learnRate = atof(argv[2]);
        regularization = atof(argv[3]);
        beta = atof(argv[4]);
        std::cout<<"passes "<<passes<<std::endl;
        std::cout<<"learnRate "<<learnRate<<std::endl;
        std::cout<<"regularization "<<regularization<<std::endl;
        std::cout<<"beta "<<beta<<std::endl;
    }
    
    std::cout << "NeuralNET" << std::endl;
    
    std::cout<<"read testset"<<std::endl;
    DiskInputSet inSet("./TESTSET", 81920);
    inSet.generate();
    std::cout<<"done"<<std::endl;
    /*auto &valSet = inSet.validationSet();
    for(auto& ex: valSet) {
        std::cout<<ex->label()<<" - ";
        ex->features().print();
    }
    
    for(unsigned int i =0; i<10; ++i) {
        auto &valSet = inSet.batch();
        for(auto& ex: valSet) {
            std::cout<<ex->label()<<" - ";
            ex->features().print();
        }
    }
    return 0;*/
    //inSet.printStatistics();
    //return 0;
    
    std::cout<<"creating model"<<std::endl;
    Model m;
    m.addLayer(std::make_unique<ParallelDenseLayer>(2, 40960, 256, ActivationFactory::create(ActivationFactory::type::linear),1.0/sqrt(6)));
    m.addLayer(std::make_unique<DenseLayer>(512,32, ActivationFactory::create(ActivationFactory::type::relu)));
    m.addLayer(std::make_unique<DenseLayer>(32,32, ActivationFactory::create(ActivationFactory::type::relu)));
    m.addLayer(std::make_unique<DenseLayer>(32, 1, ActivationFactory::create(ActivationFactory::type::linear)));
    std::cout<<"done"<<std::endl;
    /*double maxValue = 1.e30;
    for( int i = 0; i< 100; ++i ) {
        m.randomizeParams();
        double loss = m.calcAvgLoss(inSet.validationSet());
        std::cout<<"final total loss: " <<loss<<std::endl;    
        if( loss < maxValue) {testSet = generateTestSetList()
            maxValue =loss;
            std::cout<<"NEW MAX VALUE: " <<maxValue<<std::endl;  
            
                std::cout<<"serialize"<<std::endl;
                std::ofstream nnFile;
                nnFile.open ("nn.txt");
                m.serialize(nnFile);
                nnFile.close();
                std::cout<<"done"<<std::endl;
            
        }
    }
    //m.printParams();
    std::cout<<"randomized params"<<std::endl;*/
    
    std::cout<<"reload"<<std::endl;
    {
        std::cout<<"deserialize"<<std::endl;
        std::ifstream nnFile;
        nnFile.open ("nn-start.txt");
        if(m.deserialize(nnFile)){
             std::cout<<"done"<<std::endl;
        }else {
             std::cout<<"FAIL"<<std::endl;
        }
        nnFile.close();
        //m.printParamsStats();
    }
    //exit(0);

    GradientDescend gd(m, inSet, passes, learnRate, regularization, beta);
    
    gd.train();
    

    std::cout<<"-------------------------"<<std::endl;

    {
        std::cout<<"serialize"<<std::endl;
        std::ofstream nnFile;
        nnFile.open ("nn-final.txt");
        m.serialize(nnFile);
        nnFile.close();
        std::cout<<"done"<<std::endl;
    }
    //std::cout<<"randomize Params"<<std::endl;
    //m.randomizeParams();*/
    /*std::cout<<"final total loss: " <<m.calcAvgLoss(inSet.validationSet())<<std::endl;
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
    std::cout<<"final total loss: " << m.calcAvgLoss(inSet.validationSet())<<std::endl;*/
    
}


