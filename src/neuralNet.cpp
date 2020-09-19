#include <iostream>
#include <random>
#include <vector>

#include "optparse/cxxopts.hpp"
#include "labeledExample.h"
#include "model.h"
#include "linear.h"
#include "relu.h"
#include "denseLayer.h"
#include "sparse.h"
#include "dense.h"
#include "parallelDenseLayer.h"
#include "diskInputSet.h"
#include "diskInputSet2.h"
#include "memoryInputSet.h"
#include "gradientDescend.h"

void getInputSet(DiskInputSet2& inSet) {
    std::cout<<"read testset"<<std::endl;
    inSet.generate();
    
    /*for( unsigned int i = 0; i<10; ++i){
        for(auto x: inSet.batch()) {
            std::cout<<x->label()<<std::endl;
        }
        std::cout<<"---------------"<<std::endl;
    }*/
    
    std::cout<<"done"<<std::endl;
}

Model createModel() {
    std::cout<<"creating model"<<std::endl;
    
    Model m;
    m.addLayer(std::make_unique<ParallelDenseLayer>(2, 40960, 256, ActivationFactory::create(ActivationFactory::type::linear),1.0/sqrt(6)));
    m.addLayer(std::make_unique<DenseLayer>(512,32, ActivationFactory::create(ActivationFactory::type::relu)));
    m.addLayer(std::make_unique<DenseLayer>(32,32, ActivationFactory::create(ActivationFactory::type::relu)));
    m.addLayer(std::make_unique<DenseLayer>(32, 1, ActivationFactory::create(ActivationFactory::type::linear)));
    
    std::cout<<"done"<<std::endl;
    
    return m;
}


int main(int argc, const char*argv[]) {
    
    
    cxxopts::Options options("Neural Net", "Vajolet Neural net trainer");
    options.add_options()
        ("help", "Print help")
        ("p,passes", "traiing passes", cxxopts::value<unsigned int>()->default_value("30000"))
        ("e,eta", "eta", cxxopts::value<double>()->default_value("1e-4"))
        ("r,regularization", "regularization value", cxxopts::value<double>()->default_value("1.0"))
        ("b,beta", "rmsprop beta", cxxopts::value<double>()->default_value("0.9"))
        ("randomize", "randomize model parmeters", cxxopts::value<bool>()->default_value("false"))
        ("n,nPath", "weight file path", cxxopts::value<std::string>()->default_value("./nn-start.txt"))
        ("s,batchSize", "batchSize", cxxopts::value<unsigned int>()->default_value("30"))
        ("print", "print validation error")
    ;
    
    auto result = options.parse(argc, argv);
    
    if (result.count("help"))
    {
      std::cout << options.help({"", "Group"}) << std::endl;
      exit(0);
    }
    std::cout << "NeuralNET" << std::endl;
    
    DiskInputSet2 inSet("./TESTSET", 81920,result["batchSize"].as<unsigned int>());
    getInputSet(inSet);
    
    auto m = createModel();
    
    if (result["randomize"].as<bool>()) {
        m.randomizeParams();
        std::cout<<"randomized params"<<std::endl;
        std::cout<<"serialize"<<std::endl;
        std::ofstream nnFile;
        nnFile.open ("nn-start.txt");
        m.serialize(nnFile);
        nnFile.close();
        std::cout<<"done"<<std::endl;
    }
    
    if (result.count("nPath")) {
        std::cout<<"deserialize"<<std::endl;
        std::ifstream nnFile;
        nnFile.open (result["nPath"].as<std::string>());
        if(m.deserialize(nnFile)){
             std::cout<<"done"<<std::endl;
        }else {
             std::cout<<"FAIL"<<std::endl;
             exit(-1);
        }
        nnFile.close();
    }
    
    if (result.count("print"))
    {
        m.calcAvgLoss(inSet.validationSet(), true);
        exit(0);
    }

    GradientDescend gd(m,
                       inSet,
                       result["passes"].as<unsigned int>(),
                       result["eta"].as<double>(),
                       result["regularization"].as<double>(),
                       result["beta"].as<double>()
                      );
    
    gd.train();

    {
        std::cout<<"serialize"<<std::endl;
        std::ofstream nnFile;
        nnFile.open ("nn-final.txt");
        m.serialize(nnFile);
        nnFile.close();
        std::cout<<"done"<<std::endl;
    }
}


