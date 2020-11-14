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

Model createModel(double stdDev, double scaling) {
    std::cout<<"creating model"<<std::endl;
    
    Model m(scaling);
    m.addLayer(std::make_unique<ParallelDenseLayer>(2, 40960, 256, ActivationFactory::create(Activation::type::relu), 16, 1.0, stdDev));
    m.addLayer(std::make_unique<DenseLayer>(512,32, ActivationFactory::create(Activation::type::relu), 32, 64.0));
    m.addLayer(std::make_unique<DenseLayer>(32,32, ActivationFactory::create(Activation::type::relu), 32, 64.0));
    m.addLayer(std::make_unique<DenseLayer>(32, 1, ActivationFactory::create(Activation::type::linear), 32, 1.0));
    
    std::cout<<"done"<<std::endl;
    
    return m;
}


int main(int argc, const char*argv[]) {
    
    
    cxxopts::Options options("Neural Net", "Vajolet Neural net trainer");
    options.add_options()
        
        ("b,beta", "rmsprop beta", cxxopts::value<double>()->default_value("0.9"))
        ("d,stdDev", "std dev", cxxopts::value<double>()->default_value("0.0"))
        ("e,eta", "eta", cxxopts::value<double>()->default_value("1e-4"))
        ("r,regularization", "regularization value", cxxopts::value<double>()->default_value("1.0"))
        ("n,nPath", "weight file path", cxxopts::value<std::string>()->default_value("./nn-start.txt"))
        ("p,passes", "traiing passes", cxxopts::value<unsigned int>()->default_value("30000"))
        ("s,batchSize", "batchSize", cxxopts::value<unsigned int>()->default_value("30"))
        ("c,decimation", "decimation", cxxopts::value<unsigned int>()->default_value("10000"))
        ("q,quantization", "quantization #", cxxopts::value<unsigned int>()->default_value("50"))
        ("i,labelScaling", "labelScaling", cxxopts::value<unsigned int>()->default_value("30000"))

        ("help", "Print help")
        ("print", "print validation error", cxxopts::value<unsigned int>()->default_value("30"))
        ("randomize", "randomize model parmeters", cxxopts::value<bool>()->default_value("false"))
        ("stat", "printNetworkStats", cxxopts::value<bool>()->default_value("false"))
        ("rmsprop", "true ->rmsprop, false ->gd", cxxopts::value<bool>()->default_value("false"))
        
    ;
    
    auto result = options.parse(argc, argv);
    
    if (result.count("help"))
    {
      std::cout << options.help({"", "Group"}) << std::endl;
      exit(0);
    }
    std::cout << "NeuralNET" << std::endl;
    
    DiskInputSet2 inSet("./TESTSET", 81920, result["batchSize"].as<unsigned int>());
    getInputSet(inSet);
    
    // for the first layer let's use a std dev of sqrt(2/active_input)
    // we assume an average of 20 pieces on the board
    double stdDev = sqrt(2.0 /(20 * (result["batchSize"].as<unsigned int>())));
    if (result.count("stdDev")) {
        stdDev = result["stdDev"].as<double>();
    }
    auto m = createModel(stdDev, result["labelScaling"].as<unsigned int>());
    
    if (result["randomize"].as<bool>()) {
        m.randomizeParams();
        std::cout<<"randomized params"<<std::endl;
        std::cout<<"serialize"<<std::endl;
        std::ofstream nnFile;
        nnFile.open ("nn-start.txt");
        m.serialize(nnFile);
        nnFile.close();
        std::cout<<"done"<<std::endl;
        return 0;
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
        m.setQuantization(true);
        m.calcAvgLoss(inSet.validationSet(), true, result["print"].as<unsigned int>());
        m.setQuantization(false);
        return 0;
    }

    if (result.count("stat"))
    {
        for(;;) {
            for(auto& le: inSet.batch()) {
                m.forwardPass((*le).features(), true);
            }
        }
    }

    GradientDescend gd(m,
                       inSet,
                       result["passes"].as<unsigned int>(),
                       result["eta"].as<double>(),
                       result["regularization"].as<double>(),
                       result["beta"].as<double>(),
                       result["quantization"].as<unsigned int>(),
                       result["rmsprop"].as<bool>()
                      );
    
    gd.train(result["decimation"].as<unsigned int>());

    {
        std::cout<<"serialize"<<std::endl;
        std::ofstream nnFile;
        nnFile.open ("nn-final.txt");
        m.serialize(nnFile);
        nnFile.close();
        std::cout<<"done"<<std::endl;
    }
}


