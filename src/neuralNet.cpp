#include <iostream>
#include <random>
#include <vector>


#include "labeledExample.h"
#include "model.h"
#include "linear.h"
#include "relu.h"
#include "denseLayer.h"
#include "sparse.h"
#include "parallelDenseLayer.h"

double function(double x1, double x2, double x3, double x4) {
    return x1+ x2*x2 -0.2*x3*x4;
}

std::vector<std::shared_ptr<LabeledExample>> generateInput() {
    std::cout<<"create input "<<std::flush;
    
    std::vector<std::shared_ptr<LabeledExample>> input;
    for(unsigned int x1 = 0; x1<10; ++x1) {
        for(unsigned int x2 = 0; x2<10; ++x2) {
            for(unsigned int x3 = 0; x3<10; ++x3) {
                for(unsigned int x4 = 0; x4<10; ++x4) {
                    std::vector<double> inVec = {double(x1), double(x2),double(x3),double(x4)};
                    std::shared_ptr<Input> in(new SparseInput(inVec));
                    std::shared_ptr<LabeledExample> le(new LabeledExample(std::move(in),function(x1,x2,x3,x4)));
                    input.push_back(std::move(le));
                }
            }
        }
    }
    
    std::cout<<"DONE"<<std::endl;
    std::cout<<"input size "<<input.size()<<std::endl;
    
    std::cout<<"shuffle input "<<std::flush;
    std::random_device rng;
    std::shuffle(std::begin(input), std::end(input), rng);
    std::cout<<"DONE"<<std::endl;
    
    return input;
}

int main() {
    std::cout << "NeuralNET" << std::endl;
    
    auto trainSet = generateInput();
    
    std::size_t t1 = trainSet.size() * 0.9;

    
    //split
    std::vector<std::shared_ptr<LabeledExample>> validationSet(trainSet.begin() + t1, trainSet.end() );
    trainSet.erase(trainSet.begin() + t1, trainSet.end() );
    
    std::cout<<"splitted set"<<std::endl;
    
    Model m;
    m.addLayer(std::make_unique<ParallelDenseLayer>(2, 2, 4, ActivationFactory::create(ActivationFactory::type::linear)));
    m.addLayer(std::make_unique<DenseLayer>(8,8, ActivationFactory::create(ActivationFactory::type::relu)));
    m.addLayer(std::make_unique<DenseLayer>(8,8, ActivationFactory::create(ActivationFactory::type::relu)));
    m.addLayer(std::make_unique<DenseLayer>(8, 1, ActivationFactory::create(ActivationFactory::type::linear)));
    
    std::cout<<"-------------------------"<<std::endl;
    for(int i =0; i< 1; ++i){
        m.randomizeParams();
        std::cout<<"randomized params"<<std::endl;
        std::cout<<m.train(1e6, 1e-5, trainSet, validationSet, 1.0, 1)<<std::endl;
    }
    /*std::cout<<"-------------------------"<<std::endl;
    for(int i =0; i< 10; ++i){
        m.randomizeParams();
        std::cout<<m.train(10000, 0.0000001, trainSet, validationSet, 1.0, 10)<<std::endl;
    }
    std::cout<<"-------------------------"<<std::endl;
    for(int i =0; i< 10; ++i){
        m.randomizeParams();
        std::cout<<m.train(10000, 0.000001, trainSet, validationSet, 0.99998, 10)<<std::endl;
    }*/
    //m.printParams();
}


