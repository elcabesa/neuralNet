#include <cassert>
#include <iostream>
#include <random>
#include <filesystem>
#include <fstream>
#include <algorithm>

#include "diskInputSet.h"
#include "labeledExample.h"
#include "dense.h"

DiskInputSet::DiskInputSet(const std::string path, unsigned int inputSize): _path(path), _n(0), _inputSize(inputSize){}
DiskInputSet::~DiskInputSet(){}

void DiskInputSet::generate() {
    for(const auto& entry : std::filesystem::directory_iterator(_path)) {
        std::string path = entry.path();
        if (path.find("testset") != std::string::npos) {
            _testSets.push_back(path);
            //std::cout<<path<<std::endl;
        }
        if (path.find("validationset") != std::string::npos) {
            _verificationSet = path;
            //std::cout<<path<<std::endl;
        }
    }
}


const std::vector<std::shared_ptr<LabeledExample>>& DiskInputSet::validationSet() const {
    _validationSet.clear();
    //std::cout<<_verificationSet<<std::endl;
    std::ifstream ss(_verificationSet);
    if(ss.is_open())
    {
        bool finish = false;
        while(!finish){
            auto ex = parseLine(ss, finish);
            if(finish) {
                //std::cout<<"FINITO"<<std::endl;
                break;
            }
            _validationSet.push_back(std::make_shared<LabeledExample>(ex));
            //std::cout<<"LABELED EXAMPLE"<<std::endl;
            //ex.features().print();
            //std::cout<<ex.label()<<std::endl;
        }
        //std::cout<<"----------"<<std::endl;
        ss.close();
    }
    
    return _validationSet;
    
    
}

const std::vector<std::shared_ptr<LabeledExample>>& DiskInputSet::batch()const {
    
    if(_n >= _testSets.size()) {
        _n = 0;
        std::random_shuffle(_testSets.begin(), _testSets.end());
    }

    _batch.clear();
    //std::cout<<_testSets[_n]<<std::endl;
    std::ifstream ss(_testSets[_n]);
    if(ss.is_open())
    {
        bool finish = false;
        while(!finish){
            auto ex = parseLine(ss, finish);
            if(finish) {
                //std::cout<<"FINITO"<<std::endl;
                break;
            }
            _batch.push_back(std::make_shared<LabeledExample>(ex));
            //std::cout<<"LABELED EXAMPLE"<<std::endl;
            //ex.features().print();
            //std::cout<<ex.label()<<std::endl;
        }
        //std::cout<<"----------"<<std::endl;
        ss.close();
    }
    ++_n;
    return _batch;
}

LabeledExample DiskInputSet::parseLine(std::ifstream& ss, bool& finish) const {
    std::vector<unsigned int> inVec = {};
    std::shared_ptr<Input> in(new SparseInput(_inputSize, inVec));
    LabeledExample empty(in,0);
    char temp = ss.get();
    if(ss.eof()) {finish = true; return empty;} 
    if(temp != '{') {std::cout<<"missing {"<<std::endl;finish = true; return empty;}
    auto features = getFeatures(ss);
    if(ss.get() != '{') {std::cout<<"missing {"<<std::endl;finish = true; return empty;}
    double label = getLabel(ss);
    if(ss.get() != '\n') {std::cout<<"missing CR"<<std::endl;finish = true; return empty;}
    
    return LabeledExample(features, label);
}

std::shared_ptr<Input> DiskInputSet::getFeatures(std::ifstream& ss) const {
    std::string x;
    std::vector<unsigned int> inVec = {};
    char temp;
    do {
        temp = ss.get();
        while(temp != ',' && temp != '}') {
            x += temp;
            temp = ss.get();
        }
        //std::cout<<x<<std::endl;
        assert((unsigned int)std::stoi(x)<_inputSize);
        inVec.push_back(std::stoi(x));
        x= "";
    }while(temp != '}');
    std::shared_ptr<Input> feature(new SparseInput(_inputSize, inVec));
    return feature;
}

double DiskInputSet::getLabel(std::ifstream& ss) const {
    std::string x;
    char temp = ss.get();
    while(temp != '}') {
        x += temp;
        temp = ss.get();
    }
    return std::stod(x);
}
