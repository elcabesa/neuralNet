#include <cmath>
#include <iostream>
#include <random>
#include <set>

#include "labeledExample.h"
#include "pedoneCheck.h"
#include "model.h"

/*#define STAT_SIZE (63)
#define DECIMATION (1000000)
#define SCALING (4.0)
void updateStat(double val) {
    static uint64_t stat[STAT_SIZE] = {0}; 
    static int counter = 0;

    int v = std::round(SCALING * val);
    v += STAT_SIZE/2;
    v = std::min(std::max(v , 0), STAT_SIZE -1);
    ++stat[v];

    if(++counter >= DECIMATION) {
        counter = 0;
        std::cout<<"-------------"<<std::endl;
        for(int i = 0; i < STAT_SIZE; ++i) {
            std::cout<< ((i- STAT_SIZE/2) / SCALING * 2.58) <<" "<<stat[i]<<std::endl;
        }
    }
}*/

PedoneCheck::PedoneCheck(Model* model):_NNUEmodel(model) {
    std::cout<<"PEDONE CHECK"<<std::endl;
}

void PedoneCheck::caricaPesi() {
    {
        //std::cout<<"loadLayer1"<<std::endl;
        auto& l1 = _NNUEmodel->getLayer(0);
        
        if(l1.bias().size() != SizeLayer1 / 2) { std::cout<<"ERRORE DIMENSIONI BIAS LAYER1"<<std::endl;}
        //std::cout<<"bias"<<std::endl;
        for (unsigned int i = 0; i < l1.bias().size(); ++i) {
            _pesi.biasLayer1[i] = l1.bias()[i];
        }

        if(l1.weight().size() != MaxInput * SizeLayer1 / 2) { std::cout<<"ERRORE DIMENSIONI WEIGHT LAYER1"<<std::endl;}

        //std::cout<<"weight"<<std::endl;
        for (unsigned int in = 0; in < MaxInput; ++in) {
            for (unsigned int out = 0; out < SizeLayer1 / 2; ++out) {
                _pesi.pesiLayer1[in][out] = l1.weight()[out + in * SizeLayer1 / 2];
                /*if(in == 19020 && out == 2) {
                    std::cout<<"weight "<<_pesi.pesiLayer1[in][out]<<std::endl;
                }*/
            }
        }
    }
    //-------------------------------------------------------
    {
        //std::cout<<"loadLayer2"<<std::endl;
        auto& l2 = _NNUEmodel->getLayer(1);

        if(l2.bias().size() != SizeLayer2) { std::cout<<"ERRORE DIMENSIONI BIAS LAYER2"<<std::endl;}
        for (unsigned int i = 0; i < l2.bias().size(); ++i) {
            _pesi.biasLayer2[i] = l2.bias()[i];
        }

        if(l2.weight().size() != SizeLayer2 * SizeLayer1) { std::cout<<"ERRORE DIMENSIONI WEIGHT LAYER2"<<std::endl;}

        for (unsigned int in = 0; in < SizeLayer1; ++in) {
            for (unsigned int out = 0; out < SizeLayer2; ++out) {
                _pesi.pesiLayer2[in][out] = l2.weight()[out + in * SizeLayer2];
            }
        }
    }
    //-------------------------------------------------------
    {
        //std::cout<<"loadLayer3"<<std::endl;
        auto& l3 = _NNUEmodel->getLayer(2);

        if(l3.bias().size() != SizeLayer3) { std::cout<<"ERRORE DIMENSIONI BIAS LAYER3"<<std::endl;}
        for (unsigned int i = 0; i < l3.bias().size(); ++i) {
            _pesi.biasLayer3[i] = l3.bias()[i];
        }

        if(l3.weight().size() != SizeLayer2 * SizeLayer3) { std::cout<<"ERRORE DIMENSIONI WEIGHT LAYER3"<<std::endl;}

        for (unsigned int in = 0; in < SizeLayer2; ++in) {
            for (unsigned int out = 0; out < SizeLayer3; ++out) {
                _pesi.pesiLayer3[in][out] = l3.weight()[out + in * SizeLayer3];
            }
        }
    }
    //-------------------------------------------------------
    {
        //std::cout<<"loadLayer4"<<std::endl;
        auto& l4 = _NNUEmodel->getLayer(3);

        if(l4.bias().size() != 1) { std::cout<<"ERRORE DIMENSIONI BIAS LAYER4"<<std::endl;}
        _pesi.biasOutput = l4.bias()[0];

        if(l4.weight().size() != SizeLayer3) { std::cout<<"ERRORE DIMENSIONI WEIGHT LAYER4"<<std::endl;}

        for (unsigned int in = 0; in < SizeLayer3; ++in) {
            _pesi.pesiOutput[in] = l4.weight()[in];
        }
    }
}

void PedoneCheck::caricaInput(LabeledExample& b) {
    _input.NInput[0] = 0;
    _input.NInput[1] = 0;

    auto& f = b.features();
    unsigned int num = f.getElementNumber();

    for(unsigned int i = 0; i < num; ++i) {
        //std::cout<<f.getElementFromIndex(i).first<< " " <<f.getElementFromIndex(i).second<<std::endl;
        unsigned int idx = f.getElementFromIndex(i).first;
        if (idx < MaxInput) {
            _input.InputCompatto[0][_input.NInput[0]++] = idx;
        } else {
            _input.InputCompatto[1][_input.NInput[1]++] = idx - MaxInput;
        }
    } 
    if(_input.NInput[0] != _input.NInput[1]) {
        std::cout<<"ERRORE INPUT"<<std::endl;
    }
}

void PedoneCheck::calcolaRisRete() {
    // calcola layer 1
    for(unsigned int out = 0; out < SizeInputLayer; ++out) {
        _risRete.ris32Layer1[out] = _pesi.biasLayer1[out];
        _risRete.ris32Layer1[out + SizeInputLayer] = _pesi.biasLayer1[out];
    }
    for (unsigned int pl = 0; pl < 2; ++pl) {
        for (unsigned int in = 0; in < _input.NInput[pl]; ++in) {
            for (unsigned int out = 0; out < SizeInputLayer; ++out) {
                _risRete.ris32Layer1[(pl * SizeInputLayer) + out] += _pesi.pesiLayer1[_input.InputCompatto[pl][in]][out];
            }
        }
    }
    /*_varCalL1_1.addValue(_risRete.ris32Layer1[0]);
    _varCalL1_2.addValue(_risRete.ris32Layer1[128]);
    _varCalL1_3.addValue(_risRete.ris32Layer1[256]);
    _varCalL1_4.addValue(_risRete.ris32Layer1[384]);*/
    for (unsigned int out = 0; out < SizeLayer1; ++out) {
        _risRete.risLayer1[out] = _ActivationTrain(_risRete.ris32Layer1[out]);
    }

    // calcola layer 2
    for (unsigned int out = 0; out < SizeLayer2; ++out) {
        double ris = _pesi.biasLayer2[out];
        for (unsigned int in = 0; in < SizeLayer1; ++in) {
            ris += _risRete.risLayer1[in] * _pesi.pesiLayer2[in][out];
        }
        /*if(out==0)_varCalL2_1.addValue(ris);
        if(out==8)_varCalL2_2.addValue(ris);
        if(out==16)_varCalL2_3.addValue(ris);
        if(out==24)_varCalL2_4.addValue(ris);*/
        _risRete.risLayer2[out] = _ActivationTrain(ris/* / 64.0*/);
    }

    // calcola layer 3
    for (unsigned int out = 0; out < SizeLayer3; ++out) {
        double ris = _pesi.biasLayer3[out];
        for (unsigned int in = 0; in < SizeLayer2; ++in) {
            ris += _risRete.risLayer2[in] * _pesi.pesiLayer3[in][out];
        }
        /*if(out==0)_varCalL3_1.addValue(ris);
        if(out==8)_varCalL3_2.addValue(ris);
        if(out==16)_varCalL3_3.addValue(ris);
        if(out==24)_varCalL3_4.addValue(ris);*/
        _risRete.risLayer3[out] = _ActivationTrain(ris/* / 64.0*/);
    }

    // calcola output
    _risRete.output = _pesi.biasOutput;
    for (unsigned int in = 0; in < SizeLayer3; ++in) {
        _risRete.output += _risRete.risLayer3[in] * _pesi.pesiOutput[in];
    }
#ifdef COMPARE
    //std::cout<<"check layer 1"<<std::endl;
    for(unsigned int i = 0; i < SizeLayer1; ++i) {
        if(_risRete.risLayer1[i] != _NNUEmodel->getLayer(0).getOutput(i)) {
            std::cout<<"propagate layer 1 error"<<std::endl;
            std::cout << i << " " << _risRete.risLayer1[i] << " " <<_NNUEmodel->getLayer(0).getOutput(i)<<std::endl;
            exit(0);
        }
    }
    //std::cout<<"check layer 2"<<std::endl;
    for(unsigned int i = 0; i < SizeLayer2; ++i) {
        if(_risRete.risLayer2[i] != _NNUEmodel->getLayer(1).getOutput(i)) {
            std::cout<<"propagate layer 2 error"<<std::endl;
            std::cout << i << " " << _risRete.risLayer2[i] << " " <<_NNUEmodel->getLayer(1).getOutput(i)<<std::endl;
            exit(0);
        }
    }
    //std::cout<<"check layer 3"<<std::endl;
    for(unsigned int i = 0; i < SizeLayer3; ++i) {
        if(_risRete.risLayer3[i] != _NNUEmodel->getLayer(2).getOutput(i)) {
            std::cout<<"propagate layer 3 error"<<std::endl;
            std::cout << i << " " << _risRete.risLayer3[i] << " " <<_NNUEmodel->getLayer(2).getOutput(i)<<std::endl;
            exit(0);
        }
    }
    //std::cout<<"check output"<<std::endl;
    if(_risRete.output !=  _NNUEmodel->getLayer(3).getOutput(0)) {
        std::cout<<"propagate layer 4 error"<<std::endl;
        std::cout << _risRete.output << " " <<_NNUEmodel->getLayer(3).getOutput(0)<<std::endl;
        exit(0);
    }
#endif
    /*_varCalOut.addValue(_risRete.output);
    if(++_decCounter> _decimation) {
        _decCounter = 0;
        std::cout<<"--------------------"<<std::endl;
        _varCalL1_1.print();
        _varCalL1_2.print();
        _varCalL1_3.print();
        _varCalL1_4.print();
        std::cout<<"-----"<<std::endl;
        _varCalL2_1.print();
        _varCalL2_2.print();
        _varCalL2_3.print();
        _varCalL2_4.print();
        std::cout<<"-----"<<std::endl;
        _varCalL3_1.print();
        _varCalL3_2.print();
        _varCalL3_3.print();
        _varCalL3_4.print();
        std::cout<<"-----"<<std::endl;
        _varCalOut.print();
        std::cout<<"-------grad--------"<<std::endl;
        _varCalL1_1bias.print();
        _varCalL1_2bias.print();
        _varCalL1_3bias.print();
        _varCalL1_4bias.print();
        std::cout<<"-----"<<std::endl;
        _varCalL2_1bias.print();
        _varCalL2_2bias.print();
        _varCalL2_3bias.print();
        _varCalL2_4bias.print();
        std::cout<<"-----"<<std::endl;
        _varCalL3_1bias.print();
        _varCalL3_2bias.print();
        _varCalL3_3bias.print();
        _varCalL3_4bias.print();
        std::cout<<"-----"<<std::endl;
        _varCalOutbias.print();
        std::cout<<"-------gradweight--------"<<std::endl;
        for(unsigned int i = 0; i < SizeLayer3; ++i) {
            _varCalOutWeight[i].print();
        }
    }*/

}

double PedoneCheck::calcGrad(double label) {
    double err;
    // output
    _biasGrad.BiasGradOutput = _risRete.output - label;
    /*Outbias.addValue(_biasGrad.BiasGradOutput);*/
#ifdef COMPARE
    //std::cout<<"gradOutput = "<< _biasGrad.BiasGradOutput<<std::endl;
    if (_NNUEmodel->getLayer(3).getBiasSumGradient(0) != _biasGrad.BiasGradOutput) {
        std::cout << "grad out error "<< _NNUEmodel->getLayer(3).getBiasSumGradient(0)<<" "<<_biasGrad.BiasGradOutput<<std::endl;
        exit(-1);
    }
#endif

    // check layer3
    for (unsigned int in = 0; in < SizeLayer3; ++in) {
        err = _biasGrad.BiasGradOutput * _pesi.pesiOutput[in];
        _biasGrad.BiasGradLayer3[in] = err * _Derivative(_risRete.risLayer3[in])/* / 64.0*/;
        /*if(in==0)_varCalL3_1bias.addValue(_biasGrad.BiasGradLayer3[in]);
        if(in==8)_varCalL3_2bias.addValue(_biasGrad.BiasGradLayer3[in]);
        if(in==16)_varCalL3_3bias.addValue(_biasGrad.BiasGradLayer3[in]);
        if(in==24)_varCalL3_4bias.addValue(_biasGrad.BiasGradLayer3[in]);*/
    }
#ifdef COMPARE
    for(unsigned int i = 0; i < SizeLayer3; ++i) {
        if (_NNUEmodel->getLayer(2).getBiasSumGradient(i) != _biasGrad.BiasGradLayer3[i]) {
            std::cout<<_risRete.risLayer3[i]<<std::endl;
            std::cout << "grad layer3 error "<< _NNUEmodel->getLayer(2).getBiasSumGradient(i)<<" "<<_biasGrad.BiasGradLayer3[i]<<std::endl;
            std::cout << _NNUEmodel->getLayer(2).getBiasSumGradient(i) - _biasGrad.BiasGradLayer3[i]<<std::endl;
            exit(-1);
        }
    }
#endif

    // check layer2
    for (unsigned int in = 0; in < SizeLayer2; ++in) {
        err = 0.0;
        for (unsigned int out = 0; out < SizeLayer3; ++out) {
            err += _biasGrad.BiasGradLayer3[out] * _pesi.pesiLayer3[in][out];
        }
        _biasGrad.BiasGradLayer2[in] = err * _Derivative(_risRete.risLayer2[in])/* / 64.0*/;
        /*if(in==0)_varCalL2_1bias.addValue(_biasGrad.BiasGradLayer2[in]);
        if(in==8)_varCalL2_2bias.addValue(_biasGrad.BiasGradLayer2[in]);
        if(in==16)_varCalL2_3bias.addValue(_biasGrad.BiasGradLayer2[in]);
        if(in==24)_varCalL2_4bias.addValue(_biasGrad.BiasGradLayer2[in]);*/
    }
#ifdef COMPARE
    for(unsigned int i = 0; i < SizeLayer2; ++i) {
        if (_NNUEmodel->getLayer(1).getBiasSumGradient(i) != _biasGrad.BiasGradLayer2[i]) {
            std::cout<<_risRete.risLayer2[i]<<std::endl;
            std::cout << "grad layer2 error "<< _NNUEmodel->getLayer(1).getBiasSumGradient(i)<<" "<<_biasGrad.BiasGradLayer2[i]<<std::endl;
            std::cout << _NNUEmodel->getLayer(1).getBiasSumGradient(i) - _biasGrad.BiasGradLayer2[i]<<std::endl;
            exit(-1);
        }
    }
#endif

    // check layer1
    for (unsigned int in = 0; in < SizeInputLayer; ++in) {
        err = 0.0;
        for (unsigned int out = 0; out < SizeLayer2; ++out) {
            err += _biasGrad.BiasGradLayer2[out] * _pesi.pesiLayer2[in][out];
        }
        _biasGrad.BiasGradLayer1[in] = err * _Derivative(_risRete.risLayer1[in]);
    }
    for (unsigned int in = 0; in < SizeInputLayer; ++in) {
        err = 0.0;
        for (unsigned int out = 0; out < SizeLayer2; ++out) {
            err += _biasGrad.BiasGradLayer2[out] * _pesi.pesiLayer2[in + SizeInputLayer][out];
        }
        _biasGrad.BiasGradLayer1[in + SizeInputLayer] = err * _Derivative(_risRete.risLayer1[in + SizeInputLayer]);
    }

    /*_varCalL1_1bias.addValue(_biasGrad.BiasGradLayer1[0] + _biasGrad.BiasGradLayer1[0 + SizeInputLayer]);
    _varCalL1_2bias.addValue(_biasGrad.BiasGradLayer1[128] + _biasGrad.BiasGradLayer1[128 + SizeInputLayer]);
    _varCalL1_3bias.addValue(_biasGrad.BiasGradLayer1[256] + _biasGrad.BiasGradLayer1[256 + SizeInputLayer]);
    _varCalL1_4bias.addValue(_biasGrad.BiasGradLayer1[384] + _biasGrad.BiasGradLayer1[384 + SizeInputLayer]);*/
#ifdef COMPARE
    for(unsigned int i = 0; i < SizeLayer1 / 2; ++i) {
        if (_NNUEmodel->getLayer(0).getBiasSumGradient(i) != _biasGrad.BiasGradLayer1[i] + _biasGrad.BiasGradLayer1[i + SizeInputLayer]) {
            std::cout<<_risRete.risLayer1[i]<<std::endl;
            std::cout << "grad layer1 error "<< i <<" "<<_NNUEmodel->getLayer(0).getBiasSumGradient(i)<<" "<<_biasGrad.BiasGradLayer1[i] + _biasGrad.BiasGradLayer1[i+256]<<std::endl;
            std::cout << _NNUEmodel->getLayer(0).getBiasSumGradient(i) - _biasGrad.BiasGradLayer1[i] - _biasGrad.BiasGradLayer1[i + SizeInputLayer]<<std::endl;
            exit(-1);
        }
    }
#endif

    return std::pow(_risRete.output - label, 2.0) / 2.0;
}

void PedoneCheck::updateweights(double l) {
    // output layer
    _pesi.biasOutput -= l * _biasGrad.BiasGradOutput;
#ifdef COMPARE
    if(_pesi.biasOutput != _NNUEmodel->getLayer(3).bias()[0]) {
        std::cout<<"error updating bias of output layer"<<std::endl;
        std::cout<<_pesi.biasOutput<<std::endl;
        std::cout<<_NNUEmodel->getLayer(3).bias()[0]<<std::endl;
        std::cout<<(_pesi.biasOutput - _NNUEmodel->getLayer(3).bias()[0])*1e7<<std::endl;
        exit(-1);
    }
#endif

    for( unsigned int out = 0; out < SizeLayer3; ++out) {
        _pesi.pesiOutput[out] -= l * (_biasGrad.BiasGradOutput * _risRete.risLayer3[out]);
        /*_varCalOutWeight[out].addValue(_biasGrad.BiasGradOutput * _risRete.risLayer3[out]);*/
#ifdef COMPARE
        if(_pesi.pesiOutput[out] != _NNUEmodel->getLayer(3).weight()[out]) {
            std::cout<<"error updating pesi of output layer"<<std::endl;
            std::cout<<_pesi.pesiOutput[out] <<std::endl;
            std::cout<<_NNUEmodel->getLayer(3).weight()[out]<<std::endl;
            std::cout<<(_pesi.pesiOutput[out] - _NNUEmodel->getLayer(3).weight()[out])*1e7<<std::endl;
            exit(-1);
        }
#endif
    }

    // layer3
    for (unsigned int out = 0; out < SizeLayer3; ++out) {
        for (unsigned int in = 0; in < SizeLayer2; ++in) {
            _pesi.pesiLayer3[in][out] -= l * (_biasGrad.BiasGradLayer3[out] * _risRete.risLayer2[in]);
#ifdef COMPARE
            if(_pesi.pesiLayer3[in][out] != _NNUEmodel->getLayer(2).weight()[out + in * SizeLayer3]) {
                std::cout<<"error updating pesi of layer3"<<std::endl;
                std::cout<<_pesi.pesiLayer3[in][out] <<std::endl;
                std::cout<< _NNUEmodel->getLayer(2).weight()[out + in * SizeLayer3]<<std::endl;
                exit(-1);
            }
#endif
        } 
        _pesi.biasLayer3[out] -= l * _biasGrad.BiasGradLayer3[out];
#ifdef COMPARE
        if(_pesi.biasLayer3[out] != _NNUEmodel->getLayer(2).bias()[out]) {
            std::cout<<"error updating bias of layer3"<<std::endl;
            std::cout<<_pesi.biasLayer3[out] <<std::endl;
            std::cout<< _NNUEmodel->getLayer(2).bias()[out]<<std::endl;
            exit(-1);
        }
#endif

    }

    // layer2
    for (unsigned int out = 0; out < SizeLayer2; ++out) {
        for (unsigned int in = 0; in < SizeLayer1; ++in) {
            _pesi.pesiLayer2[in][out] -= l * (_biasGrad.BiasGradLayer2[out] * _risRete.risLayer1[in]);
#ifdef COMPARE
            if(_pesi.pesiLayer2[in][out] != _NNUEmodel->getLayer(1).weight()[out + in * SizeLayer2]) {
                std::cout<<"error updating pesi of layer2"<<std::endl;
                std::cout<<_pesi.pesiLayer2[in][out] <<std::endl;
                std::cout<< _NNUEmodel->getLayer(1).weight()[out + in * SizeLayer2]<<std::endl;
                exit(-1);
            }
#endif
        } 
        _pesi.biasLayer2[out] -= l * _biasGrad.BiasGradLayer2[out];  
#ifdef COMPARE
        if(_pesi.biasLayer2[out] != _NNUEmodel->getLayer(1).bias()[out]) {
            std::cout<<"error updating bias of layer2"<<std::endl;
            std::cout<<_pesi.biasLayer2[out] <<std::endl;
            std::cout<< _NNUEmodel->getLayer(1).bias()[out]<<std::endl;
            exit(-1);
        } 
#endif
    }

    //layer1
    std::set<uint16_t> features;
    for (unsigned int pl = 0; pl < 2; ++pl) {
        for (unsigned int in = 0; in < _input.NInput[pl]; ++in) {
            uint16_t idx = _input.InputCompatto[pl][in];
            features.insert(idx);
            for (unsigned int out = 0; out < SizeInputLayer; ++out) {
                //_pesi.pesiLayer1[idx][out] -= l * _biasGrad.BiasGradLayer1[out + pl * SizeInputLayer];
                _biasGrad.WeightGradLayer1[idx][out] += _biasGrad.BiasGradLayer1[out + pl * SizeInputLayer];
            }
        }
    }

    for(auto f:features) {
        for (unsigned int out = 0; out < SizeInputLayer; ++out) {
            _pesi.pesiLayer1[f][out] -= l * _biasGrad.WeightGradLayer1[f][out];
            _biasGrad.WeightGradLayer1[f][out] = 0.0;
        }
    }
    
    for (unsigned int out = 0; out < SizeInputLayer; ++out) {
        _pesi.biasLayer1[out] -= l * (_biasGrad.BiasGradLayer1[out] + _biasGrad.BiasGradLayer1[out + SizeInputLayer]);
#ifdef COMPARE
        if(_pesi.biasLayer1[out] != _NNUEmodel->getLayer(0).bias()[out]) {
            std::cout<<"error updating bias of layer1"<<std::endl;
            std::cout<<_pesi.biasLayer1[out] <<std::endl;
            std::cout<< _NNUEmodel->getLayer(0).bias()[out]<<std::endl;
            std::cout<< _NNUEmodel->getLayer(0).bias()[out + SizeInputLayer]<<std::endl;
            exit(-1);
        }
#endif
    }
#ifdef COMPARE
    for (unsigned int in = 0; in < MaxInput; ++in) {
        for (unsigned int out = 0; out < SizeLayer1 / 2; ++out) {
            if(_pesi.pesiLayer1[in][out] != _NNUEmodel->getLayer(0).weight()[in * SizeLayer1 / 2 + out]) {
                std::cout<<in<< " "<<out<<std::endl;
                std::cout<<"error updating pesi of layer1"<<std::endl;
                std::cout<<"PEDONE "<<_pesi.pesiLayer1[in][out] <<std::endl;
                std::cout<<"VAJO "<<_NNUEmodel->getLayer(0).weight()[in * SizeLayer1 / 2 + out]<<std::endl;
                std::cout<<std::abs(_pesi.pesiLayer1[in][out] - _NNUEmodel->getLayer(0).weight()[in * SizeLayer1 / 2 + out])*1e5<<std::endl;
                exit(-1);
            }
        }
    }
#endif
}


double PedoneCheck::_ActivationTrain(double x) {
    x = std::max(x, 0.01 * x);
    x = std::min(x, 1.0 + 0.01 * (x - 1.0));
    return x;
}

double PedoneCheck::_Derivative(double x) {
    return (x > 0.0 && x < 1.0) ? 1.0 : 0.01;
}
