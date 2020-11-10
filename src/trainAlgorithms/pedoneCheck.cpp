
#include <iostream>

#include "labeledExample.h"
#include "pedoneCheck.h"
#include "model.h"

PedoneCheck::PedoneCheck(Model* model):_NNUEmodel(model) {
    std::cout<<"PEDONE CHECK"<<std::endl;
}

void PedoneCheck::caricaPesi() {
    //-------------------------------------------------------
    {
        //std::cout<<"loadLayer1"<<std::endl;
        auto& l1 = _NNUEmodel->getLayer(0);
        
        if(l1.bias().size() != SizeLayer1/2) { std::cout<<"ERRORE DIMENSIONI BIAS LAYER1"<<std::endl;}
        //std::cout<<"bias"<<std::endl;
        for (unsigned int i = 0; i < l1.bias().size(); ++i) {
            _pesi.biasLayer1[i] = l1.bias()[i];
        }

        if(l1.weight().size() != MaxInput * SizeLayer1 / 2) { std::cout<<"ERRORE DIMENSIONI WEIGHT LAYER1"<<std::endl;}

        //std::cout<<"weight"<<std::endl;
        for (unsigned int in = 0; in < MaxInput; ++in) {
            for (unsigned int out = 0; out < SizeLayer1/2; ++out) {
                _pesi.pesiLayer1[in][out] = l1.weight()[out + in * SizeLayer1/2];
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

        if(l2.bias().size() != SizeLayer23) { std::cout<<"ERRORE DIMENSIONI BIAS LAYER2"<<std::endl;}
        for (unsigned int i = 0; i < l2.bias().size(); ++i) {
            _pesi.biasLayer2[i] = l2.bias()[i];
        }

        if(l2.weight().size() != SizeLayer23 * SizeLayer1) { std::cout<<"ERRORE DIMENSIONI WEIGHT LAYER2"<<std::endl;}

        for (unsigned int in = 0; in < SizeLayer1; ++in) {
            for (unsigned int out = 0; out < SizeLayer23; ++out) {
                _pesi.pesiLayer2[in][out] = l2.weight()[out + in * SizeLayer23];
            }
        }
    }
    //-------------------------------------------------------
    {
        //std::cout<<"loadLayer3"<<std::endl;
        auto& l3 = _NNUEmodel->getLayer(2);

        if(l3.bias().size() != SizeLayer23) { std::cout<<"ERRORE DIMENSIONI BIAS LAYER3"<<std::endl;}
        for (unsigned int i = 0; i < l3.bias().size(); ++i) {
            _pesi.biasLayer3[i] = l3.bias()[i];
        }

        if(l3.weight().size() != SizeLayer23 * SizeLayer23) { std::cout<<"ERRORE DIMENSIONI WEIGHT LAYER3"<<std::endl;}

        for (unsigned int in = 0; in < SizeLayer23; ++in) {
            for (unsigned int out = 0; out < SizeLayer23; ++out) {
                _pesi.pesiLayer3[in][out] = l3.weight()[out + in * SizeLayer23];
            }
        }
    }
    //-------------------------------------------------------
    {
        //std::cout<<"loadLayer4"<<std::endl;
        auto& l4 = _NNUEmodel->getLayer(3);

        if(l4.bias().size() != 1) { std::cout<<"ERRORE DIMENSIONI BIAS LAYER4"<<std::endl;}
        _pesi.biasOutput = l4.bias()[0];

        if(l4.weight().size() != SizeLayer23) { std::cout<<"ERRORE DIMENSIONI WEIGHT LAYER4"<<std::endl;}

        for (unsigned int in = 0; in < SizeLayer23; ++in) {
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

    /*for(unsigned int i = 0; i < _input.NInput[0]; ++i) {
        std::cout<<_input.InputCompatto[0][i]<<std::endl;
    }
    for(unsigned int i = 0; i < _input.NInput[1]; ++i) {
        std::cout<<_input.InputCompatto[1][i]+ 40960<<std::endl;
    }*/
    if(_input.NInput[0] != _input.NInput[1]) {
        std::cout<<"ERRORE INPUT"<<std::endl;
    }
}

void PedoneCheck::calcolaRisRete() {
    // calcola layer 1
    for(unsigned int i = 0; i< SizeLayer1/2; ++i) {
        _risRete.ris32Layer1[i] = _pesi.biasLayer1[i];
        _risRete.ris32Layer1[i + 256] = _pesi.biasLayer1[i];
    }
    for (unsigned int i = 0; i < 2; ++i) {
        for (unsigned int k = 0; k < _input.NInput[i]; ++k) {
            for (unsigned int j = 0; j < SizeLayer1 / 2; ++j) {
                _risRete.ris32Layer1[(i * (SizeLayer1 / 2)) + j] += _pesi.pesiLayer1[_input.InputCompatto[i][k]][j];
            }
        }
    }
    for (unsigned int i = 0; i < SizeLayer1; ++i) {
        _risRete.risLayer1[i] = _ActivationTrain(_risRete.ris32Layer1[i]);
    }

    // calcola layer 2
    for (unsigned int out = 0; out < SizeLayer23; ++out) {
        double ris = _pesi.biasLayer2[out];
        for (unsigned int in = 0; in < SizeLayer1; ++in) {
            ris += _risRete.risLayer1[in] * _pesi.pesiLayer2[in][out];
        }
        _risRete.risLayer2[out] = _ActivationTrain(ris / 64.0);
    }

    // calcola layer 3
    for (unsigned int i = 0; i < SizeLayer23; ++i) {
        double ris = _pesi.biasLayer3[i];
        for (unsigned int j = 0; j < SizeLayer23; ++j) {
            ris += _risRete.risLayer2[j] * _pesi.pesiLayer3[j][i];
        }
        _risRete.risLayer3[i] = _ActivationTrain(ris / 64.0);
    }

    // calcola output
    _risRete.output = _pesi.biasOutput;
    for (unsigned int i=0; i < SizeLayer23; ++i) {
        _risRete.output += _risRete.risLayer3[i] * _pesi.pesiOutput[i];
    }

    //std::cout<<"check layer 1"<<std::endl;
    for(unsigned int i = 0; i < SizeLayer1; ++i) {
        if( std::abs(_risRete.risLayer1[i] - _NNUEmodel->getLayer(0).getOutput(i))> 1e-7) {
            std::cout<<"propagate layer 1 error"<<std::endl;
            std::cout << i << " " << _risRete.risLayer1[i] << " " <<_NNUEmodel->getLayer(0).getOutput(i)<<std::endl;
            exit(0);
        }
    }
    //std::cout<<"check layer 2"<<std::endl;
    for(unsigned int i = 0; i < SizeLayer23; ++i) {
        if( std::abs(_risRete.risLayer2[i] - _NNUEmodel->getLayer(1).getOutput(i))> 1e-7) {
            std::cout<<"propagate layer 2 error"<<std::endl;
            std::cout << i << " " << _risRete.risLayer2[i] << " " <<_NNUEmodel->getLayer(1).getOutput(i)<<std::endl;
            exit(0);
        }
    }
    //std::cout<<"check layer 3"<<std::endl;
    for(unsigned int i = 0; i < SizeLayer23; ++i) {
        if( std::abs(_risRete.risLayer3[i] - _NNUEmodel->getLayer(2).getOutput(i))> 1e-7) {
            std::cout<<"propagate layer 3 error"<<std::endl;
            std::cout << i << " " << _risRete.risLayer3[i] << " " <<_NNUEmodel->getLayer(2).getOutput(i)<<std::endl;
            exit(0);
        }
    }
    //std::cout<<"check output"<<std::endl;
    if( std::abs(_risRete.output -  _NNUEmodel->getLayer(3).getOutput(0))> 1e-7) {
        std::cout<<"propagate layer 4 error"<<std::endl;
        std::cout << _risRete.output << " " <<_NNUEmodel->getLayer(3).getOutput(0)<<std::endl;
        exit(0);
    }
}

void PedoneCheck::calcGrad(double label) {
    double err;
    // output
    _biasGrad.BiasGradOutput = _risRete.output - label;
    if (std::abs(_NNUEmodel->getLayer(3).getBiasSumGradient(0) - _biasGrad.BiasGradOutput) > 1e-7) {
        std::cout << "grad out error "<< _NNUEmodel->getLayer(3).getBiasSumGradient(0)<<" "<<_biasGrad.BiasGradOutput<<std::endl;
        exit(-1);
    }

    // check layer3
    for (unsigned int i = 0; i < SizeLayer23; ++i) {
        err = _biasGrad.BiasGradOutput * _pesi.pesiOutput[i];
        _biasGrad.BiasGradLayer3[i] = err * _Derivative(_risRete.risLayer3[i]) / 64.0;
    }
    for(unsigned int i = 0; i < SizeLayer23; ++i) {
        if (std::abs(_NNUEmodel->getLayer(2).getBiasSumGradient(i) - _biasGrad.BiasGradLayer3[i]) > 1e-7) {
            std::cout<<_risRete.risLayer3[i]<<std::endl;
            std::cout << "grad out error "<< _NNUEmodel->getLayer(2).getBiasSumGradient(i)<<" "<<_biasGrad.BiasGradLayer3[i]<<std::endl;
            std::cout << _NNUEmodel->getLayer(2).getBiasSumGradient(i) - _biasGrad.BiasGradLayer3[i]<<std::endl;
            exit(-1);
        }
    }

    // check layer2
    for (unsigned int i = 0; i < SizeLayer23; ++i) {
        err = 0.0;
        for (unsigned int j = 0; j < SizeLayer23; ++j) {
            err += _biasGrad.BiasGradLayer3[j] * _pesi.pesiLayer3[i][j];
        }
        _biasGrad.BiasGradLayer2[i] = err * _Derivative(_risRete.risLayer2[i]) / 64.0;
    }
    for(unsigned int i = 0; i < SizeLayer23; ++i) {
        if (std::abs(_NNUEmodel->getLayer(1).getBiasSumGradient(i) - _biasGrad.BiasGradLayer2[i]) > 1e-7) {
            std::cout<<_risRete.risLayer2[i]<<std::endl;
            std::cout << "grad out error "<< _NNUEmodel->getLayer(1).getBiasSumGradient(i)<<" "<<_biasGrad.BiasGradLayer2[i]<<std::endl;
            std::cout << _NNUEmodel->getLayer(1).getBiasSumGradient(i) - _biasGrad.BiasGradLayer2[i]<<std::endl;
            exit(-1);
        }
    }

    // check layer1
    for (unsigned int i = 0; i < SizeLayer1/2; ++i) {
        err = 0.0;
        for (unsigned int j = 0; j < SizeLayer23; ++j) {
            err += _biasGrad.BiasGradLayer2[j] * _pesi.pesiLayer2[i][j];
        }
        _biasGrad.BiasGradLayer1[i] = err * _Derivative(_risRete.risLayer1[i]);
    }
    for (unsigned int i = 0; i < SizeLayer1/2; ++i) {
        err = 0.0;
        for (unsigned int j = 0; j < SizeLayer23; ++j) {
            err += _biasGrad.BiasGradLayer2[j] * _pesi.pesiLayer2[i+256][j];
        }
        _biasGrad.BiasGradLayer1[i + 256] = err * _Derivative(_risRete.risLayer1[i + 256]);
    }
    for(unsigned int i = 0; i < SizeLayer1/2; ++i) {
        if (std::abs(_NNUEmodel->getLayer(0).getBiasSumGradient(i) - _biasGrad.BiasGradLayer1[i] - _biasGrad.BiasGradLayer1[i + 256]) > 1e-7) {
            std::cout<<_risRete.risLayer1[i]<<std::endl;
            std::cout << "grad out error "<< _NNUEmodel->getLayer(0).getBiasSumGradient(i)<<" "<<_biasGrad.BiasGradLayer1[i]+ _biasGrad.BiasGradLayer1[i+256]<<std::endl;
            std::cout << _NNUEmodel->getLayer(0).getBiasSumGradient(i) - _biasGrad.BiasGradLayer1[i] + _biasGrad.BiasGradLayer1[i+256]<<std::endl;
            exit(-1);
        }
    }
}

void PedoneCheck::updateweights(double l) {
    // output layer
    _pesi.biasOutput -= l * _biasGrad.BiasGradOutput;
    if(std::abs(_pesi.biasOutput - _NNUEmodel->getLayer(3).bias()[0]) > 1e-7) {
        std::cout<<"error updating bias of output layer"<<std::endl;
        std::cout<<_pesi.biasOutput<<std::endl;
        std::cout<<_NNUEmodel->getLayer(3).bias()[0]<<std::endl;
        exit(-1);
    }
    for( unsigned int i = 0; i <SizeLayer23; ++i) {
        _pesi.pesiOutput[i] -= l * _biasGrad.BiasGradOutput * _risRete.risLayer3[i];
        if(std::abs(_pesi.pesiOutput[i] - _NNUEmodel->getLayer(3).weight()[i]) > 1e-7) {
            std::cout<<"error updating pesi of output layer"<<std::endl;
            std::cout<<_pesi.pesiOutput[i] <<std::endl;
            std::cout<<_NNUEmodel->getLayer(3).weight()[i]<<std::endl;
            exit(-1);
        }
    }

    // layer3
    for (unsigned int i = 0; i< SizeLayer23; ++i) {
        for (unsigned int j = 0; j< SizeLayer23; ++j) {
            _pesi.pesiLayer3[j][i] -= l * _biasGrad.BiasGradLayer3[i] * _risRete.risLayer2[j];
            if(std::abs(_pesi.pesiLayer3[j][i] - _NNUEmodel->getLayer(2).weight()[i + j * SizeLayer23])>1e-7) {
                std::cout<<"error updating pesi of layer3"<<std::endl;
                std::cout<<_pesi.pesiLayer3[j][i] <<std::endl;
                std::cout<< _NNUEmodel->getLayer(2).weight()[i + j * SizeLayer23]<<std::endl;
                exit(-1);
            }
        } 
        _pesi.biasLayer3[i] -= l * _biasGrad.BiasGradLayer3[i];
        if(std::abs(_pesi.biasLayer3[i] - _NNUEmodel->getLayer(2).bias()[i])>1e-7) {
            std::cout<<"error updating bias of layer3"<<std::endl;
            std::cout<<_pesi.biasLayer3[i] <<std::endl;
            std::cout<< _NNUEmodel->getLayer(2).bias()[i]<<std::endl;
            exit(-1);
        }
    }

    // layer2
    for (unsigned int i = 0; i< SizeLayer23; ++i) {
        for (unsigned int j = 0; j< SizeLayer1; ++j) {
            _pesi.pesiLayer2[j][i] -= l * _biasGrad.BiasGradLayer2[i] * _risRete.risLayer1[j];
            if(std::abs(_pesi.pesiLayer2[j][i] - _NNUEmodel->getLayer(1).weight()[i + j * SizeLayer23])>1e-7) {
                std::cout<<"error updating pesi of layer2"<<std::endl;
                std::cout<<_pesi.pesiLayer2[j][i] <<std::endl;
                std::cout<< _NNUEmodel->getLayer(1).weight()[i + j * SizeLayer23]<<std::endl;
                exit(-1);
            }
        } 
        _pesi.biasLayer2[i] -= l * _biasGrad.BiasGradLayer2[i];
        if(std::abs(_pesi.biasLayer2[i] - _NNUEmodel->getLayer(1).bias()[i])>1e-7) {
            std::cout<<"error updating bias of layer2"<<std::endl;
            std::cout<<_pesi.biasLayer2[i] <<std::endl;
            std::cout<< _NNUEmodel->getLayer(1).bias()[i]<<std::endl;
            exit(-1);
        }
    }

    //layer1
    for (unsigned int i = 0; i < 2; ++i) {
        for (unsigned int k = 0; k < _input.NInput[i]; ++k) {
            uint16_t idx = _input.InputCompatto[i][k];
            for (unsigned int j = 0; j < SizeLayer1 / 2; ++j) {
                /*if( idx == 19020 && j ==2) { std::cout<<"ECCCCCCCOMI"<<std::endl;
                    std::cout<<"I "<<i<<std::endl;
                    std::cout<<"biasGradient "<<  _biasGrad.BiasGradLayer1[j]<<std::endl;
                    std::cout<<"biasGradient "<<  _biasGrad.BiasGradLayer1[j + 256]<<std::endl;
                    std::cout<<_pesi.pesiLayer1[idx][j]<<std::endl;
                }*/
                _pesi.pesiLayer1[idx][j] -= l * _biasGrad.BiasGradLayer1[j + i * 256];
            }
        }
    }
    for (unsigned int i = 0; i < 2; ++i) {
        for (unsigned int k = 0; k < _input.NInput[i]; ++k) {
            uint16_t idx = _input.InputCompatto[i][k];
            for (unsigned int j = 0; j < SizeLayer1 / 2; ++j) {
                if(std::abs(_pesi.pesiLayer1[idx][j] - _NNUEmodel->getLayer(0).weight()[idx * SizeLayer1 / 2 + j])>1e-7) {
                    std::cout<<idx<< " "<<j<<std::endl;
                    std::cout<<"error updating pesi of layer1"<<std::endl;
                    std::cout<<"PEDONE "<<_pesi.pesiLayer1[idx][j] <<std::endl;
                    std::cout<<"VAJO "<<_NNUEmodel->getLayer(0).weight()[idx * SizeLayer1 / 2 + j]<<std::endl;
                    std::cout<<std::abs(_pesi.pesiLayer1[idx][j] - _NNUEmodel->getLayer(0).weight()[idx * SizeLayer1 / 2 + j])*1e5<<std::endl;
                    exit(-1);
                }
            }
        }
    }
    for (unsigned int i = 0; i<SizeLayer1/2; ++i) {
      _pesi.biasLayer1[i] -= l * _biasGrad.BiasGradLayer1[i];
      _pesi.biasLayer1[i] -= l * _biasGrad.BiasGradLayer1[i + 256];
      if (std::abs(_pesi.biasLayer1[i] - _NNUEmodel->getLayer(0).bias()[i])>1e-8) {
            std::cout<<"error updating bias of layer1"<<std::endl;
            std::cout<<_pesi.biasLayer1[i] <<std::endl;
            std::cout<< _NNUEmodel->getLayer(0).bias()[i]<<std::endl;
            exit(-1);
      }
    }

    
}


double PedoneCheck::_ActivationTrain(double x) {
    x = std::max(x, 0.0);
    x = std::min(x, 127.0);
    return x;
}

double PedoneCheck::_Derivative(double x) {
    return (x > 0.0 && x < 127.0) ? 1.0 : 0.0;
}
