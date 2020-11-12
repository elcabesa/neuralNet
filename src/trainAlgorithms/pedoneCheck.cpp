#include <cmath>
#include <iostream>
#include <random>

#include "labeledExample.h"
#include "pedoneCheck.h"
#include "model.h"

PedoneCheck::PedoneCheck(Model* model):_NNUEmodel(model) {
    std::cout<<"PEDONE CHECK"<<std::endl;
}

void PedoneCheck::caricaPesi() {
    std::random_device rnd;

    //-------------------------------------------------------
    {
        std::normal_distribution<double> dist(0.0, sqrt(2.0 / 30));
        //std::cout<<"bias"<<std::endl;
        for (unsigned int i = 0; i < SizeInputLayer; ++i) {
            _pesi.biasLayer1[i] = 0;
        }

        //std::cout<<"weight"<<std::endl;
        for (unsigned int in = 0; in < MaxInput; ++in) {
            for (unsigned int out = 0; out < SizeInputLayer; ++out) {
                _pesi.pesiLayer1[in][out] = dist(rnd);
            }
        }
    }
    //-------------------------------------------------------
    {
        std::normal_distribution<double> dist(0.0, sqrt(2.0 / SizeLayer1));
        for (unsigned int i = 0; i < SizeLayer2; ++i) {
            _pesi.biasLayer2[i] = 0;
        }

        for (unsigned int in = 0; in < SizeLayer1; ++in) {
            for (unsigned int out = 0; out < SizeLayer2; ++out) {
                _pesi.pesiLayer2[in][out] = dist(rnd);
            }
        }
    }
    //-------------------------------------------------------
    {
        std::normal_distribution<double> dist(0.0, sqrt(2.0 / SizeLayer2));
        for (unsigned int i = 0; i < SizeLayer3; ++i) {
            _pesi.biasLayer3[i] = 0;
        }

        for (unsigned int in = 0; in < SizeLayer2; ++in) {
            for (unsigned int out = 0; out < SizeLayer3; ++out) {
                _pesi.pesiLayer3[in][out] = dist(rnd);
            }
        }
    }
    //-------------------------------------------------------
    {
        std::normal_distribution<double> dist(0.0, sqrt(2.0 / SizeLayer3));
        _pesi.biasOutput =0;

        for (unsigned int in = 0; in < SizeLayer3; ++in) {
            _pesi.pesiOutput[in] = dist(rnd);
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
    for (unsigned int out = 0; out < SizeLayer1; ++out) {
        _risRete.risLayer1[out] = _ActivationTrain(_risRete.ris32Layer1[out]);
    }

    // calcola layer 2
    for (unsigned int out = 0; out < SizeLayer2; ++out) {
        double ris = _pesi.biasLayer2[out];
        for (unsigned int in = 0; in < SizeLayer1; ++in) {
            ris += _risRete.risLayer1[in] * _pesi.pesiLayer2[in][out];
        }
        _risRete.risLayer2[out] = _ActivationTrain(ris/* / 64.0*/);
    }

    // calcola layer 3
    for (unsigned int out = 0; out < SizeLayer3; ++out) {
        double ris = _pesi.biasLayer3[out];
        for (unsigned int in = 0; in < SizeLayer2; ++in) {
            ris += _risRete.risLayer2[in] * _pesi.pesiLayer3[in][out];
        }
        _risRete.risLayer3[out] = _ActivationTrain(ris/* / 64.0*/);
    }

    // calcola output
    _risRete.output = _pesi.biasOutput;
    for (unsigned int in = 0; in < SizeLayer3; ++in) {
        _risRete.output += _risRete.risLayer3[in] * _pesi.pesiOutput[in];
    }

}

double PedoneCheck::calcGrad(double label) {
    double err;
    // output
    _biasGrad.BiasGradOutput = _risRete.output - label;

    // check layer3
    for (unsigned int in = 0; in < SizeLayer3; ++in) {
        err = _biasGrad.BiasGradOutput * _pesi.pesiOutput[in];
        _biasGrad.BiasGradLayer3[in] = err * _Derivative(_risRete.risLayer3[in])/* / 64.0*/;
    }

    // check layer2
    for (unsigned int in = 0; in < SizeLayer2; ++in) {
        err = 0.0;
        for (unsigned int out = 0; out < SizeLayer3; ++out) {
            err += _biasGrad.BiasGradLayer3[out] * _pesi.pesiLayer3[in][out];
        }
        _biasGrad.BiasGradLayer2[in] = err * _Derivative(_risRete.risLayer2[in])/* / 64.0*/;
    }

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
    return std::pow(_risRete.output - label, 2.0) / 2.0;
}

void PedoneCheck::updateweights(double l) {
    // output layer
    _pesi.biasOutput -= l * _biasGrad.BiasGradOutput;

    for( unsigned int out = 0; out < SizeLayer3; ++out) {
        _pesi.pesiOutput[out] -= l * _biasGrad.BiasGradOutput * _risRete.risLayer3[out];
    }

    // layer3
    for (unsigned int out = 0; out < SizeLayer3; ++out) {
        for (unsigned int in = 0; in < SizeLayer2; ++in) {
            _pesi.pesiLayer3[in][out] -= l * _biasGrad.BiasGradLayer3[out] * _risRete.risLayer2[in];
        } 
        _pesi.biasLayer3[out] -= l * _biasGrad.BiasGradLayer3[out];

    }

    // layer2
    for (unsigned int out = 0; out < SizeLayer2; ++out) {
        for (unsigned int in = 0; in < SizeLayer1; ++in) {
            _pesi.pesiLayer2[in][out] -= l * _biasGrad.BiasGradLayer2[out] * _risRete.risLayer1[in];
        } 
        _pesi.biasLayer2[out] -= l * _biasGrad.BiasGradLayer2[out];   
    }

    //layer1
    for (unsigned int pl = 0; pl < 2; ++pl) {
        for (unsigned int in = 0; in < _input.NInput[pl]; ++in) {
            uint16_t idx = _input.InputCompatto[pl][in];
            for (unsigned int out = 0; out < SizeInputLayer; ++out) {
                _pesi.pesiLayer1[idx][out] -= l * _biasGrad.BiasGradLayer1[out + pl * SizeInputLayer];
            }
        }
    }
    
    for (unsigned int out = 0; out < SizeInputLayer; ++out) {
      _pesi.biasLayer1[out] -= l * _biasGrad.BiasGradLayer1[out];
      _pesi.biasLayer1[out] -= l * _biasGrad.BiasGradLayer1[out + SizeInputLayer];
    }
}


double PedoneCheck::_ActivationTrain(double x) {
    x = std::max(x, 0.0);
    x = std::min(x, 1.0);
    return x;
}

double PedoneCheck::_Derivative(double x) {
    return (x > 0.0 && x < 1.0) ? 1.0 : 0.01;
}
