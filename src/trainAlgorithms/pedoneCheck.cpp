#include <cmath>
#include <iostream>
#include <random>

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
    std::random_device rnd;

    //-------------------------------------------------------
    {
        std::normal_distribution<double> dist(0.0, 0.1 * sqrt(1.0 / 32.0));
        //std::cout<<"bias"<<std::endl;
        for (unsigned int i = 0; i < SizeInputLayer; ++i) {
            _pesi.biasLayer1[i] = 0.5;
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
        std::normal_distribution<double> dist(0.0, sqrt(1.0 / SizeLayer1));
        for (unsigned int in = 0; in < SizeLayer1; ++in) {
            for (unsigned int out = 0; out < SizeLayer2; ++out) {
                _pesi.pesiLayer2[in][out] = dist(rnd);
            }
        }
        for (unsigned int out = 0; out < SizeLayer2; ++out) {
            double sum = 0.0;
            for (unsigned int in = 0; in < SizeLayer1; ++in) {
                sum +=  _pesi.pesiLayer2[in][out];
            }
            _pesi.biasLayer2[out] = 0.5 - 0.5 * sum;
        }
    }
    //-------------------------------------------------------
    {
        std::normal_distribution<double> dist(0.0, sqrt(1.0 / SizeLayer2));
        for (unsigned int in = 0; in < SizeLayer2; ++in) {
            for (unsigned int out = 0; out < SizeLayer3; ++out) {
                _pesi.pesiLayer3[in][out] = dist(rnd);
            }
        }

        for (unsigned int out = 0; out < SizeLayer3; ++out) {
            double sum = 0.0;
            for (unsigned int in = 0; in < SizeLayer2; ++in) {
                sum +=  _pesi.pesiLayer3[in][out];
            }
            _pesi.biasLayer3[out] = 0.5 - 0.5 * sum;
        }
    }
    //-------------------------------------------------------
    {
        _pesi.biasOutput = 0.0;

        for (unsigned int in = 0; in < SizeLayer3; ++in) {
            _pesi.pesiOutput[in] = 0.0;
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

    // check layer3
    for (unsigned int in = 0; in < SizeLayer3; ++in) {
        err = _biasGrad.BiasGradOutput * _pesi.pesiOutput[in];
        _biasGrad.BiasGradLayer3[in] = err * _Derivative(_risRete.risLayer3[in])/* / 64.0*/;
        /*if(in==0)_varCalL3_1bias.addValue(_biasGrad.BiasGradLayer3[in]);
        if(in==8)_varCalL3_2bias.addValue(_biasGrad.BiasGradLayer3[in]);
        if(in==16)_varCalL3_3bias.addValue(_biasGrad.BiasGradLayer3[in]);
        if(in==24)_varCalL3_4bias.addValue(_biasGrad.BiasGradLayer3[in]);*/
    }

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

    return std::pow(_risRete.output - label, 2.0) / 2.0;
}

void PedoneCheck::updateweights(double l) {
    // output layer
    _pesi.biasOutput -= l * _biasGrad.BiasGradOutput;

    for( unsigned int out = 0; out < SizeLayer3; ++out) {
        _pesi.pesiOutput[out] -= l * _biasGrad.BiasGradOutput * _risRete.risLayer3[out];
        /*_varCalOutWeight[out].addValue(_biasGrad.BiasGradOutput * _risRete.risLayer3[out]);*/
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
    x = std::max(x, 0.01 * x);
    x = std::min(x, 1.0 + 0.01 * (x - 1.0));
    return x;
}

double PedoneCheck::_Derivative(double x) {
    return (x > 0.0 && x < 1.0) ? 1.0 : 0.01;
}
