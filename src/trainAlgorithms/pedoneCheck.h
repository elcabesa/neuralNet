#ifndef _PEDONE_CHECK_H
#define _PEDONE_CHECK_H

#include <cstdint>

class Model;
class LabeledExample;

class PedoneCheck {
public:
    PedoneCheck(Model* model);
    void caricaPesi();
    void caricaInput(LabeledExample& b);
    void calcolaRisRete();
    double calcGrad(double label);
    void updateweights(double l);


    static constexpr unsigned int MaxInput = 10*64*64;
    static constexpr unsigned int SizeLayer1 = 512;
    static constexpr unsigned int SizeLayer23 = 32;
    static constexpr unsigned int SizeInputCompatto = 32;

    struct PesiNNUE {
        double pesiLayer1[MaxInput][SizeLayer1/2];
        double biasLayer1[SizeLayer1/2];

        double pesiLayer2[SizeLayer1][SizeLayer23];
        double biasLayer2[SizeLayer23];

        double pesiLayer3[SizeLayer23][SizeLayer23];
        double biasLayer3[SizeLayer23];

        double pesiOutput[SizeLayer23];
        double biasOutput;
    };

    struct NNUEInput {
        uint16_t NInput[2];
        uint16_t InputCompatto[2][SizeInputCompatto];

    };

    struct RisRete {
        double ris32Layer1[SizeLayer1];
        double risLayer1[SizeLayer1];
        double risLayer2[SizeLayer23];
        double risLayer3[SizeLayer23];
        double output;
    };

    struct BiasGrad {
        double BiasGradLayer1[SizeLayer1];
        double BiasGradLayer2[SizeLayer23];
        double BiasGradLayer3[SizeLayer23];
        double BiasGradOutput;
    };

private:
    PesiNNUE _pesi;
    Model* _NNUEmodel;
    NNUEInput _input;
    RisRete _risRete;
    BiasGrad _biasGrad;

    static double _ActivationTrain(double x);
    static double _Derivative(double x);
};

#endif
