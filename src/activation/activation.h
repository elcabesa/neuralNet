#ifndef _ACTIVATION_H
#define _ACTIVATION_H

#include <memory>
#include <string>

class Activation {    
public:
    enum class type
    {
        linear,
        relu
    };
   
    Activation();
    virtual ~Activation();
    
    virtual double propagate(double input) const = 0;
    virtual double derivate(double input) const = 0;
    virtual type getType() const = 0;
};

class ActivationFactory {
public:
    
    static std::unique_ptr<Activation> create(const Activation::type t  = Activation::type::linear);
};

#endif 
