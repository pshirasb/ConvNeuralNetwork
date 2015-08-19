#ifndef __H_CNN__
#define __H_CNN__

#ifdef _EXEC
    #include <armadillo>
#else
    #include <RcppArmadillo.h>
#endif

#define NDEBUG

#ifdef DEBUG
    #define dbg_print(x) cout << "[dbg:" << __LINE__ << "] " << x << endl
    #define dbg_assert(cond) (cond || dbg_print("assert failed: " << #cond))
#else
    #define dbg_print(x)     ((void)0)
    #define dbg_assert(cond) ((void)0)
#endif

using namespace arma;

//TODO: you would want to move layers
// to seperate files eventually.
class NeuralNet;
class Layer;
class LinearLayer;
class ConvLayer;
class LogitLayer;
class InputLayer;
class CostFunction;
class MSE;
class LayerIter;
class LinearLayerIter;
class ConvLayerIter;

// NN class
class NeuralNet {
    
public:

        // Constructor
        // TODO: more feature-rich constructor needed
        NeuralNet() 
            : _layers_head() 
            , _layers_tail() 
            , _cf    () {}

        // Add layer l to the network
        bool add(Layer* l);
        bool add(CostFunction *cf) { _cf = cf; return(0); }

        double computeCost(cube x, cube y);

        // train
        double train(cube x, cube y, double alpha, long int epoch);

        // Forward pass on input and
        // return the resulting vector
        cube pred(cube input);
        cube backprop(cube x,cube y);

        void updateWeights(double alpha);
        double checkGradients(cube x,cube y);

        
        void printAll();
        /*
        { 
            if(_layers_head) {
                _layers_head->print(); 
            }
        }*/

private:
        Layer*          _layers_head;
        Layer*          _layers_tail;
        CostFunction*   _cf;

};

// Abstract class for a layer
class Layer {
public:

    friend class LayerIter;

    // Default Constructor
    Layer() 
        : _next()   , next	(_next)
        , _prev()   , prev	(_prev)
        , _units()	, units	(_units) {}

/*  unnecessary
    // Constructor to be used by children
    Layer(Layer* next_, Layer* prev_, mat w_, mat dw_, mat a_, int units_)
        : _next  (next_)	, next	(_next)
        , _prev  (prev_)	, prev	(_prev)
        , _units (units_)	, units (_units) {}
*/   
  
    virtual ~Layer() {
        // TODO: free all member variables
    }

    virtual bool connect (Layer *l) 
    {
        dbg_assert(l);        // Bad idea, need to check this in runtime
        if(_next == 0) {
            return(l->connectBack(this) ||
                  (_next = l));
        }
        return(_next->connect(l));
    }

    virtual bool connectBack(Layer *l) 
    {
        // Default
        _prev = l;       
        
        // TODO Move this extra bit this to 
        // corresponding subclasses
        _units = l->units;

        return(0);
    }    



    // Backprop
    // Phase 1: Consume
    // compute d(z) / d(theta)
    // gradient = delta * d(z) / d(theta)
    // (Optional) adjust theta += -alpha * gradient
    // Phase 2: Produce
    // compute grad = d(z) / d(a_prev_layer)
    // compute delta_new = sum(delta * grad) over output units
    // NOTE: delta_new should be a vector of size nInput
    // return(delta_new)
    virtual cube backward(cube x) = 0;
    // Forward Pass
    virtual cube forward(cube) = 0;

    virtual void applyDw(double alpha) { }    

    void print(){
        cout << "Layer with " << _units << " units. " << endl;
        if(_next)
            _next->print();
    }

    virtual LayerIter* createIterator(std::string value) {
        return(0);
    }

    virtual cube& getWeightMat()     = 0;
    virtual cube& getDWeightMat()    = 0;
    virtual cube& getActivationCube() = 0;

    // Read-Only referances to member variables
    Layer*  const & next;     /* Next layer             */
    Layer*  const & prev;     /* Previous layer         */
    const   int&    units;    /* number of output units */
    const   double  momentum = 0.0;

protected:

    Layer* _next;     /* Next layer             */
    Layer* _prev;     /* Previous layer         */
    int    _units;    /* number of output units */

};  

class LinearLayer: public Layer {
public:
    
    LinearLayer(int numUnits) 
    {
        _units = numUnits;   
        dbg_print("LinearLayer with " << _units << " units created.");
    }

    LinearLayer(cube w) 
    {
        _w   = w;
        dbg_print("LinearLayer created given w. UNSUPPORTED.");
    }

    bool connectBack (Layer *l) override;      
   

    // Forward pass on input x, sets last_activation
    // and returns a vec of size nOutput
    cube forward(cube x);          

    // makes a backward pass and computes
    // gradients on the last activations
    cube backward(cube x);

    void applyDw(double alpha);
    
    LayerIter* createIterator(std::string value) override ;
    cube& getWeightMat()     override { return _w; }
    cube& getDWeightMat()    override { return _dw; }
    cube& getActivationCube() override { return _a; }

    friend class LinearLayerIter;
    
protected:

    cube    _w;      /* Weight cube             */
    cube    _dw;     /* Weight Gradient cube    */    
    cube    _b;      /* bias cube               */
    cube    _db;     /* bias Gradient cube      */    
    cube    _a;      /* Activation patterns     */
};

// Sigmoid Activation Layer
class LogitLayer : public Layer {
public:
    //TODO: fix inconsistant return types
     
    cube forward(cube x);
    cube backward(cube x);

    bool  connectBack (Layer *l) override;      

    cube& getWeightMat()     override { return _w; }
    cube& getDWeightMat()    override { return _dw; }
    cube& getActivationCube() override { return _a; }

    
private:
    cube    _w;     /* Weight matrix          */
    cube    _dw;    /* Gradient matrix        */    
    cube    _a;     /* Activation patterns    */

};

class InputLayer: public Layer {
public:
    
    InputLayer(int x, int y, int z) {
        _units = z;
        _a     = zeros<cube>(x,y,z);
    }

    cube forward(cube x) {
        _a = x;
        if(_next)
            return(_next->forward(_a));
        return(_a);
    }

    // Not sure about this one
    cube backward(cube x){
        return(x);
    }
    
    cube& getWeightMat()     override { return _w; }
    cube& getDWeightMat()    override { return _dw; }
    cube& getActivationCube() override { return _a; }

private:
    cube    _w;     /* Weight matrix          */
    cube    _dw;    /* Gradient matrix        */    
    cube    _a;     /* Activation patterns    */
};


class CostFunction {
public:
    virtual double cost    (cube pred, cube y) = 0;
    virtual cube   gradient(cube pred, cube y) = 0;
};

class MSE : public CostFunction {
public:
    double cost     (cube pred, cube y) ;
    cube   gradient (cube pred, cube y) ;
};

// Convolution layer
class ConvLayer: public Layer {
public:
    
    friend class ConvLayerIter;

    ConvLayer(int kernel_size, int depth);

    cube backward   (cube x);
    cube forward    (cube);

    void applyDw        (double alpha);
    bool connectBack    (Layer *l);

    LayerIter* createIterator(std::string value) override ;
    
    cube& getWeightMat()     override { return _w(0); }
    cube& getDWeightMat()    override { return _dw(0); }
    cube& getActivationCube() override { return _a; }
    

private:
    
    int     _ksize;  /* size of the kernel  */
    
    field<cube>     _w;     /* Weight/Kernel Cube   */
    field<cube>     _dw;    /* Weight Update        */    
    cube            _b;     /* Bias                 */
    cube            _db;    /* Bias Update          */
    cube            _a;     /* Activation Cube      */

};

// Sumbsampling/max-pooling layer
class maxPoolLayer : public ConvLayer {

};

// Iterator Classes
class LayerIter {
public:
    virtual int     next()      = 0;
    virtual double  get()       = 0;
    virtual void    set(double) = 0;
    virtual void    reset()     = 0;
};

class LinearLayerIter : public LayerIter {
public:
    LinearLayerIter(LinearLayer* l, std::string var) 
        : _l(l) 
        , _v()
        , _b()
        , _i(0)
        , _j(0)
        , _k(0)
        , _m(0)
        , _bflag(1)
    {
        if(var == "w") {
            _v = &(_l->_w);
            _b = &(_l->_b);
        } else
        if(var == "dw") {
            _v = &(_l->_dw);
            _b = &(_l->_db);
        } else {
            dbg_print("invalid 'var' param to LinearLayerIter");
        }
    }
    int     next()      ;
    double  get()       ;
    void    set(double) ;
    void    reset()     ;
protected:
    LinearLayer* _l;
    cube*        _v;
    cube*        _b;
    uword        _i,
                 _j,
                 _k,
                 _m;
    bool         _bflag;                
};

class ConvLayerIter : public LayerIter {
public:
    ConvLayerIter(ConvLayer* l, std::string var) 
        : _l(l) 
        , _v()
        , _b()
        , _bflag(1)
        , _m( 0)
        , _i( 0)
        , _j( 0)
        , _k( 0)
    {
        if(var == "w") {
            _v = &(_l->_w);
            _b = &(_l->_b);
        } else
        if(var == "dw") {
            _v = &(_l->_dw);
            _b = &(_l->_db);
        } else {
            dbg_print("invalid 'var' param to ConvLayerIter");
        }
    }
    int     next()      ;
    double  get()       ;
    void    set(double) ;
    void    reset()     ;
protected:
    ConvLayer*    _l;
    field<cube>*  _v;
    cube*         _b;
    bool          _bflag;
    uword         _i,
                  _j,
                  _k,
                  _m;

};


mat class2mat(const mat& x) ;
mat conv2d(mat, mat);
mat conv2d(cube,cube);
mat  addPadding(mat  x, int ksize);
cube addPadding(cube x, int ksize);
#endif
