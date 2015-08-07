#include "convnn.h"
#include <omp.h>
#include <iostream>
#include <stdlib.h>

using namespace std;

/**********************************************************************
*   NeuralNet Class
***********************************************************************/
bool NeuralNet::add(Layer* l){
    //might throw bad_alloc
    if(!l){
        cout << __LINE__ << ": null param to add()." << endl;
        return(1);
    }
    
    _layers_tail = l;
    
    if(_layers_head == 0) {
        _layers_head = l;
        return(0);
    }

    return(_layers_head->connect(l));
}

void NeuralNet::printAll(){
    if(_layers_head) {
        _layers_head->print(); 
    }
}

cube NeuralNet::pred(cube input){
    
    Layer* l = _layers_head;
    if(!l){
        cout << "Empty network...cant recover..dying in 3..2..1" << endl;
    }
   
    return(_layers_head->forward(input)) ;
}

cube NeuralNet::backprop(cube x, cube y){
   
    dbg_assert(_layers_tail != 0);
    dbg_assert(_cf != 0);

    

    if(_layers_tail){
        _layers_tail->backward(_cf->gradient(pred(x),y));
    }
    return(x);

}

double NeuralNet::checkGradients(cube x,cube y){

    double epsilon = 0.0000001;
    double total  = 0;
    long int counter = 0;
    double pred_plus;
    double pred_minus;
    double gradApprox;
    //
    // Make forward pass and calculate gradients
    // NOTE: should grab gradients here to compare later
    // becuase they will change.
    backprop(x,y);
    dbg_print("Starting to calculate approx gradients");
    int i = 1;
    // Now compute approximate gradients
    for(Layer* l = _layers_head; l; l = l->next){
        LayerIter *itr_w  = l->createIterator("w");
        LayerIter *itr_dw = l->createIterator("dw");
        if(!itr_w || !itr_dw) 
            continue;
        do {
            double old_theta    = itr_w->get();
            double old_gradient = itr_dw->get();
            itr_w->set(old_theta + epsilon);
            pred_plus  = computeCost(x,y);
            itr_w->set(old_theta - epsilon);
            pred_minus = computeCost(x,y);
            gradApprox = (pred_plus - pred_minus) / (2.0f * epsilon);
            itr_w->set(old_theta);
            itr_dw->set(old_gradient);
            total      =+ abs(gradApprox - old_gradient);
            cout << abs(gradApprox - old_gradient) << endl;
            //cout << "(" << gradApprox << "," << old_gradient << ")" << endl;
            //cout << "(" << old_theta << "," << old_gradient << ")" << endl;
            counter++;

        } while (itr_w->next() && itr_dw->next());
        
        delete itr_w;
        delete itr_dw;
    }
     
    total = total / (double)counter; // Mean
    return(total);
}
double NeuralNet::computeCost(cube x, cube y){
    return(_cf->cost(pred(x),y));
}

double NeuralNet::train(cube x,cube y, double alpha, long int epoch){
    
    double cst = 0 ;
    int rnd = 0;
    cube sample_x;
    cube sample_y;

    for(int i=0; i < epoch; i++){
        rnd = rand() % (x.n_slices - 1);
        sample_x =(x.slices(rnd, rnd));
        sample_y =(y.slices(rnd, rnd));
        cst = computeCost(sample_x, sample_y);
        backprop(sample_x, sample_y);
        updateWeights(alpha);
        cout << "Epoch[" << i << "]: " << cst << endl;
    }


    
    double total = 0;
    double ctotal = 0;
    for(int i=0; i < x.n_slices; i++){
        sample_x = (x.slices(i,i));
        sample_y = (y.slices(i,i));
        cst = computeCost(sample_x, sample_y);
        total += cst;
        
        // classification error:
        uword rid, cid, sid;
        cube p = pred(sample_x);
        p.max(rid,cid,sid);
        if(sample_y(rid,cid,sid) == 1)
            ctotal++;   
    }
    total  /= x.n_slices;
    ctotal /= x.n_slices;
    cout << "Mean Training Error: " << total << endl;
    cout << "Classification Error: " << ctotal << endl;
    return 0;
}
void NeuralNet::updateWeights(double alpha){
    
    Layer *l = _layers_head;
    while(l){
        l->applyDw(alpha);
        l = l->next;
    }

}

/**********************************************************************
*   LinearLayer Class
***********************************************************************/
void LinearLayer::applyDw(double alpha)  { 
    //TODO: update bias here as well
    _w.slice(0) -= alpha * _dw.slice(0);
}    
bool LinearLayer::connectBack(Layer *l)
{   
    arma_rng::set_seed_random();
    
    //TODO: this is jenky
    uword prev_units = (l->getActivationCube()).n_elem;

    _w    = cube(_units, prev_units, 1, fill::randn);   
    _dw   = cube(_units, prev_units, 1, fill::zeros);
    _b    = cube(_units, 1         , 1, fill::randn);   
    _db   = cube(_units, 1         , 1, fill::zeros);
    _a    = cube(_units, 1         , 1, fill::zeros);
    _prev = l;
    dbg_print("connect back from LinearLayer with " << _units << " units." );
    return(0);

}

cube LinearLayer::backward(cube delta){
    
    // we need to calculate two types
    // of gradients here:
    // 1. d(z) / d(theta)
    // 2. d(z) / d(a_prev_layer)
    
    // Phase 1: Consume
    // compute d(z) / d(theta)
    if(!prev){
        cout << __LINE__ << ": _prev is null, this should not happen." << endl;
        return zeros<cube>(0,0,0);
    }
   
    dbg_assert(delta.n_cols == 1);
    dbg_assert(delta.n_slices == 1);

    dbg_print("backward at linear layer(" << _units << ")");
    mat delta_mat = delta.slice(0);
    cube prev_a = prev->getActivationCube();
    //mat dz_dw = reshape(prev_a, prev_a.n_elem, 1, 1);
    mat dz_dw = vectorise(prev_a);

    //TODO: this still looks jenky
    // Add bias term to dz_dw
    //mat bias = ones<mat>(dz_dw.n_cols, 1);
    //dz_dw.insert_rows(0, bias);  // Add bias term
    // delta_weight = delta * d(z)/d(w)
    // dz_dw is [nsamples, prev_nunits]
    // delta is [nsamples, nunits] , delta is [nunits,1]

    _dw.slice(0) = (momentum * _dw.slice(0)) + delta_mat * trans(dz_dw);
    _db.slice(0) = (momentum * _db.slice(0)) + delta_mat;

    // Phase 2: Produce
    // compute grad = d(z) / d(a_prev_layer)
    // NOTE: gradi here equals to just theta, however
    // we should remove the bias column
    // mat grad = _w.tail_cols(_w.n_cols - 1);
    

    // compute delta_new = sum(delta * grad) over output units
    // _w         [_units x prev->_units]
    // delta      [_units x 1 ]
    // new delta  [prev->_units x 1]

    mat new_delta = trans(_w.slice(0)) * delta_mat;
    
    // TODO: reduce the amount of mat copies.
    // as it stands, 5+ copies are performed...not good.


    cube next_delta = zeros<cube>(new_delta.n_rows, 1, 1);
    next_delta.slice(0) = new_delta;

    if(_prev)
        return(_prev->backward(next_delta));

    return(next_delta);

}    
cube LinearLayer::forward(cube x){
 
    //mat input = reshape(x, x.n_elem, 1, 1);
    mat input = vectorise(x);
    
    // Store Activation Values
    dbg_assert(input.n_rows == _w.slice(0).n_cols);
    
    dbg_print("forward at linear layer(" << _units << ")");

    // assuming input is [nx1]
    //_a.slice(0) = (input) * trans(_w.slice(0));
    _a.slice(0) = (_w.slice(0) * input) + _b.slice(0);

    if(_next) {
        //TODO: Maybe passing forward _a is inefficient
        //      so each later should read it from prev layer
        return(_next->forward(_a));
    }

    return(_a);
}

LayerIter* LinearLayer::createIterator(std::string value) {
    return(new LinearLayerIter(this, value));
}

/**********************************************************************
*   LogitLayer Class                                                  
***********************************************************************/
bool LogitLayer::connectBack(Layer *l)
{   
    arma_rng::set_seed_random();

    int nr = l->getActivationCube().n_rows;
    int nc = l->getActivationCube().n_cols;
    int ns = l->getActivationCube().n_slices;

    _a     = cube(nr, nc, ns, fill::zeros);
    _units = ns;
    _prev  = l;

    dbg_print("connect back from LogitLayer with " << _units << " units." );
    return(0);

}

cube LogitLayer::forward(cube x){
    dbg_print("forward at logit layer(" << _units << ")");
    _a = 1 / (1 + exp(-1 * x));
    if(_next) {
        //TODO: Maybe passing forward _a is inefficient
        //      so each later should read it from prev layer
        return(_next->forward(_a));
    }

    return(_a);
}

cube LogitLayer::backward(cube delta){
    // No need for phase1.
    
    // Phase 2: Produce
    // compute grad = d(a) / d(z)
    // delta = (nOut)x1, act = (nOutx1)
    // grad = (nOutx1)
    // element-wise multiplication
    
    //reshape delta into _a's dimensions
    //delta.reshape(_a.n_rows, _a.n_cols, _a.n_slices);
    dbg_assert(_a.n_rows == delta.n_rows);
    dbg_assert(_a.n_cols == delta.n_cols);
    dbg_assert(_a.n_slices == delta.n_slices);

    dbg_print("backward at logit layer(" << _units << ")");
    cube grad = delta % _a % (1.00f - _a);
    
    if(_prev)
        return(_prev->backward(grad));
    return(grad);
}


/**********************************************************************
*   MSE Class
***********************************************************************/
double MSE::cost(cube pred, cube y){
    dbg_assert(y.n_cols == 1 && y.n_slices == 1);
    pred.reshape(pred.n_elem,1,1);
    y.reshape(y.n_elem,1,1);
    
    double retn = 0.0f;
    for(int i=0; i < y.n_elem; i++) {
        retn += pow(pred(i,0,0) - y(i,0,0),2);
    }
    retn /= y.n_elem;
    retn *= 0.5f;
    //return(retn);
    return(0.5 * mean(mean(square(pred.slice(0) - y.slice(0)))));        
}
cube MSE::gradient(cube pred, cube y){
    dbg_assert(y.n_cols == 1 && y.n_slices == 1);
    cube retn = (pred - y) / (double)pred.n_elem;
    //retn = retn * (-1);
    return(retn);      
}
      

/**********************************************************************
*   ConvLayer Class
***********************************************************************/

ConvLayer::ConvLayer(int kernel_size, int depth) 
{
    _units = depth;
    _ksize = kernel_size;
        dbg_print("ConvLayer with " << _units << " units created.");
}

void ConvLayer::applyDw(double alpha)  { 
    for(int i=0; i < _units; i++) {
        _w(i) -= alpha * _dw(i);
        _b(0,0,i) -= alpha * _db(0,0,i);
    }
}    
bool ConvLayer::connectBack(Layer *l)
{   
    if(!l){
        dbg_print("null pointer to ConvLayer::connectBack()");
        return(1);
    }

    arma_rng::set_seed_random();
    int nr = l->getActivationCube().n_rows;
    int nc = l->getActivationCube().n_cols;
    int ns = l->getActivationCube().n_slices;

    _w    = field<cube>(_units);
    _dw   = field<cube>(_units);

    // NOTE: this is only valid for 1D fields
    for(int i=0; i < _units; i++) {
        _w(i)    = cube(_ksize, _ksize, ns, fill::randn);   
        _dw(i)   = cube(_ksize, _ksize, ns, fill::zeros);
    }

    _b    = cube(1     , 1     , _units, fill::randn); 
    _db   = cube(1     , 1     , _units, fill::zeros); 
    _a    = cube(nr    , nc    , _units, fill::zeros);
    _prev = l;
    dbg_print("connect back from ConvLayer with " << _units << " units." );
    return(0);

}

cube ConvLayer::forward(cube input) {
    
    cube result = zeros<cube>(
                    input.n_rows, 
                    input.n_cols, 
                    _units);

    input = addPadding(input, _ksize);
    for(int i=0; i < _units; i++){
        result.slice(i) = conv2d(input, _w(i)) + _b(0,0,i);
    }

    _a = result;

    if(_next) {
        return(_next->forward(result));
    }

    return(result);

}

cube ConvLayer::backward(cube delta) {
   
    // NOTE: delta may come from a linear layer
    delta.reshape(_a.n_rows, _a.n_cols, _a.n_slices);
    
    if(!_prev) {
        dbg_print("null pointer to _prev in ConvLayer::backward");
        return zeros<cube>(0,0,0);
    }

    // Compute Weight Updates
    cube input = addPadding(_prev->getActivationCube(), _ksize);
    for(int i = 0; i < _units; i++){
        // dz_dw
        for(int j=0; j < input.n_slices; j++) {
            _dw(i).slice(j) = conv2d(input.slice(j), delta.slice(i));
        }
        // dz_db
        _db(0,0,i) = accu(delta.slice(i));
    }
    

    // Compute next delta
    int nr = _prev->getActivationCube().n_rows;
    int nc = _prev->getActivationCube().n_cols;
    int ns = _prev->getActivationCube().n_slices;
    
    delta = addPadding(delta, _ksize);
    cube next_delta = zeros<cube>(nr,nc,ns);
    for(int i=0; i < ns; i++){
        for(int j=0; j < _units; j++) {
          next_delta.slice(i) += conv2d(
                                    delta.slice(j), 
                                    fliplr(flipud(_w(j).slice(i)))
                                    );
        }
    }

    if(_prev) {
        return(_prev->backward(next_delta));
    }

    return(next_delta);

}

LayerIter* ConvLayer::createIterator(std::string value) {
    return(new ConvLayerIter(this, value));
}

/**********************************************************************
*   Iterator Class
***********************************************************************/
int LinearLayerIter::next() {

    //cout << "(" << _i << "," 
                //<< _j << ","
                //<< _k << ","
                //<< _m << ")"
                //<< endl;

    // col-wise access
    if(_bflag) {
        _m++;   
        if(_m >= _b->n_rows) {
            _m = 0;
            _bflag = 0;
        }
    }
    else {
        _i++;
        if(_i >= _v->n_rows) {
            _i = 0;
            _j++;
        }
        if(_j >= _v->n_cols) {
            _j = 0;
            _k++;
        }
        if(_k >= _v->n_slices) {
            reset();
            return(0);
        }
    }
    return(1);
}

double LinearLayerIter::get() {
    if(_bflag) {
        return((*_b)(_m,0,0));
    } 
    else { 
        return((*_v)(_i,_j,_k));
    }
}

void LinearLayerIter::set(double value)  {
    if(_bflag) {
        (*_b)(_m,0,0) = value;
    }
    else {
        (*_v)(_i,_j,_k) = value;
    }
}
void LinearLayerIter::reset()      {
    _i =  0;
    _j =  0;
    _k =  0;
    _m =  0;
    _bflag = 1;
}


// ConvLayerIter
int ConvLayerIter::next() {
    // col-wise access
    /*
    cout << "(" << _i << "," 
                << _j << ","
                << _k << ","
                << _m << ")"
                << endl;
    */
    if(_bflag) {
        _bflag = 0;
        return(1);
    }
    
    _i++;
    if(_i >= (*_v)(_m).n_rows) {
        _i = 0;
        _j++;
    }
    if(_j >= (*_v)(_m).n_cols) {
        _j = 0;
        _k++;
    }
    if(_k >= (*_v)(_m).n_slices) {
        _i     = 0;
        _j     = 0;
        _k     = 0;
        _bflag = 1;
        _m++      ;
    }
    if(_m >= _v->n_elem) {
        reset();
        return(0);
    }
    return(1);
}

double  ConvLayerIter::get() {
    if(_bflag) {
        return((*_b)(0,0,_m));
    } 
    else { 
        return((*_v)(_m)(_i,_j,_k));
    }
}

void ConvLayerIter::set(double value)  {
    if(_bflag) {
        (*_b)(0,0,_m) = value;
    }
    else {
        (*_v)(_m)(_i,_j,_k) = value;
    }
}
void ConvLayerIter::reset()      {
    _i =  0;
    _j =  0;
    _k =  0;
    _m =  0;
    _bflag = 1;
}

mat addPadding(mat x, int ksize) {
    int offset = ksize/2;
    x.insert_rows(0, offset);
    x.insert_rows(x.n_rows, offset);
    x.insert_cols(0, offset);
    x.insert_cols(x.n_cols, offset);
    return(x);
}

cube addPadding(cube x, int ksize) {
    int offset = ksize/2;
    cube result = zeros<cube>(
                    x.n_rows + offset * 2,
                    x.n_cols + offset * 2,
                    x.n_slices
                    );
    result.tube(
            offset, 
            offset,
            offset + x.n_rows - 1,
            offset + x.n_cols - 1
            ) = x;
                
    return(result);
}



// Volumetric 2D Convolution
mat conv2d(cube image, cube kernel){
 
    dbg_assert(image.n_rows == image.n_cols);
    dbg_assert(kernel.n_rows == kernel.n_cols);
    dbg_assert(kernel.n_rows <= image.n_rows);
    dbg_assert(kernel.n_slices == image.n_slices);

    int kernel_size = kernel.n_rows;
    int image_size  = image.n_rows;

    int result_size = image_size - kernel_size + 1;
    mat result      = zeros<mat>(result_size, result_size);

    for(int row=0; row < result_size; row++){
        for(int col=0; col < result_size; col++){
            result(row, col) = accu(
                                 image.tube(
                                    row, 
                                    col, 
                                    row + kernel_size - 1,
                                    col + kernel_size - 1
                                 ) % kernel 
                               );
        }
    }
    
    return(result);
       
}

// Performs a Valid Convolution
// To perform Full Convolution, add proper padding
// to image before calling this function.
mat conv2d(mat image, mat kernel) {
   
    /*
    mat result(image);
    int offset = kernel.n_rows/2;
    int maxrows = image.n_rows + offset;
    int maxcols = image.n_cols + offset;

    // Add padding
    image.insert_rows(0, offset);
    image.insert_rows(image.n_rows, offset);
    image.insert_cols(0, offset);
    image.insert_cols(image.n_cols, offset);
    */

    dbg_assert(image.n_rows == image.n_cols);
    dbg_assert(kernel.n_rows == kernel.n_cols);
    dbg_assert(kernel.n_rows <= image.n_rows);

    int kernel_size = kernel.n_rows;
    int image_size  = image.n_rows;
    int result_size = image_size - kernel_size + 1;
    mat result      = zeros<mat>(result_size, result_size);

    for(int row=0; row < result_size; row++){
        for(int col=0; col < result_size; col++){
            result(row, col) = accu(
                                 image.submat(
                                    row, 
                                    col, 
                                    row + kernel_size - 1,
                                    col + kernel_size - 1
                                 ) % kernel 
                               );
        }
    }
    
    return(result);
    
}
