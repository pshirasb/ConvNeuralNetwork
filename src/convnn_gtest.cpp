#include "convnn.h"
#include "gtest/gtest.h"

/**********************************************************
 * Helper Methods
***********************************************************/
template <class T>
bool armaEQ(T a, T b) {
    if(a.n_rows != b.n_rows ||
       a.n_cols != b.n_cols ||
       a.n_elem != b.n_elem) {
       cout << "dimension mistmatch: "  
            << "[" << a.n_rows << " x " << a.n_cols << "]"
            << " != " 
            << "[" << b.n_rows << " x " << b.n_cols << "]"
            << endl;
       return 0;
    }

    double result = accu(abs(a - b)) / a.n_elem;
    if(result < 0.00001) {
        return 1;
    } 
    else {
        cout << a << endl; 
        cout << b << endl;
        return 0;
    }
}

// Convert [n,m] matrix to [n,m,1] cube
cube mat2cube(mat a) {
    cube result = zeros<cube>(a.n_rows, a.n_cols, 1);
    result.slice(0) = a;
    return result;
}


/**********************************************************
 * BasicLayerTest Case
***********************************************************/
template <class T>
Layer* createLayer(int size);

template <>
Layer* createLayer<LogitLayer>(int size) {
    return new LogitLayer();
}

template <>
Layer* createLayer<LinearLayer>(int size) {
    return new LinearLayer(size);
}

template <>
Layer* createLayer<ConvLayer>(int size) {
    return new ConvLayer(3,size);
}
template <>
Layer* createLayer<maxPoolLayer>(int size) {
    return new maxPoolLayer(3);
}

template <class T>
class BasicLayerTest: public ::testing::Test {
protected:
    BasicLayerTest() : _l(createLayer<T>(5)) {}
    ~BasicLayerTest() {
        delete _l;
    }
    Layer* _l;
};

using testing::Types;

typedef Types<LogitLayer,LinearLayer,ConvLayer,maxPoolLayer> LayerTypes;
TYPED_TEST_CASE(BasicLayerTest, LayerTypes);

TYPED_TEST(BasicLayerTest, InitializationTest) {
    EXPECT_EQ(NULL, this->_l->next);
    EXPECT_EQ(NULL, this->_l->prev);
    //EXPECT_EQ(NULL, this->_l->units);
    //EXPECT_EQ(0, this->_l->getActivationCube().n_rows);
    //EXPECT_EQ(0, this->_l->getActivationCube().n_cols);
    //EXPECT_EQ(0, this->_l->getActivationCube().n_slices);
}

TYPED_TEST(BasicLayerTest, EmptyForwardTest) {
    cube x;
    cube retn = this->_l->forward(x);
    EXPECT_EQ(0, retn.n_rows);
    EXPECT_EQ(0, retn.n_cols);
    EXPECT_EQ(0, retn.n_slices);
}

TYPED_TEST(BasicLayerTest, EmptyBackwardTest) {
    cube x;
    cube retn = this->_l->backward(x);
    EXPECT_EQ(0, retn.n_rows);
    EXPECT_EQ(0, retn.n_cols);
    EXPECT_EQ(0, retn.n_slices);
}

TYPED_TEST(BasicLayerTest, ConnectTest) {
    LinearLayer l(5);
    this->_l->connect(&l);
    EXPECT_EQ(&l, this->_l->next);
    EXPECT_EQ(this->_l, l.prev);
}

TYPED_TEST(BasicLayerTest, ConnectBackTest) {
    LinearLayer l(5);
    l.connect(this->_l);
    EXPECT_EQ(&l, this->_l->prev);
    EXPECT_EQ(this->_l, l.next);
}
TYPED_TEST(BasicLayerTest, TwoLayerForwardTest) {
    InputLayer p(5,5,1);
    p.connect(this->_l);
    cube x(5,5,1,fill::randu);
    cube y = p.forward(x);
    EXPECT_TRUE(armaEQ(y, this->_l->getActivationCube()));
    EXPECT_TRUE(armaEQ(y, this->_l->forward(x)));

}
/**********************************************************
 * InputLayerTest Case
***********************************************************/
TEST(InputLayerTest, ForwardTest) {
    InputLayer l(2,3,1);
    cube cx(2,3,4,fill::randn);
    EXPECT_TRUE(armaEQ(cx , l.forward(cx)));
    EXPECT_TRUE(armaEQ(cx, l.getActivationCube()));
    cube cy(2,3,1,fill::randu);
    EXPECT_TRUE(armaEQ(cy , l.forward(cy)));
    EXPECT_TRUE(armaEQ(cy, l.getActivationCube()));
}

TEST(InputLayerTest, BackwardTest) {
    InputLayer l(2,3,1);
    cube cx(2,3,4,fill::randn);
    EXPECT_TRUE(armaEQ(cx , l.backward(cx)));
}

/**********************************************************
 * LogitLayerTest Case
***********************************************************/
TEST(LogitLayerTest, ForwardTest1) {
    LogitLayer  l;
    mat x = {-5,-4,-3,-2,-1,0,1,2,3,4,5};
    mat y = {0.006692851,0.017986210,0.047425873,
             0.119202922,0.268941421,0.500000000,
             0.731058579,0.880797078,0.952574127,
             0.982013790,0.993307149};
    x.reshape(11,1);
    y.reshape(11,1);
    cube cx = mat2cube(x);
    cube cy = mat2cube(y);
    EXPECT_TRUE(armaEQ(cy, l.forward(cx)));
    EXPECT_TRUE(armaEQ(cy, l.getActivationCube()));
}
TEST(LogitLayerTest, ForwardTest2) {
    LogitLayer  l;
    mat x = {-5,-4,-3,-2,-1,0,1,2,3,4};
    mat y = {0.006692851,0.017986210,0.047425873,
             0.119202922,0.268941421,0.500000000,
             0.731058579,0.880797078,0.952574127,
             0.982013790};
    x.reshape(2,5);
    y.reshape(2,5);
    cube cx = mat2cube(x);
    cube cy = mat2cube(y);
    EXPECT_TRUE(armaEQ(cy, l.forward(cx)));
    EXPECT_TRUE(armaEQ(cy, l.getActivationCube()));
}

TEST(LogitLayerTest, SingleLayerDeltaTest1) {
    InputLayer p(1,1,1);
    LogitLayer l;
    double epsilon = 0.000001;
    p.connect(&l);
    cube x(1,1,1,fill::randu);
    cube y = p.forward(x);
    
    cube pred_delta = l.backward(ones<cube>(1,1,1));

    x = x + epsilon;
    cube y_plus = p.forward(x);
    x = x - (2 * epsilon);
    cube y_minus = p.forward(x);
    cube approx_delta = (y_plus - y_minus) / (2 * epsilon);
    EXPECT_NEAR(as_scalar(approx_delta), 
                as_scalar(pred_delta),
                1e-5);
}

/**********************************************************
 * SingleLayerGradientTest Case
***********************************************************/

template <class T>
class SingleLayerGradientTest: public ::testing::Test {
protected:
    SingleLayerGradientTest() : _l(createLayer<T>(10)) {}
    ~SingleLayerGradientTest() {
        delete _l;
    }
    Layer* _l;
};
TYPED_TEST_CASE(SingleLayerGradientTest, LayerTypes);

TYPED_TEST(SingleLayerGradientTest, DeltaTest) {
    
    const int    row     = 5;
    const int    col     = 5;
    const int    slices  = 3;
    const double epsilon = 0.000001;
    InputLayer p(row,col,slices);

    p.connect(this->_l);
    cube x(row, col, slices, fill::randu);
    cube y = p.forward(x);
    
    cube pred_delta = this->_l->backward(
                                 ones<cube>(
                                   this->_l->getActivationCube().n_rows,
                                   this->_l->getActivationCube().n_cols,
                                   this->_l->getActivationCube().n_slices
                                 )
                                );
    // In case the shape doesn't match
    pred_delta.reshape(row,col,slices);

    for(int i=0; i < row; i++) {
        for(int j=0; j<col; j++) {
            for(int k=0; k<slices; k++) {
                x(i,j,k) = x(i,j,k) + epsilon;
                cube y_plus  = p.forward(x);
                x(i,j,k) = x(i,j,k) - (2 * epsilon);
                cube y_minus = p.forward(x);
                x(i,j,k) = x(i,j,k) + epsilon;

                cube approx_delta = (y_plus - y_minus) / (2 * epsilon);
                EXPECT_NEAR(accu(approx_delta), pred_delta(i,j,k), 
                        1e-5);
            }
        }
    }
}
TYPED_TEST(SingleLayerGradientTest, GradientTest) {
    
    const int    row     = 5;
    const int    col     = 5;
    const int    slices  = 3;
    const double epsilon = 0.000001;
    InputLayer p(row,col,slices);

    p.connect(this->_l);
    cube x(row, col, slices, fill::randu);
    cube y = p.forward(x);
    
    cube pred_delta = this->_l->backward(
                                 ones<cube>(
                                   this->_l->getActivationCube().n_rows,
                                   this->_l->getActivationCube().n_cols,
                                   this->_l->getActivationCube().n_slices
                                 )
                                );
    // In case the shape doesn't match
    pred_delta.reshape(row,col,slices);

    LayerIter*  w_itr = this->_l->createIterator("w");
    LayerIter* dw_itr = this->_l->createIterator("dw");
    if(w_itr && dw_itr) {
        do {
            double old_w  = w_itr->get();
            double old_dw = dw_itr->get();
            w_itr->set(old_w + epsilon);
            cube y_plus  = p.forward(x);

            w_itr->set(old_w - epsilon);
            cube y_minus = p.forward(x);

            w_itr->set(old_w);
            dw_itr->set(old_dw);

            double approx = accu((y_plus - y_minus) / (2 * epsilon));
            EXPECT_NEAR(approx , old_dw, 1e-5);

        } while(w_itr->next() && dw_itr->next());
    
        delete w_itr;
        delete dw_itr;
    }

}

TEST(MSELayerTest, CostTest) {
    
    InputLayer l(5,1,1);
    MSE cf;
    mat pred = {1,2,3,5,7,9};
    mat ref  = {8,6,5,4,3,1};
    double target = 12.5f;
    cube cpred = mat2cube(pred);
    cube cref  = mat2cube(ref);
    EXPECT_NEAR(cf.cost(cpred, cref),
                target,
                1e-6);
}

TEST(MSELayerTest, GradientTest) {
    
    InputLayer l(5,1,1);
    MSE cf;
    vec pred = {1,2,3,5,7,9};
    vec ref  = {8,6,5,4,3,1};
    double epsilon = 1e-6;
    cube cpred = mat2cube(pred);
    cube cref  = mat2cube(ref);
    cube grad  = cf.gradient(cpred, cref);

    for(int i=0; i < cpred.n_rows; i++) {
        cpred(i,0,0) = cpred(i,0,0) + epsilon;
        double y_plus = cf.cost(cpred, cref);

        cpred(i,0,0) = cpred(i,0,0) - (2 * epsilon);
        double y_minus = cf.cost(cpred, cref);

        cpred(i,0,0) = cpred(i,0,0) + epsilon;
       
        double approx = (y_plus - y_minus) / (2 * epsilon);
        EXPECT_NEAR(approx, grad(i,0,0), 1e-6);

    }
}

/**********************************************************
 * MultiLayerGradientTest Case
***********************************************************/

template <class T>
class MultiLayerGradientTest: public ::testing::Test {
protected:
    MultiLayerGradientTest() 
        : _in (new InputLayer(_row, _col, _slices))
        , _l  (createLayer<T>(10))
        , _out(new LogitLayer())
        , _cf (new MSE())
    {
        _x = randu<cube>(_row, _col, _slices);
        _in->connect(_l);
        _l->connect(_out);
    }
    ~MultiLayerGradientTest() {
        delete _l;
        delete _in;
        delete _cf;
    }
    
    Layer*        _in ;
    Layer*        _out;
    Layer*        _l  ;
    CostFunction* _cf ;
    static const int    _row     = 5;
    static const int    _col     = 5;
    static const int    _slices  = 3;
    cube _x;
    cube _y;
};
TYPED_TEST_CASE(MultiLayerGradientTest, LayerTypes);

TYPED_TEST(MultiLayerGradientTest, DeltaTest) {
    
    const double epsilon = 0.000001;

    this->_y = randu<cube>(this->_out->getActivationCube().n_rows,
                this->_out->getActivationCube().n_cols,
                this->_out->getActivationCube().n_slices
                );
    cube y    = this->_in->forward(this->_x);

    cube pred_delta = this->_out->backward(
                                 this->_cf->gradient(y, this->_y));

    int row = this->_row;
    int col = this->_col;
    int slices = this->_slices;

    // In case the shape doesn't match
    pred_delta.reshape(row,col,slices);

    for(int i=0; i < row; i++) {
        for(int j=0; j<col; j++) {
            for(int k=0; k<slices; k++) {
                this->_x(i,j,k) = this->_x(i,j,k) + epsilon;
                double y_plus  = this->_cf->cost(
                                this->_in->forward(this->_x),
                                this->_y);
                                
                this->_x(i,j,k) = this->_x(i,j,k) - (2 * epsilon);
                double y_minus  = this->_cf->cost(
                                this->_in->forward(this->_x),
                                this->_y);
                this->_x(i,j,k) = this->_x(i,j,k) + epsilon;

                double approx_delta = (y_plus - y_minus) / (2 * epsilon);
                EXPECT_NEAR((approx_delta), pred_delta(i,j,k), 
                        1e-5);
            }
        }
    }
}

TYPED_TEST(MultiLayerGradientTest, GradientTest) {
    
    const double epsilon = 0.000001;
    this->_y = randu<cube>(this->_out->getActivationCube().n_rows,
                this->_out->getActivationCube().n_cols,
                this->_out->getActivationCube().n_slices
                );
    cube y    = this->_in->forward(this->_x);

    cube pred_delta = this->_out->backward(
                                 this->_cf->gradient(y, this->_y));

    int row = this->_row;
    int col = this->_col;
    int slices = this->_slices;
    // In case the shape doesn't match
    pred_delta.reshape(row,col,slices);

    LayerIter*  w_itr = this->_l->createIterator("w");
    LayerIter* dw_itr = this->_l->createIterator("dw");
    if(w_itr && dw_itr) {
        do {
            double old_w  = w_itr->get();
            double old_dw = dw_itr->get();
            w_itr->set(old_w + epsilon);
            double y_plus  = this->_cf->cost(
                                this->_in->forward(this->_x),
                                this->_y);

            w_itr->set(old_w - epsilon);
            double y_minus = this->_cf->cost(
                                this->_in->forward(this->_x),
                                this->_y);

            w_itr->set(old_w);
            dw_itr->set(old_dw);

            double approx = ((y_plus - y_minus) / (2 * epsilon));
            EXPECT_NEAR(approx , old_dw, 1e-5);

        } while(w_itr->next() && dw_itr->next());
    
        delete w_itr;
        delete dw_itr;
    }

}

/**********************************************************
 * TwoLayerGradientTest Case
***********************************************************/

template <class T>
class TwoLayerGradientTest: public ::testing::Test {
protected:
    TwoLayerGradientTest() 
        : _in (new InputLayer(_row, _col, _slices))
        , _l  (createLayer<T>(10))
        , _out(new LogitLayer())
    {
        _x = randu<cube>(_row, _col, _slices);
        _in->connect(_l);
        _l->connect(_out);
    }
    ~TwoLayerGradientTest() {
        delete _l;
        delete _in;
    }
    
    Layer*        _in ;
    Layer*        _l  ;
    Layer*        _out;
    static const int    _row     = 9;
    static const int    _col     = 9;
    static const int    _slices  = 9;
    cube _x;
    cube _y;
};
TYPED_TEST_CASE(TwoLayerGradientTest, LayerTypes);

TYPED_TEST(TwoLayerGradientTest, DeltaTest) {
    
    const double epsilon = 0.000001;

    int prow = this->_out->getActivationCube().n_rows;
    int pcol = this->_out->getActivationCube().n_cols;
    int pslices = this->_out->getActivationCube().n_slices;

    this->_y = randu<cube>(prow, pcol, pslices);
    cube y    = this->_in->forward(this->_x);

    cube pred_delta = this->_out->backward(
                        ones<cube>(prow, pcol, pslices));

    int row = this->_row;
    int col = this->_col;
    int slices = this->_slices;

    // In case the shape doesn't match
    ASSERT_EQ(pred_delta.n_elem,(row * col * slices));
    pred_delta.reshape(row,col,slices);

    for(int i=0; i < row; i++) {
        for(int j=0; j<col; j++) {
            for(int k=0; k<slices; k++) {
                this->_x(i,j,k) = this->_x(i,j,k) + epsilon;
                cube y_plus  = this->_in->forward(this->_x);
                                
                this->_x(i,j,k) = this->_x(i,j,k) - (2 * epsilon);
                cube y_minus  = this->_in->forward(this->_x);
                                
                this->_x(i,j,k) = this->_x(i,j,k) + epsilon;

                cube approx_delta = (y_plus - y_minus) / (2 * epsilon);
                EXPECT_NEAR(accu(approx_delta), pred_delta(i,j,k), 
                        1e-5);
            }
        }
    }
}

/**********************************************************
 * NeuralNetTest Case
***********************************************************/
TEST(NeuralNetTest, AddLayerTest) {

    NeuralNet nn;
    InputLayer  a(2,3,4);
    ConvLayer   b(3,10);
    LinearLayer c(5);
    LogitLayer  d;
    MSE         e;

    nn.add(&a);
    nn.add(&b);
    nn.add(&c);
    nn.add(&d);
    nn.add(&e);

    EXPECT_TRUE(a.next == &b);
    EXPECT_TRUE(b.next == &c);
    EXPECT_TRUE(c.next == &d);
    EXPECT_TRUE(d.next == NULL);
    EXPECT_TRUE(a.prev == NULL);
    EXPECT_TRUE(b.prev == &a);
    EXPECT_TRUE(c.prev == &b);
    EXPECT_TRUE(d.prev == &c);
}

TEST(NeuralNetTest, ComputeCostTest) {

    NeuralNet nn;
    InputLayer  a(7,7,3);
    ConvLayer   b(3,10);
    LinearLayer c(5);
    LogitLayer  d;
    MSE         e;
  
    nn.add(&a);
    nn.add(&b);
    nn.add(&c);
    nn.add(&d);
    nn.add(&e);

    cube x(7,7,3, fill::randn);
    cube y(5,1,1, fill::randu);

    cube prediction = a.forward(x);
    double prediction_cost = e.cost(prediction, y);
    
    a.forward(zeros<cube>(7,7,3));  // Clear activation values

    EXPECT_NEAR(nn.computeCost(x,y), prediction_cost, 1e-5);
    
}
