#include "convnn.h"
#include <omp.h>
#include <iostream>
#include <iomanip>
using namespace std;
using namespace arma;

#ifndef _EXEC
RcppExport SEXP Rconv2d(SEXP x, SEXP k, SEXP p) {
    return(Rcpp::wrap(conv2d(Rcpp::as<mat>(x), Rcpp::as<mat>(k),Rcpp::as<int>(p))));
}
#endif

mat class2mat(const mat& x) {
    
    dbg_assert(x.n_cols == 1);
    mat result = mat(x.n_rows, 10, fill::zeros);

    for(int i = 0; i < x.n_rows; i++) {
        result(i, x(i,0)) = 1;
    }
    return(result);
}


int main(){

    /*    
    mat x = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,
             24,25};
    x.reshape(5,5);

    mat y = {4,5};

    cube cx(5,5,1);
    cx.slice(0) = x;
   
    cx = randu<cube>(10,10,1);
    cube cy(10,10,50,fill::randu);


    NeuralNet nn;
    nn.add( new InputLayer(cx.n_rows, cx.n_cols, cx.n_slices) );
    nn.add( new ConvLayer(5,50) );
    //nn.add( new LogitLayer());
    //nn.add( new LinearLayer(10) );
    nn.add( new LogitLayer());
    nn.add( new MSE() );
    
    cout << nn.checkGradients(cx,cy) << endl;
    */

     
    mat data;
    data.load("../data/mnist.csv", csv_ascii);
    mat x = data.tail_cols(data.n_cols - 1);
    mat y = class2mat(data.head_cols(1));

    cube cx(x.n_rows, x.n_cols, 1);
    cx.slice(0) = x;
    cx.reshape(x.n_cols, 1, x.n_rows, 1);

    cube cy(y.n_rows, y.n_cols, 1);
    cy.slice(0) = y;
    cy.reshape(y.n_cols, 1, y.n_rows, 1);
    
    cout << data.n_rows << ", " << data.n_cols << endl;


    NeuralNet nn;
    nn.add( new InputLayer(cx.n_rows, cx.n_cols, 1));
    nn.add( new LinearLayer(10) );
    //nn.add( new LogitLayer() );
    //nn.add( new LinearLayer(10) );
    //nn.add( new LogitLayer() );
    nn.add( new MSE() );
    //nn.printAll();
    //cout << nn.pred(x) << endl;
    //nn.backprop(x,y);
    cout << "Gradient Accuracy = " << nn.checkGradients(cx.slices(0,0),cy.slices(0,0)) << endl;
    //cout << nn.pred(cx.slices(0,0)) << endl;
    //nn.train(cx, cy, 0.01, 2); 
    //cout << "Gradient Accuracy = " << nn.checkGradients(cx.slices(0,0),cy.slices(0,0)) << endl;


    /*
    mat x = {1,1,1,0,0,1,0,0,1,1};
    x.reshape(2,4);
    cube cx(2,4,1);
    cx.slice(0) = x;
    cx.reshape(2,1,4);

    mat y = {0,1,1,0};
    y.reshape(1,4);
    cube cy(1,4,1);
    cy.slice(0) = y;
    cy.reshape(1,1,4);

    cout << x << endl;
    cout << cy << endl;

    NeuralNet nn;
    nn.add( new InputLayer(2,1,1) );
    nn.add( new LinearLayer(1) );
    nn.add( new LogitLayer() );
    nn.add( new MSE() );
    //nn.train(cx,cy, 0.01, 200);
    cout << "Gradient Acc = " << nn.checkGradients(cx.slices(1,1), cy.slices(1,1)) << endl;
    */   
    return 0;
}

/*
mat conv2d(mat image, mat kernel, int parallel) {
    
    mat result(image);
    int offset = kernel.n_rows/2;
    int maxrows = image.n_rows + offset;
    int maxcols = image.n_cols + offset;

    // Add padding
    image.insert_rows(0, offset);
    image.insert_rows(image.n_rows, offset);
    image.insert_cols(0, offset);
    image.insert_cols(image.n_cols, offset);

    #pragma omp parallel for if(parallel) 
    for(int row=offset; row < maxrows; row++){
        for(int col=offset; col < maxcols; col++){
            int fcol = col - offset;
            int lcol = col + offset;
            int frow = row - offset;
            int lrow = row + offset;
            result(row - offset, col - offset) = 
                accu(image.submat(frow,fcol,lrow,lcol) % kernel);

        }
    }
    return(result);
}
*/
