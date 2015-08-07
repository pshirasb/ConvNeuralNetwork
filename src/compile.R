require(RcppArmadillo)

PKG_LIBS <- sprintf("%s $(LAPACK_LIBS) $(BLAS_LIBS) $(FLIBS) -fopenmp",
                    Rcpp:::RcppLdFlags() )
PKG_CPPFLAGS <- sprintf( "%s %s -fopenmp -std=c++0x", 
                    Rcpp:::RcppCxxFlags(),
                    RcppArmadillo:::RcppArmadilloCxxFlags() )

Sys.setenv(
    PKG_LIBS = PKG_LIBS ,
    PKG_CPPFLAGS = PKG_CPPFLAGS
)

R <- file.path( R.home(component = "bin"), "R" )
system( sprintf("%s CMD SHLIB %s", R, 
                paste( commandArgs(TRUE),collapse = " ") ) )
