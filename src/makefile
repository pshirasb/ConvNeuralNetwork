# Makefile for Conv

CC = g++
CFLAGS     = -std=c++0x -D_EXEC -fopenmp -g
LDFLAGS    = -larmadillo 
TEST_FLAGS = -lgtest_main -lgtest
FILES      = convnn.cpp 
MAIN       = main.cpp
TEST_FILES = convnn_gtest.cpp
HEADERS   = convnn.h
INCLUDE   = `Rscript -e "RcppArmadillo:::CxxFlags()"`
RFLAGS    = -o convnn.so --preclean --clean

all: out gtest

out: $(FILES) $(MAIN)
	 $(CC) -o out $(CFLAGS) $(LDFLAGS) $(FILES) $(MAIN) $(HEADERS)
	 #Rscript compile.R $(RFLAGS) $(FILES)

gtest: $(TEST_FILES) 
	   $(CC) -o gtest $(CFLAGS) $(LDFLAGS) $(TEST_FLAGS) $(FILES) $(TEST_FILES) $(HEADERS)
