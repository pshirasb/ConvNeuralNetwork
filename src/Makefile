# Makefile for Conv

# compiler
CC = g++
# pre-processor settings (-I, -D, etc)
CPPFLAGS = -D_EXEC -std=c++0x 
# compiler settings
CXXFLAGS = -Wall -D_EXEC -fopenmp -g 
# linker settings
LDFLAGS  = -larmadillo
# linker settings for test
GLDFLAGS = -lgtest_main -lgtest

INCLUDE    = `Rscript -e "RcppArmadillo:::CxxFlags()"`
RFLAGS     = -o convnn.so --preclean --clean

#TODO: DO NOT REMOVE
#out: $(FILES) $(MAIN)
	 #$(CC) -o out $(CFLAGS) $(LDFLAGS) $(FILES) $(MAIN) $(HEADERS)
	 ##Rscript compile.R $(RFLAGS) $(FILES)

#gtest: $(TEST_FILES) 
	   #$(CC) -o gtest $(CFLAGS) $(LDFLAGS) $(TEST_FLAGS) $(FILES) $(TEST_FILES) $(HEADERS)


SHARED_SRCS = convnn.cpp
SHARED_DEPS = $(SHARED_SRCS:.cpp=.d)
SHARED_OBJS = $(SHARED_SRCS:.cpp=.o)

EXEC_SRCS = main.cpp
EXEC_DEPS = $(EXEC_SRCS:.cpp=.d)
EXEC_OBJS = $(EXEC_SRCS:.cpp=.o)
EXEC_BIN  = out

TEST_SRCS = convnn_gtest.cpp
TEST_DEPS = $(TEST_SRCS:.cpp=.d)
TEST_OBJS = $(TEST_SRCS:.cpp=.o)
TEST_BIN  = gtest

.PHONY: all bin test clean

all : bin test
bin : $(EXEC_BIN)
test: $(TEST_BIN)

-include $(EXEC_DEPS)
-include $(SHARED_DEPS)
-include $(TEST_DEPS)

# Executable
$(EXEC_BIN): $(SHARED_OBJS) $(EXEC_OBJS) 
	@echo CC $^
	@$(CC) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

# Test Suite
$(TEST_BIN): $(SHARED_OBJS) $(TEST_OBJS)
	@echo CC $^
	@$(CC) $(CXXFLAGS) $(LDFLAGS) $(GLDFLAGS) $^ -o $@

.cpp.o:
	@echo CC $<
	@$(CC) $(CPPFLAGS) -MMD -c $< -o $@
	
clean:
	-rm $(TEST_OBJS) $(TEST_DEPS) $(TEST_BIN)
	-rm $(EXEC_OBJS) $(EXEC_DEPS) $(EXEC_BIN)
	-rm $(SHARED_OBJS) $(SHARED_DEPS) 
