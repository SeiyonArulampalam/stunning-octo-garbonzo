# FILE=_fill_in_0.cpp
FILE=2_ssparse_solve.cpp

default:

ssparse:
	g++ -I/usr/include/suitesparse ${FILE} -L/usr/local/lib -lsuitesparseconfig -lsuitesparse_mongoose -lcholmod -Wl,-rpath=/usr/local/lib
	
cusparse:
	nvcc -Xcompiler -I${HOME}/git/stunning-octo-garbonzo/include/ solve_cusparse.cu -o solve.x -lcusparse