////////////////////////////////////////////////////////////////////////
//
// Disclaimer:
// This code and instructions are modified versions of "ADMM Decoder"
// downloaded from https://sites.google.com/site/xishuoliu/codes.
// Full credits goes to original authors.
//
// To use in MATLAB:
//      (1)  Set MATLAB working directory to the directory of this file.
//      (2)  Run command "mex -setup".
//      (3)  Choose a compiler (e.g. Visual Studio, MinGW64 etc).
//      (4)  Run command "mex proj_pp_mex.cpp"
//      (5)  Alternatively, you can use g++ compiler and switch on "O3" by "mex -v CC=g++ LD=g++ COPTIMFLAGS='-O3' proj_pp_mex.cpp"
//   2. Use the function:
//       x = proj_pp_mex(v)
//       v -- a column vector (N-by-1) which is the vector you want to project
//       x -- results of the projection   
//   3. Examples:
//      (1) x = proj_pp_mex(2*rand(10,1));
//
////////////////////////////////////////////////////////////////////////

#include <mex.h>
#include "projection.cpp"

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray *prhs[]){
    if(nrhs != 0)
    {
        const mxArray *matrixIn = prhs[0];
        int m = (int)mxGetM(matrixIn);
        int n = (int)mxGetN(matrixIn);

        if(m > 1 && n > 1)
        {
           mexErrMsgTxt("Function only takes vectors!");
        }
        if(m == 0 || n == 0)
        {
            mexErrMsgTxt("Function does not take empty matrix!");
        }
        if(m == 1 && n == 1)
        {
            mexErrMsgTxt("Function does not take scalars!");
        }
        if(m > 1 && n == 1)
        {
            plhs[0] = mxCreateDoubleMatrix(m,n,mxREAL);
            double *in;
            double *out;
            in = mxGetPr(prhs[0]);
            out = mxGetPr(plhs[0]);
            ProjPolytope(in, out, m, n);
        }
        if(m == 1 && n > 1)
        {
            plhs[0] = mxCreateDoubleMatrix(m,n,mxREAL);
            double *in;
            double *out;
            in = mxGetPr(prhs[0]);
            out = mxGetPr(plhs[0]);
            ProjPolytope(in, out, n, m);
        }
    }
    else
        mexErrMsgTxt("No input!");
}
