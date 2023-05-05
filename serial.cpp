#include <iostream>
#include <cmath>
#include <chrono>

using namespace std;
using namespace std::chrono;

int N = 256;
const double h = 1.0/(N-1);

void load_input_data(double *A, double *B, double *x, int n){
    FILE *f;
    f = fopen("input.txt", "r");
    if (f == NULL) {
        printf("Error opening input file.\n");
        exit(1);
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n-1; j++) {
            fscanf(f, "%lf", &A[i*n + j]);
        }
        fscanf(f, "%lf", &B[i]);
        x[i] = 0.0;
    }
    fclose(f);
}

void jacobi(double* u, double* f, double* u_new, double h, double tol, int maxiter)
{
    int iter = 0;

    while (iter < maxiter)
    {
        // Jacobi iteration
        for (int i = 1; i < N-1; i++)
        {
            for (int j = 1; j < N-1; j++)
            {
                int index = i*N + j;
                u_new[index] = 0.25 * (u[index-N] + u[index-1] + u[index+1] + u[index+N] - h*h*f[index]);
            }
        }

        // Check for convergence
        double diff_norm = 0.0;
        for (int i = 1; i < N-1; i++)
        {
            for (int j = 1; j < N-1; j++)
            {
                int index = i*N + j;
                diff_norm += pow(u_new[index] - u[index], 2);
            }
        }
        diff_norm = sqrt(diff_norm)/(N-2)/(N-2);

        if (diff_norm < tol)
        {
            break;
        }

        // Swap u and u_new
        double* temp = u;
        u = u_new;
        u_new = temp;

        iter++;
    }
}

int jacobi_algorithm(int s){
    // size
    N = s;
    // generating matrix
    char command_str[300];
    sprintf(command_str, "python generate_matrix.py %d input.txt", s);
    system(command_str);


    double* u = new double[N*N];
    double* f = new double[N*N];
    double* u_new = new double[N*N];
    double tol = 1e-6;
    int maxiter = 1000;

    // Initialize u and f
    load_input_data(f, u, u_new, N);
    
    auto start = high_resolution_clock::now();

    // Solve using Jacobi method
    jacobi(u, f, u_new, h, tol, maxiter);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    cout << "Time: " << duration.count() << " ms" << endl;
    
    FILE *fp;
    fp = fopen("serial_results.txt", "a");
    if (fp == NULL){
        printf("Error in opening file !!\n");
        return -1;
    }
    fprintf(fp, "serial %d %f\n", s, duration.count());
    fclose(fp);

    // Free memory
    delete[] u;
    delete[] f;
    delete[] u_new;

    return 0;
}

int main(){
    for(int s = 100; s < 1050; s += 50){
        jacobi_algorithm(s);
    }
}