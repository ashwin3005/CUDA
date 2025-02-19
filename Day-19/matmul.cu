#include <stdio.h>

int matMul(float *A, float *B, float *RES, int arow, int acol, int brow, int bcol){
    if(acol != brow){
        printf("Not possible for multipling the matrices, check for the dimensions !");
        return 1;
    }
    int res_row = arow;
    int res_col = bcol;

    for(int i=0; i<res_row; i++){
        for(int j=0; j<res_col; j++){
            float dotprod = 0.0f;
            for(int k=0; k<acol; k++){
                dotprod += A[i*acol + k] * B[k*bcol + j];
            }
            RES[i*res_col + j] = dotprod;   
        }
    }
    return 0;
}
void printMatrix(float *A, int rows, int cols);
int main(){
    int arow = 2;
    int acol = 3;
    int brow = 3;
    int bcol = 2;
    float *A, *B, *RES;
    A = (float*)malloc(arow*acol*sizeof(float));
    B = (float*)malloc(brow*bcol*sizeof(float));
    RES = (float*)malloc(arow*bcol*sizeof(float));

    printf("Enter the inputs for A matrix.\n");
    for(int i=0; i<arow; i++){
        for(int j=0; j<acol; j++){
            float in;
            printf("Enter value for (%d, %d): ", i, j);
            scanf("%f", &in);
            A[i* acol + j] = in;
        }
    }
    printf("Enter the inputs for B matrix.\n");
    for(int i=0; i<brow; i++){
        for(int j=0; j<bcol; j++){
            float in;
            printf("Enter value for (%d, %d): ", i, j);
            scanf("%f", &in);
            B[i* bcol + j] = in;
        }
    }
    matMul(A, B, RES, arow, acol, brow, bcol);

    printf("Matrix A:\n");
    printMatrix(A, arow, acol);
    printf("Matrix B:\n");
    printMatrix(B, brow, bcol);
    printf("Result Matrix:\n");
    printMatrix(RES, arow, bcol);
}

void printMatrix(float *A, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", A[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}