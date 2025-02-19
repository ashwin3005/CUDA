/*
Trying to implement a simple linear layer
*/

#include <stdio.h>

void linear_layer(float* input, float* output, float* weights, float* bias, int input_size, int output_size) {
    for (int i = 0; i < output_size; i++) {
        output[i] = 0.0f;  
        for (int j = 0; j < input_size; j++) {
            output[i] += input[j] * weights[i * input_size + j];  // Multiply and sum
        }
        output[i] += bias[i];  // Add bias
    }
}

int main() {
    int input_size = 3;   
    int output_size = 2;  

    // input layer
    float input[3] = {1.0f, 2.0f, 3.0f};

    // weights matrix (2 X 3)
    float weights[6] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};  

    // bias
    float bias[2] = {0.1f, 0.2f};

    float output[2];

    // function call
    linear_layer(input, output, weights, bias, input_size, output_size);

    printf("Output: ");
    for (int i = 0; i < output_size; i++) {
        printf("%f ", output[i]);
    }
    printf("\n");

    return 0;
}
