#include <stdio.h>

int main()
{
    int matrix_width = 3;

    int *matrix_a_host;
    int *matrix_b_host;
    int *matrix_c_host;

    matrix_a_host = (int *)malloc(matrix_width*matrix_width*sizeof(int));
    matrix_b_host = (int *)malloc(matrix_width*matrix_width*sizeof(int));
    matrix_c_host = (int *)malloc(matrix_width*matrix_width*sizeof(int));

    for(int row = 0; row < matrix_width; row++)
    {
        for(int col = 0; col < matrix_width; col++)
        {
            matrix_a_host[row * matrix_width + col] = row + col;
            matrix_b_host[row * matrix_width + col] = row * col + col;
        }
    }

    for(int row = 0; row < matrix_width; row++)
    {
        for(int col = 0; col < matrix_width; col++)
        {
            int single_element = 0;
            for(int k = 0; k < matrix_width; k++)
            {
                single_element += matrix_a_host[row * matrix_width + k] * matrix_b_host[matrix_width  * k + col];
            }
            matrix_c_host[row * matrix_width + col] = single_element;
        }
    }

    printf("\n-------------Matrix a-----------------\n");
    for(int i = 0; i < matrix_width * matrix_width; i++)
    {
        if((i + 1) % matrix_width)
            printf("%d ", *(matrix_a_host + i));
        else
            printf("%d \n", *(matrix_a_host + i));
    }

    printf("\n-------------Matrix b-----------------\n");
    for(int i = 0; i < matrix_width * matrix_width; i++)
    {
        if((i + 1) % matrix_width)
            printf("%d ", *(matrix_b_host + i));
        else
            printf("%d \n", *(matrix_b_host + i));
    }

    printf("\n-------------Matrix c-----------------\n");
    for(int i = 0; i < matrix_width * matrix_width; i++)
    {
        if((i + 1) % matrix_width)
            printf("%d ", *(matrix_c_host + i));
        else
            printf("%d \n", *(matrix_c_host + i));
    }

    free(matrix_a_host);
    free(matrix_b_host);
    free(matrix_c_host);

    return 1;
}