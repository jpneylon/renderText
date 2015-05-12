#ifndef DEFS_H_INCLUDED
#define DEFS_H_INCLUDED
#include <cuda_runtime_api.h>

#endif // DEFS_H_INCLUDED

#define SUCCESS 1
#define FAILURE 0

#define PI 3.1416

/////////////////////////////// Structure Definitions ////////////////////////////////////////////////
typedef struct
{
    float3 inc;
    int3 count;
    float max;
    float min;
    float *matrix;
} FLOAT_GRID;

typedef struct
{
    float3 start;
    float3 inc;
    int3 count;
    float max;
    float min;
    int mslice;
    unsigned char *matrix;
} CHAR_GRID;

void writeToTextFile(const char *, FLOAT_GRID *);

#define GRID_VALUE(GRID_ptr, i, j, k)\
    ((GRID_ptr)->matrix[(i) + (GRID_ptr)->count.x * ((j) + ((k) * (GRID_ptr)->count.y))])

