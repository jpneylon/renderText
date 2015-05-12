/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// Simple 3D volume renderer

#ifndef _VOLUMERENDER_KERNEL_CU_
#define _VOLUMERENDER_KERNEL_CU_

#include <cuda_runtime_api.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include "defs.h"

typedef unsigned int  uint;
typedef unsigned char uchar;

cudaArray *d_volumeArray1 = 0;

cudaArray *d_transferFuncArray1;

typedef unsigned char VolumeType;

texture<VolumeType, 3, cudaReadModeNormalizedFloat> tex1;         // 3D texture

texture<float4, 1, cudaReadModeElementType>         transferTex1; // 1D transfer function texture

typedef struct {
    float4 m[3];
} float3x4;

__constant__ float3x4 c_invViewMatrix;  // inverse view matrix
__constant__ int dID;

struct Ray {
	float3 o;	// origin
	float3 d;	// direction
};

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm

__device__
int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
{
    // compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.0f) / r.d;
    float3 tbot = invR * (boxmin - r.o);
    float3 ttop = invR * (boxmax - r.o);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

	*tnear = largest_tmin;
	*tfar = smallest_tmax;

	return smallest_tmax > largest_tmin;
}

// transform vector by matrix (no translation)
__device__
float3 mul(const float3x4 &M, const float3 &v)
{
    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}

// transform vector by matrix with translation
__device__
float4 mul(const float3x4 &M, const float4 &v)
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = 1.0f;
    return r;
}

__device__ uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w*255)<<24) | (uint(rgba.z*255)<<16) | (uint(rgba.y*255)<<8) | uint(rgba.x*255);
}


__device__ float4
get_pix_val( int src, int maxSteps, Ray eyeRay, float tstep, float tnear, float tfar,
		     float Offset, float Scale, float dens, float weight, float opacity )
{
    float t = tnear;
    float3 pos = eyeRay.o + eyeRay.d*tnear;
    float3 step = eyeRay.d*tstep;
    float4 sum = make_float4(0.0f);

    for(int i=0; i<maxSteps; i++)
    {
        // read from 3D texture
        // remap position to [0, 1] coordinates
        float sample;
        float4 col;

        if (src == 1){
                sample = tex3D(tex1, pos.x*0.5f+0.5f, pos.y*0.5f+0.5f, pos.z*0.5f+0.5f);
                col = tex1D(transferTex1, (sample-Offset)*Scale);
        }

        //sample *= 64.0f;    // scale for 10-bit data

        // lookup in transfer function texture
        col.w *= dens * weight;

        // "under" operator for back-to-front blending
        //sum = lerp(sum, col, col.w);

        // pre-multiply alpha
        col.x *= col.w;
        col.y *= col.w;
        col.z *= col.w;
        // "over" operator for front-to-back blending
        sum = sum + col*(1.0f - sum.w);

        // exit early if opaque
        //if (sum.w > opacity)
        //    break;

        t += tstep;
        if (t > tfar) break;

        pos += step;
    }

    return sum;
}



__global__ void
d_render(uint *d_output, uint imageW, uint imageH,
         float dens1, float bright1,
         float Offset1, float Scale1, float weight1)
{
    const int maxSteps = 500;
    const float tstep = 0.01f;
    const float opacityThreshold = 0.95f;
    const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
    const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);

	uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;
    if ((x >= imageW) || (y >= imageH)) return;

    float u = (x / (float) imageW)*2.0f-1.0f;
    float v = (y / (float) imageH)*2.0f-1.0f;

    // calculate eye ray in world space
    Ray eyeRay;
    eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
    eyeRay.d = normalize(make_float3(u, v, -2.0f));
    eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

    // find intersection with box
	float tnear, tfar;
	int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);
    if (!hit) return;
	if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

    // march along ray from front to back, accumulating color
    float4 sum1 = make_float4(0.0f);

    sum1 += get_pix_val( 1, maxSteps, eyeRay, tstep, tnear, tfar, Offset1, Scale1, dens1, weight1, opacityThreshold );

    sum1 *= bright1;

    // write output color
    d_output[y*imageW + x] = rgbaFloatToInt(sum1);
}



__constant__ float dmax, dmin;
__global__ void deviceDub2Char( float *input, unsigned char *out ){

    int pos = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*gridDim.x*blockDim.x;

    float value = 255 * (input[pos] - dmin) / abs(dmax - dmin);

    out[pos] = (unsigned char) value;
}




extern "C" void cudaInitVdens( FLOAT_GRID *dens1,
                                CHAR_GRID *vdens1,
                                float data_max1 )
{
    int Xc = vdens1->count.x;
    int Yc = vdens1->count.y;
    int Zc = vdens1->count.z;
    size_t float_size = Xc*Yc*Zc*sizeof(float);
/*
    size_t freeMem, totalMem;
    checkCudaErrors(cudaMemGetInfo(&freeMem,&totalMem));
    printf("\n Free Memory: %lu / %lu (float_size = %lu)\n",freeMem,totalMem,float_size);
    fflush(stdout);
*/
    float *ddata;
    checkCudaErrors( cudaMalloc( (void **) &ddata, float_size) );
    checkCudaErrors( cudaMemset( ddata, 0.0, float_size ) );

    dim3 block(Xc);
    dim3 grid(Yc,Zc);
    size_t data_size = Xc*Yc*Zc*sizeof(unsigned char);
    //int max_size = max( max( Xc, Yc) , Zc );
/*
    checkCudaErrors(cudaMemGetInfo(&freeMem,&totalMem));
    printf("\n Free Memory: %lu / %lu (data_size = %lu)\n",freeMem,totalMem,data_size);
    fflush(stdout);
*/
    unsigned char *d_charvol;
    checkCudaErrors( cudaMalloc( (void**) &d_charvol, data_size ) );
    checkCudaErrors( cudaMemset( d_charvol, 0.0, data_size ) );

    checkCudaErrors( cudaMemcpy( ddata, dens1->matrix, float_size, cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaHostAlloc( (void**) &vdens1->matrix, data_size, cudaHostAllocPortable ) );
    checkCudaErrors( cudaMemcpyToSymbol( dmax, &data_max1, sizeof(float) ) );
    checkCudaErrors( cudaMemcpyToSymbol( dmin, &dens1->min, sizeof(float) ) );

    deviceDub2Char<<<grid,block>>>(ddata,d_charvol);
    cudaThreadSynchronize();

    checkCudaErrors( cudaMemcpy( vdens1->matrix, d_charvol, data_size, cudaMemcpyDeviceToHost ) );

    checkCudaErrors(cudaFree(d_charvol));
    checkCudaErrors(cudaFree(ddata));
/*
    checkCudaErrors(cudaMemGetInfo(&freeMem,&totalMem));
    printf("\n Free Memory: %lu / %lu \n",freeMem,totalMem);
    fflush(stdout);
*/
}


cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VolumeType>();
cudaMemcpy3DParms copyParams1 = {0};

extern "C"
void initCudaDens( void *h_volume1, cudaExtent volumeSize, int colorScale )
{
    // create 3D array
    checkCudaErrors( cudaMalloc3DArray(&d_volumeArray1, &channelDesc, volumeSize) );

    // copy data to 3D array
    copyParams1.srcPtr   = make_cudaPitchedPtr(h_volume1, volumeSize.width*sizeof(VolumeType), volumeSize.width, volumeSize.height);
    copyParams1.dstArray = d_volumeArray1;
    copyParams1.extent   = volumeSize;
    copyParams1.kind     = cudaMemcpyHostToDevice;
    checkCudaErrors( cudaMemcpy3D(&copyParams1) );

    // set texture parameters
    tex1.normalized = true;                      // access with normalized texture coordinates
    tex1.filterMode = cudaFilterModeLinear;      // linear interpolation
    tex1.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
    tex1.addressMode[1] = cudaAddressModeClamp;

    // bind array to 3D texture
    checkCudaErrors(cudaBindTextureToArray(tex1, d_volumeArray1, channelDesc));

    // create transfer function texture

    float4 transferFunc3[] = {
        {  0.5, 0.0, 0.5, 1.0, },
        {  1.0, 0.0, 1.0, 1.0, },
        {  0.5, 0.0, 1.0, 1.0, },
        {  0.0, 0.0, 1.0, 1.0, },
        {  0.0, 0.5, 1.0, 1.0, },
        {  0.0, 1.0, 1.0, 1.0, },
        {  0.0, 1.0, 0.5, 1.0, },
        {  0.0, 0.0, 0.0, 0.0  },
        {  0.5, 1.0, 0.0, 1.0, },
        {  1.0, 1.0, 0.0, 1.0, },
        {  1.0, 0.825, 0.0, 1.0, },
        {  1.0, 0.65, 0.0, 1.0, },
        {  1.0, 0.325, 0.0, 1.0, },
        {  1.0, 0.0, 0.0, 1.0  },
        {  1.0, 0.5, 0.5, 1.0  },
    };

    float4 transferFunc2[] = {
        {  0.0, 0.0, 0.0, 0.0  },
        {  0.5, 0.0, 0.5, 1.0, },
        {  1.0, 0.0, 1.0, 1.0, },
        {  0.5, 0.0, 1.0, 1.0, },
        {  0.0, 0.0, 1.0, 1.0, },
        {  0.0, 0.5, 1.0, 1.0, },
        {  0.0, 1.0, 1.0, 1.0, },
        {  0.0, 1.0, 0.5, 1.0, },
        {  0.0, 1.0, 0.0, 1.0, },
        {  0.0, 1.0, 0.0, 1.0, },
        {  0.5, 1.0, 0.0, 1.0, },
        {  1.0, 1.0, 0.0, 1.0, },
        {  1.0, 0.825, 0.0, 1.0, },
        {  1.0, 0.65, 0.0, 1.0, },
        {  1.0, 0.325, 0.0, 1.0, },
        {  1.0, 0.0, 0.0, 1.0  },
        {  1.0, 0.5, 0.5, 1.0  },
        {  1.0, 1.0, 1.0, 1.0  }
    };

    float4 transferFunc1[] = {
        {  0.0, 0.0, 0.0, 0.0  },
        {  0.2, 0.2, 0.2, 1.0, },
        {  0.4, 0.4, 0.4, 1.0, },
        {  0.6, 0.6, 0.6, 1.0, },
        {  0.8, 0.8, 0.8, 1.0, },
        {  1.0, 1.0, 1.0, 1.0, }
    };

    cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float4>();
    cudaArray* d_transferFuncArray1;

    if (colorScale <= 1)
    {
        checkCudaErrors(cudaMallocArray( &d_transferFuncArray1, &channelDesc2, sizeof(transferFunc1)/sizeof(float4), 1));
        checkCudaErrors(cudaMemcpyToArray( d_transferFuncArray1, 0, 0, transferFunc1, sizeof(transferFunc1), cudaMemcpyHostToDevice));
    }
    else if (colorScale == 2)
    {
        checkCudaErrors(cudaMallocArray( &d_transferFuncArray1, &channelDesc2, sizeof(transferFunc2)/sizeof(float4), 1));
        checkCudaErrors(cudaMemcpyToArray( d_transferFuncArray1, 0, 0, transferFunc2, sizeof(transferFunc2), cudaMemcpyHostToDevice));
    }
    else if (colorScale == 3)
    {
        checkCudaErrors(cudaMallocArray( &d_transferFuncArray1, &channelDesc2, sizeof(transferFunc3)/sizeof(float4), 1));
        checkCudaErrors(cudaMemcpyToArray( d_transferFuncArray1, 0, 0, transferFunc3, sizeof(transferFunc3), cudaMemcpyHostToDevice));
    }


    transferTex1.filterMode = cudaFilterModeLinear;
    transferTex1.normalized = true;    // access with normalized texture coordinates
    transferTex1.addressMode[0] = cudaAddressModeClamp;   // wrap texture coordinates

    // Bind the array to the texture
    checkCudaErrors( cudaBindTextureToArray( transferTex1, d_transferFuncArray1, channelDesc2));
}


extern "C"
void freeCudaBuffers()
{
    checkCudaErrors(cudaUnbindTexture(tex1));

    checkCudaErrors(cudaUnbindTexture(transferTex1));

    cudaFreeArray(d_transferFuncArray1);
}


extern "C"
void render_kernel(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH,
                    float dens1, float bright1, float Offset1, float Scale1, float weight1)
{

	d_render<<<gridSize, blockSize>>>( d_output, imageW, imageH, dens1, bright1,
										Offset1, Scale1, weight1 );
}

extern "C"
void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix)
{
    checkCudaErrors( cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix, 0, cudaMemcpyHostToDevice) );
}


#endif // #ifndef _VOLUMERENDER_KERNEL_CU_
