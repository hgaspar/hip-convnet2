/*
 * Copyright 2014 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <hip_runtime.h>
#include "../include/nvmatrix_kernels.cuh"

__global__ void kTile(const float* src, float* tgt, const uint srcWidth, const uint srcHeight, const uint tgtWidth, const uint tgtHeight) {
    const int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const int numThreads = hipBlockDim_x * hipGridDim_x;
    //    const unsigned int numEls = tgtWidth * tgtHeight;
    for (uint i = idx; i < tgtWidth * tgtHeight; i += numThreads) {
        const uint y = i / tgtWidth;
        const uint x = i % tgtWidth;
        const uint srcY = y % srcHeight;
        const uint srcX = x % srcWidth;
        tgt[i] = src[srcY * srcWidth + srcX];
    }
}

__global__ void kDotProduct_r(float* a, float* b, float* target,  const uint numElements) {
    __shared__ float shmem[DP_BLOCKSIZE];

    uint eidx = DP_BLOCKSIZE * hipBlockIdx_x + hipThreadIdx_x;
    shmem[hipThreadIdx_x] = 0;
    if (eidx < hipGridDim_x * DP_BLOCKSIZE) {
        for (; eidx < numElements; eidx += hipGridDim_x * DP_BLOCKSIZE) {
            shmem[hipThreadIdx_x] += a[eidx] * b[eidx];
        }
    }
    __syncthreads();
    if (hipThreadIdx_x < 256) {
        shmem[hipThreadIdx_x] += shmem[hipThreadIdx_x + 256];
    }
    __syncthreads();
    if (hipThreadIdx_x < 128) {
        shmem[hipThreadIdx_x] += shmem[hipThreadIdx_x + 128];
    }
    __syncthreads();
    if (hipThreadIdx_x < 64) {
        shmem[hipThreadIdx_x] += shmem[hipThreadIdx_x + 64];
    }
    __syncthreads();
    if (hipThreadIdx_x < 32) {
        volatile float* mysh = &shmem[hipThreadIdx_x];
        *mysh += mysh[32];
        *mysh += mysh[16];
        *mysh += mysh[8];
        *mysh += mysh[4];
        *mysh += mysh[2];
        *mysh += mysh[1];
        if (hipThreadIdx_x == 0) {
            target[hipBlockIdx_x] = *mysh;
        }
    }
}

__global__ void kSetupCurand(curandState *state, unsigned long long seed) {
    const uint tidx = NUM_RND_THREADS_PER_BLOCK * hipBlockIdx_x + hipThreadIdx_x;
    /* Each thread gets same seed, a different sequence number,
     no offset */
    curand_init(seed, tidx, 0, &state[tidx]);
}

