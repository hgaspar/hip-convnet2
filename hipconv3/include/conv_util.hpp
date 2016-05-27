#include "hip_runtime.h"
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

#ifndef CONV_UTIL_CUH
#define	CONV_UTIL_CUH

#include "../../hipmatrix/include/hipmatrix.hpp"
#ifndef MIN
#define MIN(a, b) ((a) > (b) ? (b) : (a))
#endif
#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

void convLocalMaxUndo(HIPmatrix& images, HIPmatrix& maxGrads, HIPmatrix& maxActs, HIPmatrix& target,
                      int subsX, int startX, int strideX, int outputsX);
void convLocalAvgUndo(HIPmatrix& avgGrads, HIPmatrix& target,
                      int subsX, int startX, int strideX, int outputsX, int imgSize, bool sum);

void convLocalAvgUndo(HIPmatrix& avgGrads, HIPmatrix& target,
                      int subsX, int startX, int strideX, int outputsX, int imgSize, bool sum,
                      float scaleTargets, float scaleOutput);
void convLocalMaxUndo(HIPmatrix& images, HIPmatrix& maxGrads, HIPmatrix& maxActs, HIPmatrix& target,
                      int subsX, int startX, int strideX, int outputsX, float scaleTargets, float scaleOutput);

void convResponseNorm(HIPmatrix& images, HIPmatrix& denoms, HIPmatrix& target, int numFilters, int sizeX, float addScale, float powScale, float minDiv);
void convResponseNormUndo(HIPmatrix& outGrads, HIPmatrix& denoms, HIPmatrix& inputs, HIPmatrix& acts, HIPmatrix& target, int numFilters,
                         int sizeX, float addScale, float powScale, float scaleTargets, float scaleOutput);
void convContrastNorm(HIPmatrix& images, HIPmatrix& meanDiffs, HIPmatrix& denoms, HIPmatrix& target, int numFilters, int sizeX, float addScale, float powScale, float minDiv);
void convContrastNormUndo(HIPmatrix& outGrads, HIPmatrix& denoms, HIPmatrix& meanDiffs, HIPmatrix& acts, HIPmatrix& target, int numFilters,
                         int sizeX, float addScale, float powScale, float scaleTargets, float scaleOutput);

void convGaussianBlur(HIPmatrix& images, HIPmatrix& filter, HIPmatrix& target, bool horiz, int numChannels,
                      float scaleTargets, float scaleOutputs);
void convBedOfNails(HIPmatrix& images, HIPmatrix& target, int numChannels, int imgSize, int startX,
                    int strideX, float scaleTargets, float scaleOutput);
void convBedOfNailsUndo(HIPmatrix& actsGrad, HIPmatrix& target, int numChannels, int imgSize,
                        int startX, int strideX, float scaleTargets, float scaleOutput);

void convResizeBilinear(HIPmatrix& images, HIPmatrix& target, int imgSize, int tgtSize, float scale);
void convRGBToYUV(HIPmatrix& images, HIPmatrix& target);
void convRGBToLAB(HIPmatrix& images, HIPmatrix& target, bool center);
void convCrop(HIPmatrix& imgs, HIPmatrix& target, int imgSize, int tgtSize, int startY, int startX);
void normalizeLocalWeights(HIPmatrix& weights, int numModules, float norm);
void convContrastNormCrossMap(HIPmatrix& images, HIPmatrix& meanDiffs, HIPmatrix& target,
                             int numFilters, int sizeF, float addScale, float powScale, float minDiv, bool blocked);
void convResponseNormCrossMapUndo(HIPmatrix& outGrads, HIPmatrix& inputs, HIPmatrix& acts, HIPmatrix& target, int numFilters,
                         int sizeF, float addScale, float powScale, float minDiv, bool blocked, float scaleTargets, float scaleOutput);
void convResponseNormCrossMap(HIPmatrix& images, HIPmatrix& target, int numFilters, int sizeF, float addScale,
                              float powScale, bool blocked);
void convResponseNormCrossMap(HIPmatrix& images, HIPmatrix& target, int numFilters, int sizeF, float addScale,
                              float powScale, float minDiv, bool blocked);
void convReflectHorizontal(HIPmatrix& images, HIPmatrix& targets, int imgSize);

void convCrossMapMaxPoolUndo(HIPmatrix& images, HIPmatrix& maxGrads, HIPmatrix& maxActs, HIPmatrix& target,
                             const int imgSize, const int startF, const int poolSize,
                             const int stride, const float scaleTargets, const float scaleOutputs);
template<bool sum>
class AvgPooler {
public:
    __device__ inline float operator()(const float a, const float b) const {
        return a + b;
    }
    __device__ inline float getBaseValue() const {
        return 0;
    }
    __device__ inline float output(const float a, const int regionSize) const {
        return sum ? a : (a / regionSize);
    }
};

class MaxPooler {
public:
    __device__ inline float operator()(const float a, const float b) const {
        return fmaxf(a, b);
    }
    __device__ inline float getBaseValue() const {
        return -2e38;
    }
    __device__ inline float output(const float a, const int regionSize) const {
        return a;
    }
};

class MaxAbsPooler {
public:
    __device__ inline float operator()(const float a, const float b) const {
        return fabsf(a) > fabsf(b) ? a : b;
    }
    __device__ inline float getBaseValue() const {
        return 0.0f;
    }
    __device__ inline float output(const float a, const int regionSize) const {
        return a;
    }
};

/*
 * Block size B_YxB_X
 * hipBlockIdx_x determines output.x, image idx in batches of B_X*imgsPerThread
 * hipBlockIdx_y determines output.y, filter idx in batches of B_Y*filtersPerThread
 *
 * So each block does one output for some number of images/filters.
 *
 * hipThreadIdx_x determines img idx
 * hipThreadIdx_y determines filter idx
 *
 * imgs:        (numFilters, imgPixels, numImages)
 * target:      (numFilters, numOutputs, numImages)
 *
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 */

template<class Agg, int B_Y, int B_X, int imgsPerThread, int filtersPerThread, bool checkCaseBounds>
__global__ void kLocalPool(float* imgs, float* target, const int imgSize, const int numFilters,
                           const int numImages, const int subsX, const int startX, const int strideX,
                           const int outputsX, Agg agg) {
    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int numFilterBlocks = DIVUP(numFilters, B_Y*filtersPerThread);
    const int outputIdxX = hipBlockIdx_x / numImgBlocks;
    const int outputIdxY = hipBlockIdx_y / numFilterBlocks;
    const int blockImgIdx = (hipBlockIdx_x % numImgBlocks) * B_X * imgsPerThread;
    const int blockFilterIdx = (hipBlockIdx_y % numFilterBlocks) * B_Y * filtersPerThread;
    const int myFilterIdx = (blockFilterIdx + hipThreadIdx_y*filtersPerThread);
    if (myFilterIdx >= numFilters) {
        return;
    }

    const int outputIdx = outputIdxY * outputsX + outputIdxX;
    const int numOutputs = outputsX * outputsX;
    const int imgPixels = imgSize * imgSize;

    const int startImgPxX = startX + outputIdxX * strideX;
    const int startImgPxY = startX + outputIdxY * strideX;
    const int imgIdx = blockImgIdx + hipThreadIdx_x;

    imgs += myFilterIdx * imgPixels * numImages + imgIdx;
    target += (myFilterIdx * numOutputs + outputIdx) * numImages + imgIdx;

    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[f][i] = agg.getBaseValue();
        }
    }

    const int loopStartY = MAX(0, startImgPxY);
    const int loopStartX = MAX(0, startImgPxX);
    const int loopEndY = MIN(imgSize, startImgPxY + subsX);
    const int loopEndX = MIN(imgSize, startImgPxX + subsX);
    const int regionSize = (loopEndY - loopStartY) * (loopEndX - loopStartX);
    for (int y = loopStartY; y < loopEndY; y++) {
        for (int x = loopStartX; x < loopEndX; x++) {
            const int imgPx = y * imgSize + x;
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        prod[f][i] = agg(prod[f][i], imgs[(f * imgPixels + imgPx) * numImages + i * B_X]);
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                target[f * numOutputs * numImages + i * B_X] = agg.output(prod[f][i], regionSize);
            }
        }
    }
}


/*
 * Block size B_YxB_X
 * hipBlockIdx_x determines pixel.x, image idx in batches of B_X*imgsPerThread
 * hipBlockIdx_y determines pixel.y, output idx in batches of B_Y
 *
 * So each block does one pixel for some number of images/filters.
 *
 * hipThreadIdx_x determines img idx
 * hipThreadIdx_y determines output idx
 *
 * imgs:        (numFilters, imgPixels, numImages)
 * target:      (numOutputs, imgPixels, numImages) (out)
 *
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 */
template<class Agg, int B_Y, int B_X, int imgsPerThread, bool checkCaseBounds>
__global__ void kPoolCrossMap(float* imgs, float* target, const int imgSize,
                              const int numFilters, const int numImages, const int startF, const int poolSize,
                              const int numOutputs, const int stride, Agg agg) {
    const int imgPixels = imgSize * imgSize;
    const int numImgBlocks = DIVUP(numImages, B_X*imgsPerThread);
//    const int numOutputs = DIVUP(numFilters, stride);
    const int numOutputBlocks = DIVUP(numOutputs,B_Y);
    const int pxIdxX = hipBlockIdx_x / numImgBlocks;
    const int pxIdxY = hipBlockIdx_y / numOutputBlocks;
    const int blockImgIdx = (hipBlockIdx_x % numImgBlocks) * B_X * imgsPerThread;
    const int outputIdx = (hipBlockIdx_y % numOutputBlocks) * B_Y + hipThreadIdx_y;
//    const int filterIdx = outputIdx * stride;

    const int pxIdx = pxIdxY * imgSize + pxIdxX;
    const int imgIdx = blockImgIdx + hipThreadIdx_x;

    if (outputIdx < numOutputs) {
        imgs += (pxIdx) * numImages + imgIdx;
        target += (outputIdx * imgPixels + pxIdx) * numImages + imgIdx;

        float prod[imgsPerThread];
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                prod[i] = agg.getBaseValue();
            }
        }

        const int myStartF = startF + outputIdx * stride;
        const int loopStartF = max(0, myStartF);
        const int loopEndF = min(numFilters, myStartF + poolSize);

        for (int f = loopStartF; f < loopEndF; ++f) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                    prod[i] = agg(prod[i], imgs[f * imgPixels * numImages + i * B_X]);
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                target[i * B_X] = agg.output(prod[i], poolSize);
            }
        }
    }
}

/*
 * imgs:        (numFilters, imgPixels, numImages)
 * target:      (numOutputs, imgPixels, numImages)
 */
template<class Pooler>
void convPoolCrossMap(HIPmatrix& images, HIPmatrix& target, const int startF, const int poolSize,
                      const int numOutputs, const int stride, const int imgSize, Pooler pooler) {
    int numImages = images.getNumCols();
    int imgPixels = imgSize * imgSize;
    int numFilters = images.getNumRows() / imgPixels;
    assert(images.getNumRows() == numFilters * imgPixels);

    assert(!images.isTrans());
    assert(!target.isTrans());
    assert(images.isContiguous());
//    assert(numFilters % 4 == 0);
//    assert(numImages % 128 == 0);
    assert(stride <= poolSize);
    assert(startF <= 0);
    assert(startF + (numOutputs-1) * stride + poolSize >= numFilters); // All filters must be covered

    hipStream_t stream = HIPmatrix::getDefaultStream();
    target.resize(imgPixels*numOutputs, numImages);
    int imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;

    dim3 threads(32, 4);
    dim3 blocks(imgSize * DIVUP(numImages, threads.x * imgsPerThread), imgSize * DIVUP(numOutputs, threads.y));
    bool checkCaseBounds = numImages % (threads.x*imgsPerThread) != 0;
    if (!checkCaseBounds) {
        if (imgsPerThread == 4) {
            hipFuncSetCacheConfig(kPoolCrossMap<Pooler, 4, 32, 4, false>, hipFuncCachePreferShared);
            hipLaunchKernel(HIP_KERNEL_NAME(kPoolCrossMap<Pooler, 4, 32, 4, false>), dim3(blocks), dim3(threads), 0, stream, images.getDevData(), target.getDevData(),
                                                              imgSize, numFilters, numImages, startF, poolSize, numOutputs, stride, pooler);

        } else if (imgsPerThread == 2) {
            hipFuncSetCacheConfig(kPoolCrossMap<Pooler, 4, 32, 2, false>, hipFuncCachePreferShared);
            hipLaunchKernel(HIP_KERNEL_NAME(kPoolCrossMap<Pooler, 4, 32, 2, false>), dim3(blocks), dim3(threads), 0, stream, images.getDevData(), target.getDevData(),
                                                              imgSize, numFilters, numImages, startF, poolSize, numOutputs, stride, pooler);

        } else if (imgsPerThread == 1) {
            hipFuncSetCacheConfig(kPoolCrossMap<Pooler, 4, 32, 1, false>, hipFuncCachePreferShared);
            hipLaunchKernel(HIP_KERNEL_NAME(kPoolCrossMap<Pooler, 4, 32, 1, false>), dim3(blocks), dim3(threads), 0, stream, images.getDevData(), target.getDevData(),
                                                              imgSize, numFilters, numImages, startF, poolSize, numOutputs, stride, pooler);
        }
    } else {
        if (imgsPerThread == 1) {
            hipFuncSetCacheConfig(kPoolCrossMap<Pooler, 4, 32, 1, true>, hipFuncCachePreferShared);
            hipLaunchKernel(HIP_KERNEL_NAME(kPoolCrossMap<Pooler, 4, 32, 1, true>), dim3(blocks), dim3(threads), 0, stream, images.getDevData(), target.getDevData(),
                                                              imgSize, numFilters, numImages, startF, poolSize, numOutputs, stride, pooler);
        } else {
            assert(false);
        }
    }
    getLastCudaError("convPoolCrossMap: kernel execution failed");
}

/*
 * Block size 16xB_X
 * hipBlockIdx_x determines 4x4 pixel.x region, image idx in batches of B_X*imgsPerThread
 * hipBlockIdx_y determines 4x4 pixel.y region, filter idx in batches of filtersPerThread
 *
 * So each block does a 4x4 region for some number of images/filters.
 *
 * hipThreadIdx_x determines img idx
 * hipThreadIdx_y determines pixel idx
 *
 * imgs:        (numFilters, imgPixels, numImages)
 * target:      (numFilters, numOutputs, numImages)
 *
 * B_X one of 8, 16, 32
 * imgsPerThread one of 1, 2, 4, 8, 16
 *
 * B_XximgsPerThread MUST be divisible by 32.
 * Number of filters MUST be divisible by filtersPerThread.
 *
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 *
 * Final write-out will not be fully coalesced unless B_X is 32. But there's a lot more
 * reading than writing here, and the reading is all coalesced, so it should be OK.
 *
 * To be used when the stride is 1 and the pooling region is fairly large.
 */
template<class Agg, int B_X, int imgsPerThread, int filtersPerThread, bool checkCaseBounds>
__global__ void kLocalPool2(float* imgs, float* target, const int imgSize, const int numFilters,
                           const int numImages, const int subsX, const int startX,
                           const int outputsX, Agg agg) {
    __shared__ float shImgs[filtersPerThread][B_X*imgsPerThread];
    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int numFilterBlocks = numFilters/(filtersPerThread);
    const int blockOutputX = 4*(hipBlockIdx_x / numImgBlocks);
    const int blockOutputY = 4*(hipBlockIdx_y / numFilterBlocks);
    const int blockImgIdx = (hipBlockIdx_x % numImgBlocks) * B_X * imgsPerThread;
    const int blockFilterIdx = (hipBlockIdx_y % numFilterBlocks) * filtersPerThread;

//    const int blockOutputIdx = blockOutputY * outputsX + blockOutputX;
    const int numOutputs = outputsX * outputsX;
    const int imgPixels = imgSize * imgSize;

    const int tidx = hipThreadIdx_y * B_X + hipThreadIdx_x;
    const int loadY = tidx / 32, loadX = tidx % 32;

    const int myX = hipThreadIdx_y % 4;
    const int myY = hipThreadIdx_y / 4;

    const int myOutputIdxY = blockOutputY + myY;
    const int myOutputIdxX = blockOutputX + myX;
    const int myOutputIdx = myOutputIdxY * outputsX + myOutputIdxX;

    const int startImgPxX = startX + blockOutputX;
    const int startImgPxY = startX + blockOutputY;
    const int endImgPxX = startImgPxX + subsX;
    const int endImgPxY = startImgPxY + subsX;

    const int myStartImgPxY = startImgPxY + myY;
    const int myStartImgPxX = startImgPxX + myX;
    const int myEndImgPxY = endImgPxY + myY;
    const int myEndImgPxX = endImgPxX + myX;

    const int loopStartY = MAX(startImgPxY, 0);
    const int loopStartX = MAX(startImgPxX, 0);
    const int loopEndY = MIN(imgSize, endImgPxY + 3);
    const int loopEndX = MIN(imgSize, endImgPxX + 3);

    const int imgIdx = blockImgIdx + hipThreadIdx_x;

    imgs += (blockFilterIdx + loadY) * imgPixels * numImages + blockImgIdx + loadX;
    target += (blockFilterIdx * numOutputs + myOutputIdx) * numImages + imgIdx;

    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[f][i] = agg.getBaseValue();
        }
    }
    int regionSize = 0;
    for (int y = loopStartY; y < loopEndY; y++) {
        const bool isInY = y >= myStartImgPxY && y < myEndImgPxY ;
        for (int x = loopStartX; x < loopEndX; x++) {
            // Load a pixel
            const int px = y * imgSize + x;
            #pragma unroll
            for (int ly = 0; ly < filtersPerThread; ly += B_X/2) {
                if (filtersPerThread % (B_X/2) == 0 || ly + loadY < filtersPerThread) {
                    #pragma unroll
                    for (int lx = 0; lx < B_X*imgsPerThread; lx += 32) {
                        if (!checkCaseBounds || lx + loadX + blockImgIdx < numImages) {
                            shImgs[ly + loadY][lx + loadX] = imgs[(ly * imgPixels + px) * numImages + lx];
                        }
                    }
                }
            }
            __syncthreads();

            // Is this pixel in my region?
            if (isInY && x >= myStartImgPxX && x < myEndImgPxX) {
                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                        #pragma unroll
                        for (int f = 0; f < filtersPerThread; f++) {
                            prod[f][i] = agg(prod[f][i], shImgs[f][hipThreadIdx_x + i * B_X]);
                        }
                    }
                }
                ++regionSize;
            }
            __syncthreads();

        }
    }
    if (myOutputIdxY < outputsX && myOutputIdxX < outputsX) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    target[f * numOutputs * numImages + i * B_X] = agg.output(prod[f][i], regionSize);
                }
            }
        }
    }
}

/*
 * imgs:        (numFilters, imgPixels, numImages)
 * target:      (numFilters, outputs, numImages)
 */
template<class Pooler>
void convLocalPool(HIPmatrix& images, HIPmatrix& target, int numFilters,
                   int subsX, int startX, int strideX, int outputsX, Pooler pooler) {
    int numImages = images.getNumCols();
    int imgPixels = images.getNumRows() / numFilters;
    assert(images.getNumRows() == numFilters * imgPixels);
    int imgSize = int(sqrt(imgPixels));
    assert(imgSize * imgSize == imgPixels);

    assert(!images.isTrans());
    assert(!target.isTrans());
    assert(images.isContiguous());
//    assert(numFilters % 4 == 0);
//    assert(numImages % 128 == 0);
    hipStream_t stream = HIPmatrix::getDefaultStream();
    int outputs = outputsX * outputsX;
    target.resize(numFilters*outputs, numImages);

    if (strideX == 1 && subsX >= 6 && outputsX > 1) {
        // NOTE: this part has not been optimized for GPU
        int imgsPerThread = numImages % 128 == 0 ? 8 : 4;
        int filtersPerThread = numFilters % 4 == 0 ? 4 : numFilters % 3 == 0 ? 3 : numFilters % 2 == 0 ? 2 : 1;
        int bx = 8;
        bool checkCaseBounds = numImages % (bx*imgsPerThread) != 0;
        assert((imgsPerThread * bx) % 32 == 0);
        assert(numFilters % filtersPerThread == 0);
        dim3 threads(bx, 16);
        dim3 blocks(DIVUP(outputsX, 4) * DIVUP(numImages, bx*imgsPerThread), DIVUP(outputsX, 4) * numFilters / filtersPerThread);
//        printf("threads: %dx%d, blocks: %dx%d, imgSize: %d, numFilters: %d, numImages: %d, subsX: %d, startX: %d, outputsX: %d\n",
//                threads.y, threads.x, blocks.y, blocks.x, imgSize, numFilters, numImages, subsX, startX, outputsX);
        if (imgsPerThread == 8) {
            if (filtersPerThread == 1) {
                 if (checkCaseBounds) {
                    hipFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 1, true>, hipFuncCachePreferShared);
                    hipLaunchKernel(HIP_KERNEL_NAME(kLocalPool2<Pooler, 8, 8, 1, true>), dim3(blocks), dim3(threads), 0, stream, images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                } else {
                    hipFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 1, false>, hipFuncCachePreferShared);
                    hipLaunchKernel(HIP_KERNEL_NAME(kLocalPool2<Pooler, 8, 8, 1, false>), dim3(blocks), dim3(threads), 0, stream, images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                }
            } else if (filtersPerThread == 2) {
                if (checkCaseBounds) {
                    hipFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 2, true>, hipFuncCachePreferShared);
                    hipLaunchKernel(HIP_KERNEL_NAME(kLocalPool2<Pooler, 8, 8, 2, true>), dim3(blocks), dim3(threads), 0, stream, images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                } else {
                    hipFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 2, false>, hipFuncCachePreferShared);
                    hipLaunchKernel(HIP_KERNEL_NAME(kLocalPool2<Pooler, 8, 8, 2, false>), dim3(blocks), dim3(threads), 0, stream, images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                }
            } else if (filtersPerThread == 3) {
                if (checkCaseBounds) {
                    hipFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 3, true>, hipFuncCachePreferShared);
                    hipLaunchKernel(HIP_KERNEL_NAME(kLocalPool2<Pooler, 8, 8, 3, true>), dim3(blocks), dim3(threads), 0, stream, images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                } else {
                    hipFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 3, false>, hipFuncCachePreferShared);
                    hipLaunchKernel(HIP_KERNEL_NAME(kLocalPool2<Pooler, 8, 8, 3, false>), dim3(blocks), dim3(threads), 0, stream, images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                }
            } else if (filtersPerThread == 4) {
                if (checkCaseBounds) {
                    hipFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 4, true>, hipFuncCachePreferShared);
                    hipLaunchKernel(HIP_KERNEL_NAME(kLocalPool2<Pooler, 8, 8, 4, true>), dim3(blocks), dim3(threads), 0, stream, images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                } else {
                    hipFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 4, false>, hipFuncCachePreferShared);
                    hipLaunchKernel(HIP_KERNEL_NAME(kLocalPool2<Pooler, 8, 8, 4, false>), dim3(blocks), dim3(threads), 0, stream, images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                }
            }
        } else if (imgsPerThread == 4) {
            if (filtersPerThread == 1) {
                 if (checkCaseBounds) {
                    hipFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 1, true>, hipFuncCachePreferShared);
                    hipLaunchKernel(HIP_KERNEL_NAME(kLocalPool2<Pooler, 8, 4, 1, true>), dim3(blocks), dim3(threads), 0, stream, images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                } else {
                    hipFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 1, false>, hipFuncCachePreferShared);
                    hipLaunchKernel(HIP_KERNEL_NAME(kLocalPool2<Pooler, 8, 4, 1, false>), dim3(blocks), dim3(threads), 0, stream, images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                }
            } else if (filtersPerThread == 2) {
                if (checkCaseBounds) {
                    hipFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 2, true>, hipFuncCachePreferShared);
                    hipLaunchKernel(HIP_KERNEL_NAME(kLocalPool2<Pooler, 8, 4, 2, true>), dim3(blocks), dim3(threads), 0, stream, images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                } else {
                    hipFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 2, false>, hipFuncCachePreferShared);
                    hipLaunchKernel(HIP_KERNEL_NAME(kLocalPool2<Pooler, 8, 4, 2, false>), dim3(blocks), dim3(threads), 0, stream, images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                }
            } else if (filtersPerThread == 3) {
                if (checkCaseBounds) {
                    hipFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 3, true>, hipFuncCachePreferShared);
                    hipLaunchKernel(HIP_KERNEL_NAME(kLocalPool2<Pooler, 8, 4, 3, true>), dim3(blocks), dim3(threads), 0, stream, images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                } else {
                    hipFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 3, false>, hipFuncCachePreferShared);
                    hipLaunchKernel(HIP_KERNEL_NAME(kLocalPool2<Pooler, 8, 4, 3, false>), dim3(blocks), dim3(threads), 0, stream, images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                }
            } else if (filtersPerThread == 4) {
                if (checkCaseBounds) {
                    hipFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 4, true>, hipFuncCachePreferShared);
                    hipLaunchKernel(HIP_KERNEL_NAME(kLocalPool2<Pooler, 8, 4, 4, true>), dim3(blocks), dim3(threads), 0, stream, images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                } else {
                    hipFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 4, false>, hipFuncCachePreferShared);
                    hipLaunchKernel(HIP_KERNEL_NAME(kLocalPool2<Pooler, 8, 4, 4, false>), dim3(blocks), dim3(threads), 0, stream, images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                }
            }
        }
    } else {
        int filtersPerThread = numFilters % 16 == 0 ? 4 : 1;
        int imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
        bool checkCaseBounds = numImages % (32*imgsPerThread) != 0;
        dim3 threads(32, 4);
        dim3 blocks(DIVUP(numImages,32*imgsPerThread) * outputsX, DIVUP(numFilters, 4 * filtersPerThread) * outputsX);
        if (imgsPerThread == 4) {
            if (filtersPerThread == 1) {
                if (checkCaseBounds) {
                    hipFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 4, 1, true>, hipFuncCachePreferL1);
                    hipLaunchKernel(HIP_KERNEL_NAME(kLocalPool<Pooler, 4, 32, 4, 1, true>), dim3(blocks), dim3(threads), 0, stream, images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                } else {
                    hipFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 4, 1, false>, hipFuncCachePreferL1);
                    hipLaunchKernel(HIP_KERNEL_NAME(kLocalPool<Pooler, 4, 32, 4, 1, false>), dim3(blocks), dim3(threads), 0, stream, images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                }
            } else {
                if (checkCaseBounds) {
                    hipFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 4, 4, true>, hipFuncCachePreferL1);
                    hipLaunchKernel(HIP_KERNEL_NAME(kLocalPool<Pooler, 4, 32, 4, 4, true>), dim3(blocks), dim3(threads), 0, stream, images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                } else {
                    hipFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 4, 4, false>, hipFuncCachePreferL1);
                    hipLaunchKernel(HIP_KERNEL_NAME(kLocalPool<Pooler, 4, 32, 4, 4, false>), dim3(blocks), dim3(threads), 0, stream, images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                }
            }
        } else if (imgsPerThread == 2) {
            if (filtersPerThread == 1) {
                if (checkCaseBounds) {
                    hipFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 2, 1, true>, hipFuncCachePreferL1);
                    hipLaunchKernel(HIP_KERNEL_NAME(kLocalPool<Pooler, 4, 32, 2, 1, true>), dim3(blocks), dim3(threads), 0, stream, images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                } else {
                    hipFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 2, 1, false>, hipFuncCachePreferL1);
                    hipLaunchKernel(HIP_KERNEL_NAME(kLocalPool<Pooler, 4, 32, 2, 1, false>), dim3(blocks), dim3(threads), 0, stream, images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                }
            } else {
                if (checkCaseBounds) {
                    hipFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 2, 4, true>, hipFuncCachePreferL1);
                    hipLaunchKernel(HIP_KERNEL_NAME(kLocalPool<Pooler, 4, 32, 2, 4, true>), dim3(blocks), dim3(threads), 0, stream, images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                } else {
                    hipFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 2, 4, false>, hipFuncCachePreferL1);
                    hipLaunchKernel(HIP_KERNEL_NAME(kLocalPool<Pooler, 4, 32, 2, 4, false>), dim3(blocks), dim3(threads), 0, stream, images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                }
            }
        } else {
            if (filtersPerThread == 1) {
                if (checkCaseBounds) {
                    hipFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 1, 1, true>, hipFuncCachePreferL1);
                    hipLaunchKernel(HIP_KERNEL_NAME(kLocalPool<Pooler, 4, 32, 1, 1, true>), dim3(blocks), dim3(threads), 0, stream, images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                } else {
                    hipFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 1, 1, false>, hipFuncCachePreferL1);
                    hipLaunchKernel(HIP_KERNEL_NAME(kLocalPool<Pooler, 4, 32, 1, 1, false>), dim3(blocks), dim3(threads), 0, stream, images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                }
            } else {
                if (checkCaseBounds) {
                    hipFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 1, 4, true>, hipFuncCachePreferL1);
                    hipLaunchKernel(HIP_KERNEL_NAME(kLocalPool<Pooler, 4, 32, 1, 4, true>), dim3(blocks), dim3(threads), 0, stream, images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                } else {
                    hipFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 1, 4, false>, hipFuncCachePreferL1);
                    hipLaunchKernel(HIP_KERNEL_NAME(kLocalPool<Pooler, 4, 32, 1, 4, false>), dim3(blocks), dim3(threads), 0, stream, images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                }
            }
        }
    }
    getLastCudaError("convLocalPool: kernel execution failed");
}

#endif	/* CONV_UTIL_CUH */

