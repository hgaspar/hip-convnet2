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

#include <set>
#include <vector>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <typeinfo>
#include <map>
#include <cuda.h>
#include <signal.h>
#include "../include/hipmatrix.hpp"
#include "../include/hipmatrix_operators.hpp"

using namespace std;

/*
 * Device random number generator pointers.
 */
//map<int,curandGenerator_t> HIPmatrix::rndGen;
map<int,MemorySegment*> HIPmatrix::_rndDevStates;
map<int,int> HIPmatrix::_rndDevThreads;
pthread_mutex_t* HIPmatrix::_rndMutex = makeMutex();
pthread_mutex_t* HIPmatrix::_cublasMutex = makeMutex();
pthread_mutex_t* HIPmatrix::_streamMutex = makeMutex();
std::map<int,hipblasHandle_t> HIPmatrix::_cublasHandles;
std::map<int,hipStream_t> HIPmatrix::_defaultStreams;

pthread_mutex_t* HIPmatrix::makeMutex() {
    pthread_mutex_t* m = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t));
    pthread_mutex_init(m, NULL);
    return m;
}
/*
   Do not call resize in _init because resize is a virtual function
   which is overridden in base class. Since C++ is retarded and unable
   to call overridden functions from constructors, we shall call resize
   separately from every constructor after calling _init.
*/
void HIPmatrix::_init(bool isTrans) {
    _numRows = 0;
    _numCols = 0;
    _numElements = 0;
    _ownsData = true;

    _isTrans = isTrans;
    _memSegment = NULL;

    _stride = 0;
    _texObj = 0;
}

HIPmatrix::HIPmatrix() : _deleted(false) {
    _init(false);
}

HIPmatrix::HIPmatrix(bool isTrans) : _deleted(false) {
    _init(isTrans);
}

HIPmatrix::HIPmatrix(int numRows, int numCols, bool isTrans) : _deleted(false) {
    _init(isTrans);
    resize(numRows, numCols);
}

HIPmatrix::HIPmatrix(const Matrix& like, bool copy) : _deleted(false) {
    _init(like.isTrans());
    resize(like.getNumRows(), like.getNumCols());
    if (copy) {
        copyFromHost(like);
    }
}

HIPmatrix::HIPmatrix(const HIPmatrix& like, bool copy) : _deleted(false) {
    _init(like.isTrans());
    resize(like.getNumRows(), like.getNumCols());
    if (copy) {
        like.copy(*this);
    }
}

/*
 * Initializes HIPmatrix with same dimensions as given matrix but
 * does not copy any data.
 */
HIPmatrix::HIPmatrix(const HIPmatrix& like) : _deleted(false) {
    _init(like.isTrans());
    resize(like.getNumRows(), like.getNumCols());
}

/*
 * Initializes HIPmatrix with same dimensions as given matrix but
 * does not copy any data.
 */
HIPmatrix::HIPmatrix(const Matrix& like) : _deleted(false) {
    _init(false);
    resize(like.getNumRows(), like.getNumCols());
}

HIPmatrix::HIPmatrix(MemorySegment* mem, int numRows, int numCols, int stride, bool isTrans) :
    _numRows(numRows),
    _numCols(numCols),
    _numElements(numRows*numCols),
    _ownsData(false),
    _memSegment(mem),
    _isTrans(isTrans),
    _deleted(false),
    _texObj(0) {
    _stride = stride < 0 ? getLeadingDim() : stride;
}

HIPmatrix::~HIPmatrix() {
    if (!_deleted) {
        deallocTexture();
        if(_ownsData && _numElements > 0) {
            dealloc();
        } else {
            // dealloc deletes the mem segment. But if this is a view,
            // then we still need to delete the mem segment object.
//            assert(_memSegment == NULL || _memSegment->getSize() == 0);
            delete _memSegment;
        }
    }
}

void HIPmatrix::copyFromHost(const Matrix& hostMatrix) {
    copyFromHost(hostMatrix, false, getDefaultStream());
}

void HIPmatrix::copyFromHost(const Matrix& hostMatrix, bool resizeTarget) {
    copyFromHost(hostMatrix, resizeTarget, getDefaultStream());
}

void HIPmatrix::copyFromHost(const Matrix& hostMatrix, bool resizeTarget, hipStream_t stream) {
    if (resizeTarget) {
        resize(hostMatrix);
    } else {
        assert(isSameDims(hostMatrix));
    }
    setTrans(hostMatrix.isTrans());

    if (getNumElements() > 0) {
        CUBLAS_CALL(cublasSetMatrixAsync(hostMatrix.getLeadingDim(), hostMatrix.getFollowingDim(), sizeof(float),
                                    hostMatrix.getData(), hostMatrix.getLeadingDim(), getDevData(), _stride, stream));
        syncStream(stream);
    }
}

void HIPmatrix::copyToHost(Matrix& hostMatrix) const {
    copyToHost(hostMatrix, false, getDefaultStream());
}

void HIPmatrix::copyToHost(Matrix& hostMatrix, bool resizeTarget) const {
    copyToHost(hostMatrix, resizeTarget, getDefaultStream());
}

void HIPmatrix::copyToHost(Matrix& hostMatrix, bool resizeTarget, hipStream_t stream) const {
    if (resizeTarget) {
        hostMatrix.resize(_numRows, _numCols);
    } else {
        assert(isSameDims(hostMatrix));
    }
    hostMatrix.setTrans(_isTrans);

    if (getNumElements() > 0) {
        CUBLAS_CALL(cublasGetMatrixAsync(getLeadingDim(),getFollowingDim(), sizeof(float),
                                         getDevData(), getStride(), hostMatrix.getData(), hostMatrix.getLeadingDim(), stream));
        syncStream(stream);
    }
}

void HIPmatrix::copy(HIPmatrix& dest) const {
    copy(dest, getDefaultStream());
}

void HIPmatrix::copy(HIPmatrix& dest, hipStream_t stream) const {
    if (&dest != this) {
        if (!isSameDims(dest)) {
            dest.resize(*this);
        }
        copy(dest, 0, -1, 0, -1, 0, 0, stream);
    }
}

HIPmatrix& HIPmatrix::copy() const {
    HIPmatrix& c = construct();
    copy(c);
    return c;
}

void HIPmatrix::rightMult(HIPmatrix &b, float scaleAB, HIPmatrix &target) {
    rightMult(b, scaleAB, target, getDefaultStream());
}

void HIPmatrix::rightMult(HIPmatrix &b, float scaleAB, HIPmatrix &target, hipStream_t stream) {
//    if(&target != this && &target != &b) {
//        target.resize(_numRows, b.getNumCols());
//        target.setTrans(true);
//    }
    target.addProduct(*this, b, 0, scaleAB, stream);
}

void HIPmatrix::rightMult(HIPmatrix &b, float scaleAB) {
    rightMult(b, scaleAB, *this);
}

void HIPmatrix::rightMult(HIPmatrix &b, HIPmatrix& target) {
    rightMult(b, 1, target);
}

void HIPmatrix::addProduct(HIPmatrix& a, HIPmatrix &b, float scaleThis, float scaleAB) {
    addProduct(a, b, scaleThis, scaleAB, getDefaultStream());
}

/*
 * This will only work if this matrix is in column-major order! In other words,
 * if isTrans() returns true.
 */
void HIPmatrix::addProduct(HIPmatrix& a, HIPmatrix &b, float scaleThis, float scaleAB, hipStream_t stream) {
    assert(a.getNumCols() == b.getNumRows());

    if (scaleThis == 0) {
        resize(a.getNumRows(), b.getNumCols());
        setTrans(true);
    }

    assert(this->getNumRows() == a.getNumRows());
    assert(this->getNumCols() == b.getNumCols());
    assert(_isTrans);
    CUBLAS_CALL(cublasSetStream_v2(getCublasHandle(), stream));
    CUBLAS_CALL(cublasSgemm_v2(getCublasHandle(), a.getTransChar(), b.getTransChar(), a.getNumRows(), b.getNumCols(), a.getNumCols(),
                               &scaleAB, a.getDevData(), a.getStride(), b.getDevData(), b.getStride(),
                               &scaleThis, getDevData(), getStride()));
}

void HIPmatrix::addProduct(HIPmatrix& a, HIPmatrix &b) {
    addProduct(a, b, 1, 1);
}

void HIPmatrix::assertSame(NVMatrixV& a) {
    for (int i = 1; i < a.size(); ++i) {
        assert(a[i]->isSameDims(*a[0]));
        assert(a[i]->isTrans() == a[0]->isTrans());
        assert(a[i]->getStride() == a[0]->getStride());
        assert(a[i]->getDataDeviceID() == a[0]->getDataDeviceID());
    }
}

void HIPmatrix::batchedMatrixMultiply(NVMatrixV& a, NVMatrixV& b, NVMatrixV& target, float scaleTarget, float scaleAB,
                                     const float** aPtrsDev, const float** bPtrsDev, float** tgtPtrsDev) {
    batchedMatrixMultiply(a, b, target, scaleTarget, scaleAB, getDefaultStream(), aPtrsDev, bPtrsDev, tgtPtrsDev);
}

void HIPmatrix::batchedMatrixMultiply(NVMatrixV& a, NVMatrixV& b, NVMatrixV& target, float scaleTarget, float scaleAB) {
    batchedMatrixMultiply(a, b, target, scaleTarget, scaleAB, getDefaultStream());
}

void HIPmatrix::batchedMatrixMultiply(NVMatrixV& a, NVMatrixV& b, NVMatrixV& target, float scaleTarget, float scaleAB, hipStream_t stream,
                                     const float** aPtrsDev, const float** bPtrsDev, float** tgtPtrsDev) {
    assert(a.size() == b.size());
    assert(a.size() == target.size());
    assertSame(a);
    assertSame(b);
    assertSame(target);

    const int batch = a.size();
    if (batch > 0) {
        const int rows = a[0]->getNumRows(), inner = a[0]->getNumCols(), cols = b[0]->getNumCols();

        assert(inner == b[0]->getNumRows());
        assert(target[0]->getNumRows() == rows);
        assert(target[0]->getNumCols() == cols);

        const int lda = a[0]->getStride(), ldb = b[0]->getStride(), ldc = target[0]->getStride();
        hipblasOperation_t atrans = a[0]->getTransChar(), btrans = b[0]->getTransChar();

        CUBLAS_CALL(cublasSetStream_v2(getCublasHandle(), stream));
        CUBLAS_CALL(hipblasSgemmBatched(getCublasHandle(), atrans, btrans, rows, cols, inner, &scaleAB, aPtrsDev, lda, bPtrsDev, ldb, &scaleTarget, tgtPtrsDev, ldc, batch));
    }
}

void HIPmatrix::batchedMatrixMultiply(NVMatrixV& a, NVMatrixV& b, NVMatrixV& target, float scaleTarget, float scaleAB, hipStream_t stream) {
    assert(a.size() == b.size());
    assert(a.size() == target.size() || target.size() == 0);

    const int batch = a.size();
    if (batch > 0) {
        const int rows = a[0]->getNumRows(), cols = b[0]->getNumCols();

        const float* aPtrs[batch], *bPtrs[batch], *tgtPtrs[batch];
        for (int i = 0; i < batch; ++i) {
            if (target.size() <= i) {
                target.push_back(new HIPmatrix(rows, cols, true));
            }
            aPtrs[i] = a[i]->getDevData();
            bPtrs[i] = b[i]->getDevData();
            tgtPtrs[i] = target[i]->getDevData();
        }

//        const float** aPtrsDev, **bPtrsDev;
//        float **tgtPtrsDev;
//        checkCudaErrors(hipMalloc(&aPtrsDev, batch * sizeof(float*)));
//        checkCudaErrors(hipMalloc(&bPtrsDev, batch * sizeof(float*)));
//        checkCudaErrors(hipMalloc(&tgtPtrsDev, batch * sizeof(float*)));
        MemorySegment* aPtrsDev = DEVICE_MEMORY_MANAGER::getInstance(getDeviceID()).malloc(batch * sizeof(float*));
        MemorySegment* bPtrsDev = DEVICE_MEMORY_MANAGER::getInstance(getDeviceID()).malloc(batch * sizeof(float*));
        MemorySegment* tgtPtrsDev = DEVICE_MEMORY_MANAGER::getInstance(getDeviceID()).malloc(batch * sizeof(float*));

        checkCudaErrors(hipMemcpyAsync(aPtrsDev, aPtrs, batch * sizeof(float*), hipMemcpyHostToDevice, stream));
        checkCudaErrors(hipMemcpyAsync(bPtrsDev, bPtrs, batch * sizeof(float*), hipMemcpyHostToDevice, stream));
        checkCudaErrors(hipMemcpyAsync(tgtPtrsDev, tgtPtrs, batch * sizeof(float*), hipMemcpyHostToDevice, stream));

        batchedMatrixMultiply(a, b, target, scaleTarget, scaleAB, stream, const_cast<const float**>(aPtrsDev->getData<float*>()),
                                                                          const_cast<const float**>(bPtrsDev->getData<float*>()),
                                                                          tgtPtrsDev->getData<float*>());

//        checkCudaErrors(hipFree(aPtrsDev));
//        checkCudaErrors(hipFree(bPtrsDev));
//        checkCudaErrors(hipFree(tgtPtrsDev));
        DEVICE_MEMORY_MANAGER::getInstance(getDeviceID()).free(aPtrsDev);
        DEVICE_MEMORY_MANAGER::getInstance(getDeviceID()).free(bPtrsDev);
        DEVICE_MEMORY_MANAGER::getInstance(getDeviceID()).free(tgtPtrsDev);
    }
}

template <class Randomizer>
void HIPmatrix::_unaryRandomize(HIPmatrix& target, Randomizer rnd) {
    _unaryRandomize(target, rnd, getDefaultStream());
}

template <class Randomizer>
void HIPmatrix::_unaryRandomize(HIPmatrix& target, Randomizer rnd, hipStream_t stream) {
    assert(isRndInitialized());
    assert(isContiguous() && target.isContiguous());
    if (!isSameDims(target)) {
        target.resize(*this);
    }
    assert(isTrans() == target.isTrans());
    hipLaunchKernel(HIP_KERNEL_NAME(kUnaryRandomize), dim3(NUM_RND_BLOCKS), dim3(NUM_RND_THREADS_PER_BLOCK), 0, stream, getDevData(), target.getDevData(), getCurandState(), getNumElements(), rnd);
    getLastCudaError("kUnaryRandomize: Kernel execution failed");
}

template <class Randomizer>
void HIPmatrix::_binaryRandomize(HIPmatrix& data2, HIPmatrix& target, Randomizer rnd) {
    _binaryRandomize(data2, target, rnd, getDefaultStream());
}

template <class Randomizer>
void HIPmatrix::_binaryRandomize(HIPmatrix& data2, HIPmatrix& target, Randomizer rnd, hipStream_t stream) {
    assert(isRndInitialized());
    assert(isContiguous() && data2.isContiguous() && target.isContiguous());
    assert(isSameDims(data2));
    assert(isTrans() == data2.isTrans());
    if (!isSameDims(target)) {
        target.resize(*this);
    }
    assert(isTrans() == target.isTrans());
    hipLaunchKernel(HIP_KERNEL_NAME(kBinaryRandomize), dim3(NUM_RND_BLOCKS), dim3(NUM_RND_THREADS_PER_BLOCK), 0, stream, getDevData(), data2.getDevData(), target.getDevData(), getCurandState(), getNumElements(), rnd);
    getLastCudaError("kBinaryRandomize: Kernel execution failed");
}

void HIPmatrix::initRandom(unsigned long long seed, int numStreams) {
    HIPmatrix::initRandom(seed, numStreams, HIPmatrix::getDefaultStream());
}

void HIPmatrix::initRandom(unsigned long long seed, int numStreams, hipStream_t stream) {
//    printf("init random on device %d\n", getDeviceID());
    pthread_mutex_lock(_rndMutex);
    assert(!isRndInitialized(true));
    int d = getDeviceID();
//    _rndDevStates[d] = NULL;
    _rndDevThreads[d] = numStreams;
    _rndDevStates[d] = DEVICE_MEMORY_MANAGER::getInstance(d).malloc(numStreams * sizeof(curandState));
//    checkCudaErrors(hipMalloc((void **)&_rndDevStates[d], numStreams * sizeof(curandState)));
    pthread_mutex_unlock(_rndMutex);
    hipLaunchKernel(HIP_KERNEL_NAME(kSetupCurand), dim3(NUM_RND_BLOCKS), dim3(NUM_RND_THREADS_PER_BLOCK), 0, stream, getCurandState(), 1 + seed*2); // so there's no chance it'll be correlated with the other one
    getLastCudaError("kSetupCurand: Kernel execution failed");
}

void HIPmatrix::initRandom(unsigned long long seed) {
    initRandom(seed, NUM_RND_STREAMS);
}

void HIPmatrix::initRandom() {
    HIPmatrix::initRandom(time(0));
}

void HIPmatrix::initCublas() {
    int d = getDeviceID();
    pthread_mutex_lock(_cublasMutex);
    assert(_cublasHandles.count(d) == 0);
    CUBLAS_CALL(hipblasCreate(&_cublasHandles[d]));
    // It appears that hipblasCreate causes a host -> device copy on stream 0,
    // so we synchronize with it because we run everything else on other
    // streams.
    syncDevice();
    pthread_mutex_unlock(_cublasMutex);
}

void HIPmatrix::destroyCublas() {
    int d = getDeviceID();
    pthread_mutex_lock(_cublasMutex);
    assert(_cublasHandles.count(d) > 0);
    CUBLAS_CALL(hipblasDestroy(_cublasHandles[d]));
    _cublasHandles.erase(d);
    pthread_mutex_unlock(_cublasMutex);
}

hipblasHandle_t HIPmatrix::getCublasHandle() {
    return getCublasHandle(getDeviceID());
}

hipblasHandle_t HIPmatrix::getCublasHandle(int deviceID) {
    pthread_mutex_lock(_cublasMutex);
    assert(_cublasHandles.count(deviceID) > 0);
    hipblasHandle_t h = _cublasHandles[deviceID];
    pthread_mutex_unlock(_cublasMutex);
    return h;
}

hipStream_t HIPmatrix::getDefaultStream() {
    return getDefaultStream(HIPmatrix::getDeviceID());
}

hipStream_t HIPmatrix::getDefaultStream(int deviceID) {
    if (deviceID >= 0) {
        pthread_mutex_lock(_streamMutex);
        if (_defaultStreams.count(deviceID) == 0) {
            int oldDeviceID = getDeviceID();
            HIPmatrix::setDeviceID(deviceID);
            checkCudaErrors(hipStreamCreateWithFlags(&_defaultStreams[deviceID], hipStreamNonBlocking));
            HIPmatrix::setDeviceID(oldDeviceID);
        }
        hipStream_t s = _defaultStreams[deviceID];
        pthread_mutex_unlock(_streamMutex);
        return s;
    }
    return 0;
}

void HIPmatrix::syncDevice() {
    checkCudaErrors(hipDeviceSynchronize());
}

void HIPmatrix::syncStream(hipStream_t stream) {
    checkCudaErrors(hipStreamSynchronize(stream));
}

void HIPmatrix::syncStream() {
    syncStream(getDefaultStream());
}

curandState* HIPmatrix::getCurandState() {
    /*
     * Even though we're only reading from the map here, it's important to grab
     * the mutex because another thread may be writing to it.
     */
    pthread_mutex_lock(_rndMutex);
    int d = getDeviceID();
    assert(isRndInitialized(true));
    curandState* r = _rndDevStates[d]->getData<curandState>();
    pthread_mutex_unlock(_rndMutex);
    return r;
}

curandState* HIPmatrix::getCurandState(int numStreams) {
    int d = getDeviceID();
    pthread_mutex_lock(_rndMutex);
    assert(isRndInitialized(true));
    bool realloc = numStreams >  _rndDevThreads[d];
    pthread_mutex_unlock(_rndMutex);

    if (realloc) {
        destroyRandom();
        initRandom(time(0), numStreams);
    }
    return getCurandState();
}

int HIPmatrix::getDataDeviceID() const {
    if (getDevData() == NULL) {
        return DEVICE_NULL;
    }
    struct cudaPointerAttributes atts;
    checkCudaErrors(cudaPointerGetAttributes(&atts, getDevData()));
    return atts.memoryType == cudaMemoryTypeDevice ? atts.device : DEVICE_HOST;
}


int HIPmatrix::getDeviceID() {
    int d;
    checkCudaErrors(hipGetDevice(&d));
//    if (d == 0) {
//        raise(SIGABRT);
//    }
    return d;
}

void HIPmatrix::setDeviceID(int d) {
    assert(d >= 0);
//    printf("Setting device to %d\n", d);
//    if (d == 0) {
//        raise(SIGABRT);
//    }
    checkCudaErrors(hipSetDevice(d));
}

bool HIPmatrix::canAccessPeer(int srcDevice, int tgtDevice) {
    if (srcDevice == tgtDevice) {
        return true;
    }
    int canAccess;
    checkCudaErrors(hipDeviceCanAccessPeer(&canAccess, srcDevice, tgtDevice));
    return canAccess;
}

bool HIPmatrix::isRndInitialized(bool haveLock) {
    if (!haveLock) {
        pthread_mutex_lock(_rndMutex);
    }
    bool b = _rndDevStates.count(getDeviceID()) != 0;
    if (!haveLock) {
        pthread_mutex_unlock(_rndMutex);
    }
    return b;
}

bool HIPmatrix::isRndInitialized() {
    return isRndInitialized(false);
}

void HIPmatrix::destroyRandom() {
    int d = getDeviceID();
    pthread_mutex_lock(_rndMutex);
    assert(isRndInitialized(true));
//    checkCudaErrors(hipFree(_rndDevStates[d]));
    DEVICE_MEMORY_MANAGER::getInstance(d).free(_rndDevStates[d]);
    _rndDevStates.erase(d);
    _rndDevThreads.erase(d);
    pthread_mutex_unlock(_rndMutex);
}

void HIPmatrix::binarizeProbs() {
    binarizeProbs(*this);
}

void HIPmatrix::binarizeProbs(HIPmatrix& target) {
    _unaryRandomize(target, BinarizeUnaryRandomizer());
}

void HIPmatrix::randomizeUniform() {
    assert(isContiguous());
    assert(isRndInitialized());
//    CURAND_CALL(curandGenerateUniform(rndGen, _devData, getNumElements()));
    _unaryRandomize(*this, UniformUnaryRandomizer());
}

void HIPmatrix::randomizeGaussian() {
    randomizeGaussian(1);
}

void HIPmatrix::randomizeGaussian(float stdev) {
    randomizeGaussian(0, stdev);
}

void HIPmatrix::randomizeGaussian(float mean, float stdev) {
    assert(isContiguous());
    assert(isRndInitialized());
//    CURAND_CALL(curandGenerateNormal(rndGen, _devData, getNumElements(), mean, stdev));
    _unaryRandomize(*this, GaussianUnaryRandomizer(mean, stdev));
}

/*
 * Kind of a hack since we don't actually need the contents of this matrix for it,
 * so we don't really need a binary randomizer.
 */
void HIPmatrix::randomizeGaussian(HIPmatrix& stdevs) {
    randomizeGaussian(0, stdevs);
}

void HIPmatrix::randomizeGaussian(float mean, HIPmatrix& stdevs) {
    _binaryRandomize(stdevs, *this, GaussianBinaryRandomizer(mean));
}

void HIPmatrix::randomizeGaussian(float mean, float stdevMult, HIPmatrix& stdevs) {
    _binaryRandomize(stdevs, *this, ScaledGaussianBinaryRandomizer(mean, stdevMult));
}

void HIPmatrix::addGaussianNoise() {
    addGaussianNoise(1);
}

void HIPmatrix::addGaussianNoise(float stdev) {
    addGaussianNoise(stdev, *this);
}

void HIPmatrix::addGaussianNoise(float stdev, HIPmatrix& target) {
    _unaryRandomize(target, AddGaussianUnaryRandomizer(stdev));
}

void HIPmatrix::addGaussianNoise(HIPmatrix& stdevs, bool var) {
    addGaussianNoise(stdevs, var, *this);
}

void HIPmatrix::addGaussianNoise(HIPmatrix& stdevs) {
    addGaussianNoise(stdevs, false, *this);
}

void HIPmatrix::addGaussianNoise(HIPmatrix& stdevs, bool var, HIPmatrix& target) {
    if (var) {
        _binaryRandomize(stdevs, target, AddGaussianBinaryRandomizer<true>());
    } else {
        _binaryRandomize(stdevs, target, AddGaussianBinaryRandomizer<false>());
    }
}

void HIPmatrix::biggerThan(HIPmatrix& b, HIPmatrix& target) {
    applyBinary(NVMatrixBinaryOps::BiggerThan(), b, target);
}

void HIPmatrix::biggerThan(HIPmatrix& b) {
    biggerThan(b, *this);
}

void HIPmatrix::equals(HIPmatrix& b, HIPmatrix& target) {
    applyBinary(NVMatrixBinaryOps::Equals(), b, target);
}

void HIPmatrix::equals(HIPmatrix& m) {
    equals(m, *this);
}

void HIPmatrix::biggerThanVector(HIPmatrix& vec, HIPmatrix& target) {
    applyBinaryV(NVMatrixBinaryOps::BiggerThan(), vec, target);
}

void HIPmatrix::biggerThanVector(HIPmatrix& vec) {
    biggerThanVector(vec, *this);
}

void HIPmatrix::_checkBounds(int startRow, int endRow, int startCol, int endCol) const {
    assert(startRow >= 0 && startRow <= _numRows);
    assert(endRow >= startRow && endRow <= _numRows);

    assert(startCol >= 0 && startCol <= _numCols);
    assert(endCol >= startCol && endCol <= _numCols);
}

/*
 * The only place where stride is supported for now!
 * Will ALWAYS return a view of the original data, sometimes non-contiguous.
 */
HIPmatrix& HIPmatrix::slice(int startRow, int endRow, int startCol, int endCol) const {
    endRow = endRow < 0 ? this->_numRows : endRow;
    endCol = endCol < 0 ? this->_numCols : endCol;
    _checkBounds(startRow, endRow, startCol, endCol);

    if (!isTrans()) {
        return construct(new MemorySegment(this->getDevData() + startRow * _stride + startCol), endRow - startRow, endCol - startCol, _stride, false);
    }
    return construct(new MemorySegment(this->getDevData() + startCol * _stride + startRow), endRow - startRow, endCol - startCol, _stride, true);
}

/* this will NEVER return a view */
void HIPmatrix::slice(int startRow, int endRow, int startCol, int endCol, HIPmatrix& target) const {
    endRow = endRow < 0 ? this->_numRows : endRow;
    endCol = endCol < 0 ? this->_numCols : endCol;
    _checkBounds(startRow, endRow, startCol, endCol);

    int sliceRows = endRow - startRow, sliceCols = endCol - startCol;
    if (target.getNumRows() != sliceRows || target.getNumCols() != sliceCols) {
        target.resize(sliceRows, sliceCols);
    }
    this->copy(target, startRow, endRow, startCol, endCol, 0, 0);
}

HIPmatrix& HIPmatrix::sliceRows(int startRow, int endRow) const {
    return slice(startRow, endRow, 0, -1);
}

void HIPmatrix::sliceRows(int startRow, int endRow, HIPmatrix& target) const {
    slice(startRow, endRow, 0, -1, target);
}

HIPmatrix& HIPmatrix::sliceCols(int startCol, int endCol) const {
    return slice(0, -1, startCol, endCol);
}

void HIPmatrix::sliceCols(int startCol, int endCol, HIPmatrix& target) const {
    slice(0, -1, startCol, endCol, target);
}

NVMatrixV& HIPmatrix::splitRows(int numParts) {
    assert(getNumRows() % numParts == 0);
    NVMatrixV& v = *new NVMatrixV();
    int partSize = getNumRows() / numParts;
    for (int p = 0; p < numParts; ++p) {
        v.push_back(&sliceRows(p * partSize, (p+1) * partSize));
    }
    return v;
}

NVMatrixV& HIPmatrix::splitCols(int numParts) {
    assert(getNumCols() % numParts == 0);
    NVMatrixV& v = *new NVMatrixV();
    int partSize = getNumCols() / numParts;
    for (int p = 0; p < numParts; ++p) {
        v.push_back(&sliceCols(p * partSize, (p+1) * partSize));
    }
    return v;
}

/*
 * Guaranteed to not change the data if the number of elements doesn't change.
 * So you can use this to "reshape" a matrix.
 */
bool HIPmatrix::resize(int numRows, int numCols, bool trans) {
    setTrans(trans);
    bool reallocated = false;
    if (numRows != _numRows || numCols != _numCols) {
        assert(_ownsData || (_numElements == numRows * numCols && isContiguous()));
        if (_numElements != numRows * numCols) {
            if (_numElements > 0) { // free old memory
                dealloc();
            }
            if (numRows * numCols > 0) { // allocate new memory
                alloc(numCols * numRows);
            } else {
                _memSegment = NULL;
            }
            reallocated = true;
        }
        _numRows = numRows;
        _numCols = numCols;
        _numElements = numRows * numCols;
        _stride = getLeadingDim();
    }
    return reallocated;
}

bool HIPmatrix::resize(int numRows, int numCols) {
    return resize(numRows, numCols, isTrans());
}

bool HIPmatrix::resize(const HIPmatrix& like) {
    setTrans(like.isTrans());
    return resize(like.getNumRows(), like.getNumCols());
}

bool HIPmatrix::resize(const Matrix& like) {
    setTrans(like.isTrans());
    return resize(like.getNumRows(), like.getNumCols());
}

void HIPmatrix::reshape(int numRows, int numCols) {
    assert(isContiguous());
    assert(_numElements == numRows*numCols);
    _numRows = numRows;
    _numCols = numCols;
    _stride = getLeadingDim();
}

HIPmatrix& HIPmatrix::reshaped(int numRows, int numCols) const {
    assert(isContiguous());
    assert(_numElements == numRows*numCols);
    return construct(new MemorySegment(*_memSegment), numRows, numCols, -1, _isTrans);
}

void HIPmatrix::copy(HIPmatrix &dest, int srcStartRow, int srcEndRow,
                    int srcStartCol, int srcEndCol,
                    int destStartRow, int destStartCol) const {
    copy(dest, srcStartRow, srcEndRow, srcStartCol, srcEndCol, destStartRow, destStartCol, getDefaultStream());
}

void HIPmatrix::copy(HIPmatrix &dest, int srcStartRow, int srcEndRow,
                    int srcStartCol, int srcEndCol,
                    int destStartRow, int destStartCol, hipStream_t stream) const {
    srcEndRow = srcEndRow < 0 ? _numRows : srcEndRow;
    srcEndCol = srcEndCol < 0 ? _numCols : srcEndCol;
    HIPmatrix* srcSlice = &slice(srcStartRow, srcEndRow, srcStartCol, srcEndCol);
    HIPmatrix* destSlice = &dest.slice(destStartRow, destStartRow + srcEndRow - srcStartRow, destStartCol, destStartCol + srcEndCol - srcStartCol);
    if (srcSlice->isContiguous() && destSlice->isContiguous() && srcSlice->isSameDims(*destSlice) && srcSlice->isTrans() == destSlice->isTrans()) {
        // The commonest case.
        checkCudaErrors(hipMemcpyAsync(destSlice->getDevData(), srcSlice->getDevData(), srcSlice->getNumDataBytes(), hipMemcpyDefault, stream));
    } else {
        srcSlice->apply(NVMatrixOps::Identity(), *destSlice, stream);
    }
    delete srcSlice;
    delete destSlice;
}


HIPmatrix& HIPmatrix::getTranspose() {
    return construct(new MemorySegment(*_memSegment), _numCols, _numRows, _stride, !_isTrans);
}

HIPmatrix& HIPmatrix::getClone() {
    return construct(new MemorySegment(*_memSegment), _numRows, _numCols, _stride, _isTrans);
}

void HIPmatrix::transpose(HIPmatrix& target) {
    flipTrans(target);
    target.setTrans(!target.isTrans());
    target.reshape(target.getNumCols(), target.getNumRows());
}

void HIPmatrix::transpose() {
    int tmp = _numCols;
    _numCols = _numRows;
    _numRows = tmp;
    _isTrans = !_isTrans;
}

bool HIPmatrix::transpose(bool trans) {
    bool oldTrans = _isTrans;
    if (oldTrans != trans) {
        transpose();
    }
    return oldTrans;
}

/*
 * Flips the ordering of the matrix from row-major to column-major and vice versa.
 * This creates temporary storage -- not a cheap operation.
 *
 * This is not equivalent to a "hard transpose". The resultant matrix still has
 * the same dimensions, its layout in memory just changes.
 */
HIPmatrix& HIPmatrix::flipTrans() {
    HIPmatrix& meTrans = construct(*this);
    flipTrans(meTrans);
    return meTrans;
}

void HIPmatrix::flipTrans(HIPmatrix& target) {
    flipTrans(target, getDefaultStream());
}

void HIPmatrix::flipTrans(HIPmatrix& target, hipStream_t stream) {
    assert(&target != this);
    target.resize(_numRows, _numCols);
    target.setTrans(!isTrans());
//    target.printShape("target");
//    this->printShape("this");
    apply(NVMatrixOps::Identity(), target, stream);
}

void HIPmatrix::squaredDiff(HIPmatrix& b) {
    squaredDiff(b, *this);
}

void HIPmatrix::squaredDiff(HIPmatrix& b, HIPmatrix& target) {
    applyBinary(NVMatrixBinaryOps::SquaredDiff(), b, target);
}

void HIPmatrix::add(HIPmatrix& b, float scaleA, float scaleB, HIPmatrix& target) {
    add(b, scaleA, scaleB, target, HIPmatrix::getDefaultStream());
}

void HIPmatrix::add(HIPmatrix& b, float scaleA, float scaleB, HIPmatrix& target, hipStream_t stream) {
    if (scaleA == 0) {
        b.scale(scaleB, target, stream);
    } else if (scaleB == 0) {
        scale(scaleA, target, stream);
    } else if (scaleA == 1 && scaleB == 1) { // slight optimization
        applyBinary(NVMatrixBinaryOps::Add(), b, target, stream);
    } else if (scaleA == 1) {
        applyBinary(NVMatrixBinaryOps::WeightedAdd1(scaleB), b, target, stream);
    } else {
        applyBinary(NVMatrixBinaryOps::WeightedAdd(scaleA, scaleB), b, target, stream);
    }
}

void HIPmatrix::add(HIPmatrix& b, float scaleB, HIPmatrix& target) {
    add(b, 1, scaleB, target);
}

void HIPmatrix::add(HIPmatrix& b, HIPmatrix& target) {
    add(b, 1, target);
}

void HIPmatrix::add(HIPmatrix& b, float scaleB) {
    add(b, scaleB, *this);
}

void HIPmatrix::add(HIPmatrix& b, float scaleA, float scaleB) {
    add(b, scaleA, scaleB, *this);
}

void HIPmatrix::add(HIPmatrix& b) {
    add(b, 1, *this);
}

void HIPmatrix::subtract(HIPmatrix& b, HIPmatrix& target) {
    add(b, -1, target);
}

void HIPmatrix::subtract(HIPmatrix& b) {
    add(b, -1);
}

void HIPmatrix::eltwiseMult(HIPmatrix& b, HIPmatrix& target) {
    applyBinary(NVMatrixBinaryOps::Multiply(), b, target);
}

void HIPmatrix::eltwiseMult(HIPmatrix& b) {
    eltwiseMult(b, *this);
}

void HIPmatrix::eltwiseDivide(HIPmatrix& b, HIPmatrix& target) {
    applyBinary(NVMatrixBinaryOps::Divide(), b, target);
}

void HIPmatrix::eltwiseDivide(HIPmatrix& b) {
    eltwiseDivide(b, *this);
}

void HIPmatrix::tile(int timesY, int timesX, HIPmatrix& target) {
    tile(timesY, timesX, target, getDefaultStream());
}

void HIPmatrix::tile(int timesY, int timesX, HIPmatrix& target, hipStream_t stream) {
    assert(isContiguous() && target.isContiguous());
    assert(timesX > 0 && timesY > 0);
    target.resize(_numRows*timesY, _numCols*timesX);
    target.setTrans(_isTrans);
    if(!isTrans()) {
        hipLaunchKernel(HIP_KERNEL_NAME(kTile), dim3(NUM_TILE_BLOCKS), dim3(NUM_TILE_THREADS_PER_BLOCK), 0, stream, getDevData(), target.getDevData(), _numCols, _numRows, target._numCols, target._numRows);
    } else {
        hipLaunchKernel(HIP_KERNEL_NAME(kTile), dim3(NUM_TILE_BLOCKS), dim3(NUM_TILE_THREADS_PER_BLOCK), 0, stream, getDevData(), target.getDevData(), _numRows, _numCols, target._numRows, target._numCols);
    }
    getLastCudaError("Kernel execution failed");
}

void HIPmatrix::addVector(HIPmatrix& vec, float scaleVec, HIPmatrix& target) {
    addVector(vec, scaleVec, target, getDefaultStream());
}

void HIPmatrix::addVector(HIPmatrix& vec, float scaleVec, HIPmatrix& target, hipStream_t stream) {
    applyBinaryV(NVMatrixBinaryOps::ScaledAdd(scaleVec), vec, target, stream);
}

void HIPmatrix::addVector(HIPmatrix& vec) {
    addVector(vec, 1);
}

void HIPmatrix::addVector(HIPmatrix& vec, float scaleVec) {
    addVector(vec, scaleVec, *this);
}

void HIPmatrix::addVector(HIPmatrix& vec, HIPmatrix& target) {
    addVector(vec, 1, target);
}

void HIPmatrix::equalsVector(HIPmatrix& vec, HIPmatrix& target) {
    applyBinaryV(NVMatrixBinaryOps::Equals(), vec, target);
}

void HIPmatrix::equalsVector(HIPmatrix& vec) {
    equalsVector(vec, *this);
}

void HIPmatrix::eltwiseMultByVector(HIPmatrix& vec, HIPmatrix& target) {
    eltwiseMultByVector(vec, target, getDefaultStream());
}

void HIPmatrix::eltwiseMultByVector(HIPmatrix& vec, HIPmatrix& target, hipStream_t stream) {
    applyBinaryV(NVMatrixBinaryOps::Multiply(), vec, target, stream);
}

void HIPmatrix::eltwiseMultByVector(HIPmatrix& vec, hipStream_t stream) {
    eltwiseMultByVector(vec, *this, stream);
}

void HIPmatrix::eltwiseMultByVector(HIPmatrix& vec) {
    eltwiseMultByVector(vec, *this);
}

void HIPmatrix::eltwiseDivideByVector(HIPmatrix& vec) {
    eltwiseDivideByVector(vec,  *this);
}

void HIPmatrix::eltwiseDivideByVector(HIPmatrix& vec, HIPmatrix& target) {
    applyBinaryV(NVMatrixBinaryOps::Divide(), vec, target);
}

template<class Agg, class UnaryOp, class BinaryOp>
void HIPmatrix::_aggregate(int axis, HIPmatrix& target, Agg agg, UnaryOp uop, BinaryOp bop, hipStream_t stream) {
    _aggregate(axis, target, agg, uop, bop, stream, NULL);
}

/*
 * TODO: this is a mess, fix it. it works pretty fast but it's too ugly.
 * TODO: this function is _really_ bad for very long aggregations of few columns.
 */
template<class Agg, class UnaryOp, class BinaryOp>
void HIPmatrix::_aggregate(int axis, HIPmatrix& target, Agg agg, UnaryOp uop, BinaryOp bop, hipStream_t stream, HIPmatrix* tmp) {
    assert(axis == 0 || axis == 1);
    assert(isContiguous()  && target.isContiguous());
    assert(&target != this);
    int width = _isTrans ? _numRows : _numCols;
    int height = _isTrans ? _numCols : _numRows;

    target.setTrans(_isTrans);
    assert(width > 0);
    assert(height > 0);
    if((axis == 0 && !_isTrans) || (axis == 1 && _isTrans)) { //col sum
        target.resize(!_isTrans ? 1 : _numRows, !_isTrans ? _numCols : 1);
//        int height = getFollowingDim();
        if ((height <= 2048 || width >= 4096)) {
            int numBlocks = DIVUP(width, NUM_SUM_COLS_THREADS_PER_BLOCK);
            assert(numBlocks * NUM_SUM_COLS_THREADS_PER_BLOCK >= width);
            assert(numBlocks < NUM_BLOCKS_MAX);
            hipLaunchKernel(HIP_KERNEL_NAME(kDumbAggCols<Agg, UnaryOp, BinaryOp>), dim3(numBlocks), dim3(NUM_SUM_COLS_THREADS_PER_BLOCK), 0, stream, getTextureObject(), target.getDevData(), width, height, agg, uop, bop);
            getLastCudaError("kDumbAggCols: Kernel execution failed");
        } else { // Specialize the case when we have very long columns and few of them
            const int sumLength = 128;
            bool deltmp = tmp == NULL;
            if (tmp == NULL) {
                tmp = new HIPmatrix(false);
            }

            int numBlocksX = DIVUP(width, NUM_SUM_COLS_THREADS_PER_BLOCK);
            int numBlocksY = DIVUP(height, sumLength);
            tmp->resize(numBlocksY, width);

            dim3 blocks(numBlocksX, numBlocksY);
            dim3 threads(NUM_SUM_COLS_THREADS_PER_BLOCK);
            hipLaunchKernel(HIP_KERNEL_NAME(kAggCols<Agg, UnaryOp>), dim3(blocks), dim3(threads), 0, stream, getTextureObject(), tmp->getDevData(), width, height, sumLength, agg, uop);
            getLastCudaError("kAggCols: Kernel execution failed");

            int numBlocks = DIVUP(width, NUM_SUM_COLS_THREADS_PER_BLOCK);
            hipLaunchKernel(HIP_KERNEL_NAME(kDumbAggCols<Agg, NVMatrixOps::Identity, BinaryOp>), dim3(numBlocks), dim3(NUM_SUM_COLS_THREADS_PER_BLOCK), 0, stream, tmp->getTextureObject(), target.getDevData(), width, numBlocksY, agg, NVMatrixOps::Identity(), bop);
            getLastCudaError("kDumbAggCols: Kernel execution failed");
            if (deltmp) {
                delete tmp;
            }
        }
    } else { // row sum
        target.resize(_isTrans ? 1 : _numRows, _isTrans ? _numCols : 1);
        if (width > 1) {
            if (height >= 16384) { // linear aggregation
                int numBlocksX = 1;
                int numBlocksY = DIVUP(height, AGG_SHORT_ROWS_THREADS_Y*AGG_SHORT_ROWS_LOOPS_Y);
                int numThreadsX = width <= 4 ? 4 : width <= 8 ? 8 : width <= 12 ? 12 : width <= 16 ? 16 : AGG_SHORT_ROWS_THREADS_X;
                int numThreadsY = AGG_SHORT_ROWS_THREADS_Y;
                while (numBlocksY > NUM_BLOCKS_MAX) {
                    numBlocksY = DIVUP(numBlocksY,2);
                    numBlocksX *= 2;
                }
                dim3 grid(numBlocksX, numBlocksY), threads(numThreadsX, numThreadsY);
                if(width <= 16) {
                    if(width <= 4) {
                        hipLaunchKernel(HIP_KERNEL_NAME(kAggShortRows<Agg, UnaryOp, BinaryOp, 1, 4>), dim3(grid), dim3(threads), 0, stream, getDevData(), target.getDevData(),width, height, agg, uop, bop);
                    } else if(width <= 8) {
                        hipLaunchKernel(HIP_KERNEL_NAME(kAggShortRows<Agg, UnaryOp, BinaryOp, 1, 8>), dim3(grid), dim3(threads), 0, stream, getDevData(), target.getDevData(),width, height, agg, uop, bop);
                    } else if(width <= 12) {
                        hipLaunchKernel(HIP_KERNEL_NAME(kAggShortRows<Agg, UnaryOp, BinaryOp, 1, 12>), dim3(grid), dim3(threads), 0, stream, getDevData(), target.getDevData(),width, height, agg, uop, bop);
                    } else {
                        hipLaunchKernel(HIP_KERNEL_NAME(kAggShortRows<Agg, UnaryOp, BinaryOp, 1, 16>), dim3(grid), dim3(threads), 0, stream, getDevData(), target.getDevData(),width, height, agg, uop, bop);
                    }
                } else if(width <= 32) {
                    hipLaunchKernel(HIP_KERNEL_NAME(kAggShortRows<Agg, UnaryOp, BinaryOp, 2, AGG_SHORT_ROWS_THREADS_X>), dim3(grid), dim3(threads), 0, stream, getDevData(), target.getDevData(),width, height, agg, uop, bop);
                } else if(width <= 48){
                    hipLaunchKernel(HIP_KERNEL_NAME(kAggShortRows<Agg, UnaryOp, BinaryOp, 3, AGG_SHORT_ROWS_THREADS_X>), dim3(grid), dim3(threads), 0, stream, getDevData(), target.getDevData(),width, height, agg, uop, bop);
                } else if(width <= 64){
                    hipLaunchKernel(HIP_KERNEL_NAME(kAggShortRows<Agg, UnaryOp, BinaryOp, 4, AGG_SHORT_ROWS_THREADS_X>), dim3(grid), dim3(threads), 0, stream, getDevData(), target.getDevData(),width, height, agg, uop, bop);
                } else {
                    hipLaunchKernel(HIP_KERNEL_NAME(kAggShortRows2<Agg, UnaryOp, BinaryOp>), dim3(grid), dim3(threads), 0, stream, getDevData(), target.getDevData(),width, height, agg, uop, bop);
                }
            } else {
                if (width >= 512) {
                    // NOTE: this is the only case which I bothered to try to optimize for Kepler
                    dim3 threads(AWR_NUM_THREADS);
                    dim3 blocks(1, height);
                    hipLaunchKernel(HIP_KERNEL_NAME(kAggRows_wholerow_nosync), dim3(blocks), dim3(threads), 0, stream, getDevData(), target.getDevData(), width, height, agg, uop, bop);
                } else {

                    int numThreadsX = width <= 64 ? 32 : (width <= 128 ? 64 : (width <= 256 ? 128 : (width <= 512 ? 256 : 512)));
                    int numThreadsY = 1;
                    int numBlocksX = DIVUP(width, 2*numThreadsX);
                    int numBlocksY = std::min(height, NUM_BLOCKS_MAX);

                    dim3 grid(numBlocksX, numBlocksY), threads(numThreadsX, numThreadsY);
                    assert(numBlocksX <= NUM_BLOCKS_MAX);
                    assert(numBlocksY <= NUM_BLOCKS_MAX);

                    if(width <= 64) {
                        hipLaunchKernel(HIP_KERNEL_NAME(kAggRows<Agg, UnaryOp, BinaryOp, 32>), dim3(grid), dim3(threads), 0, stream, getDevData(), target.getDevData(),
                                                   width, height, target.getLeadingDim(), agg, uop, bop);
                    } else if(width <= 128) {
                        hipLaunchKernel(HIP_KERNEL_NAME(kAggRows<Agg, UnaryOp, BinaryOp, 64>), dim3(grid), dim3(threads), 0, stream, getDevData(), target.getDevData(),
                                                   width, height, target.getLeadingDim(), agg, uop, bop);
                    } else if(width <= 256) {
                        hipLaunchKernel(HIP_KERNEL_NAME(kAggRows<Agg, UnaryOp, BinaryOp, 128>), dim3(grid), dim3(threads), 0, stream, getDevData(), target.getDevData(),
                                                   width, height, target.getLeadingDim(), agg, uop, bop);
                    } else if(width <= 512) {
                        hipLaunchKernel(HIP_KERNEL_NAME(kAggRows<Agg, UnaryOp, BinaryOp, 256>), dim3(grid), dim3(threads), 0, stream, getDevData(), target.getDevData(),
                                                   width, height, target.getLeadingDim(), agg, uop, bop);
                    } else {
                        hipLaunchKernel(HIP_KERNEL_NAME(kAggRows<Agg, UnaryOp, BinaryOp, 512>), dim3(grid), dim3(threads), 0, stream, getDevData(), target.getDevData(),
                                                   width, height, target.getLeadingDim(), agg, uop, bop);
                    }

                    getLastCudaError("agg rows: Kernel execution failed");
                }
            }
        } else {
            target.applyBinary(NVMatrixBinaryOps::CompositeSecond<UnaryOp, BinaryOp>(uop, bop), *this, target, stream);
//            copy(target, stream);
        }
    }
}

template<class Agg, class UnaryOp, class BinaryOp>
void HIPmatrix::_aggregate(int axis, HIPmatrix& target, Agg agg, UnaryOp uop, BinaryOp bop) {
    _aggregate(axis, target, agg, uop, bop, getDefaultStream());
}

template<class Agg, class BinaryOp>
void HIPmatrix::_aggregate(int axis, HIPmatrix& target, Agg agg, BinaryOp bop) {
    _aggregate(axis, target, agg, NVMatrixOps::Identity(), bop, getDefaultStream());
}

template<class Agg, class BinaryOp>
void HIPmatrix::_aggregate(int axis, HIPmatrix& target, Agg agg, BinaryOp bop, hipStream_t stream) {
    _aggregate(axis, target, agg, NVMatrixOps::Identity(), bop, stream);
}

template<class Agg, class UnaryOp, class BinaryOp>
HIPmatrix& HIPmatrix::_aggregate(int axis, Agg agg, UnaryOp uop, BinaryOp bop) {
    HIPmatrix &sumVec = construct();
    _aggregate(axis, sumVec, agg, uop, bop);
    return sumVec;
}

template<class Agg, class UnaryOp, class BinaryOp>
HIPmatrix& HIPmatrix::_aggregate(int axis, Agg agg, UnaryOp uop, BinaryOp bop, hipStream_t stream) {
    HIPmatrix &sumVec = construct();
    _aggregate(axis, sumVec, agg, uop, bop, stream);
    return sumVec;
}

template<class Agg, class BinaryOp>
HIPmatrix& HIPmatrix::_aggregate(int axis, Agg agg, BinaryOp bop) {
    return _aggregate(axis, agg, NVMatrixOps::Identity(), bop);
}

template<class Agg, class BinaryOp>
HIPmatrix& HIPmatrix::_aggregate(int axis, Agg agg, BinaryOp bop, hipStream_t stream) {
    return _aggregate(axis, agg, NVMatrixOps::Identity(), bop, stream);
}



template<class Agg, class UnaryOp, class BinaryOp>
void HIPmatrix::_aggregate(int axis, HIPmatrix& target, Agg agg, UnaryOp uop, BinaryOp bop, HIPmatrix& tmp) {
    _aggregate(axis, target, agg, uop, bop, getDefaultStream(), tmp);
}

template<class Agg, class BinaryOp>
void HIPmatrix::_aggregate(int axis, HIPmatrix& target, Agg agg, BinaryOp bop, HIPmatrix& tmp) {
    _aggregate(axis, target, agg, NVMatrixOps::Identity(), bop, getDefaultStream(), &tmp);
}

template<class Agg, class BinaryOp>
void HIPmatrix::_aggregate(int axis, HIPmatrix& target, Agg agg, BinaryOp bop, hipStream_t stream, HIPmatrix& tmp) {
    _aggregate(axis, target, agg, NVMatrixOps::Identity(), bop, stream, &tmp);
}

template<class Agg, class UnaryOp, class BinaryOp>
HIPmatrix& HIPmatrix::_aggregate(int axis, Agg agg, UnaryOp uop, BinaryOp bop, HIPmatrix& tmp) {
    HIPmatrix &sumVec = construct();
    _aggregate(axis, sumVec, agg, uop, bop, tmp);
    return sumVec;
}

template<class Agg, class UnaryOp, class BinaryOp>
HIPmatrix& HIPmatrix::_aggregate(int axis, Agg agg, UnaryOp uop, BinaryOp bop, hipStream_t stream, HIPmatrix& tmp) {
    HIPmatrix &sumVec = construct();
    _aggregate(axis, sumVec, agg, uop, bop, stream, tmp);
    return sumVec;
}

template<class Agg, class BinaryOp>
HIPmatrix& HIPmatrix::_aggregate(int axis, Agg agg, BinaryOp bop, HIPmatrix& tmp) {
    return _aggregate(axis, agg, NVMatrixOps::Identity(), bop, tmp);
}

template<class Agg, class BinaryOp>
HIPmatrix& HIPmatrix::_aggregate(int axis, Agg agg, BinaryOp bop, hipStream_t stream, HIPmatrix& tmp) {
    return _aggregate(axis, agg, NVMatrixOps::Identity(), bop, stream, tmp);
}

void HIPmatrix::inRangeInc(float lower, float upper) {
    inRangeInc(lower, upper, *this);
}
void HIPmatrix::inRangeInc(float lower, float upper, HIPmatrix& target) {
    apply(NVMatrixOps::InRange<false>(lower, upper), target);
}

void HIPmatrix::inRangeExc(float lower, float upper) {
    inRangeExc(lower, upper, *this);
}

void HIPmatrix::inRangeExc(float lower, float upper, HIPmatrix& target) {
    apply(NVMatrixOps::InRange<true>(lower, upper), target);
}

void HIPmatrix::biggerThanScalar(float scalar) {
    biggerThanScalar(scalar, *this);
}

void HIPmatrix::biggerThanScalar(float scalar, HIPmatrix& target) {
    apply(NVMatrixOps::BiggerThanScalar(scalar), target);
}

void HIPmatrix::smallerThanScalar(float scalar) {
    smallerThanScalar(scalar, *this);
}

void HIPmatrix::smallerThanScalar(float scalar, HIPmatrix& target) {
    apply(NVMatrixOps::SmallerThanScalar(scalar), target);
}

void HIPmatrix::addScalar(float scaleThis, float scalar, HIPmatrix& target) {
    apply(NVMatrixOps::WeightedAddScalar(scaleThis, scalar), target);
}

void HIPmatrix::addScalar(float scalar, HIPmatrix& target) {
    apply(NVMatrixOps::AddScalar(scalar), target);
}

void HIPmatrix::addScalar(float scalar) {
    addScalar(scalar, *this);
}

void HIPmatrix::minWithScalar(float scalar, HIPmatrix& target) {
    apply(NVMatrixOps::MinWithScalar(scalar), target);
}

void HIPmatrix::minWithScalar(float scalar) {
    minWithScalar(scalar, *this);
}

void HIPmatrix::maxWithScalar(float scalar, HIPmatrix& target) {
    apply(NVMatrixOps::MaxWithScalar(scalar), target);
}

void HIPmatrix::maxWithScalar(float scalar) {
    maxWithScalar(scalar, *this);
}

void HIPmatrix::pow(float p, HIPmatrix& target) {
    apply(NVMatrixOps::Pow(p), target);
}

void HIPmatrix::pow(float p) {
    pow(p, *this);
}

void HIPmatrix::scale(float _scale) {
    scale(_scale, *this);
}

void HIPmatrix::scale(float _scale, hipStream_t stream) {
    scale(_scale, *this, stream);
}

void HIPmatrix::scale(float _scale, HIPmatrix& target) {
    scale(_scale, target, HIPmatrix::getDefaultStream());
}

void HIPmatrix::scale(float _scale, HIPmatrix& target, hipStream_t stream) {
    if (_scale != 1 || &target != this) { // optimize away scale by 1
        if (_scale == 1) {
            copy(target, stream);
        } else {
            apply(NVMatrixOps::MultByScalar(_scale), target, stream);
        }
    }
}

void HIPmatrix::zero() {
    apply(NVMatrixOps::Zero());
}

void HIPmatrix::zero(HIPmatrix& like) {
    resize(like);
    zero();
}

void HIPmatrix::max(int axis, HIPmatrix& target) {
    _aggregate(axis, target, NVMatrixAggs::Max(), NVMatrixBinaryOps::Second());
}

void HIPmatrix::max(int axis, HIPmatrix& target, HIPmatrix& tmp) {
    _aggregate(axis, target, NVMatrixAggs::Max(), NVMatrixBinaryOps::Second(), tmp);
}

void HIPmatrix::addSum(HIPmatrix& a, int axis, float scaleThis, float scaleSum) {
    addSum(a, axis, scaleThis, scaleSum, getDefaultStream());
}

void HIPmatrix::addSum(HIPmatrix& a, int axis, float scaleThis, float scaleSum, hipStream_t stream) {
    if (scaleThis != 0) {
        a._aggregate(axis, *this, NVMatrixAggs::Sum(), NVMatrixBinaryOps::WeightedAdd(scaleThis, scaleSum), stream);
    } else {
        a._aggregate(axis, *this, NVMatrixAggs::Sum(), NVMatrixBinaryOps::SecondScaled(scaleSum), stream);
    }
}

void HIPmatrix::addMax(HIPmatrix& a, int axis, float scaleThis, float scaleMax) {
    addMax(a, axis, scaleThis, scaleMax, getDefaultStream());
}

void HIPmatrix::addMax(HIPmatrix& a, int axis, float scaleThis, float scaleMax, hipStream_t stream) {
    if (scaleThis != 0) {
        a._aggregate(axis, *this, NVMatrixAggs::Max(), NVMatrixBinaryOps::WeightedAdd(scaleThis, scaleMax), stream);
    } else {
        a._aggregate(axis, *this, NVMatrixAggs::Max(), NVMatrixBinaryOps::SecondScaled(scaleMax), stream);
    }
}

void HIPmatrix::sum(int axis, HIPmatrix& target) {
    sum(axis, target, getDefaultStream());
}

void HIPmatrix::sum(int axis, HIPmatrix& target, hipStream_t stream) {
    _aggregate(axis, target, NVMatrixAggs::Sum(), NVMatrixBinaryOps::Second(), stream);
}

void HIPmatrix::sum(int axis, HIPmatrix& target, HIPmatrix& tmp) {
    sum(axis, target, getDefaultStream(), tmp);
}

void HIPmatrix::sum(int axis, HIPmatrix& target, hipStream_t stream, HIPmatrix& tmp) {
    _aggregate(axis, target, NVMatrixAggs::Sum(), NVMatrixBinaryOps::Second(), stream, tmp);
}

void HIPmatrix::sumOfSquares(int axis, HIPmatrix& target) {
    sumOfSquares(axis, target, getDefaultStream());
}

void HIPmatrix::sumOfSquares(int axis, HIPmatrix& target, hipStream_t stream) {
    _aggregate(axis, target, NVMatrixAggs::Sum(), NVMatrixOps::Square(), NVMatrixBinaryOps::Second(), stream);
}

void HIPmatrix::min(int axis, HIPmatrix& target) {
    _aggregate(axis, target, NVMatrixAggs::Min(), NVMatrixBinaryOps::Second());
}

HIPmatrix& HIPmatrix::max(int axis) {
    return _aggregate(axis, NVMatrixAggs::Max(), NVMatrixBinaryOps::Second());
}

HIPmatrix& HIPmatrix::sum(int axis) {
    return _aggregate(axis, NVMatrixAggs::Sum(), NVMatrixBinaryOps::Second());
}

HIPmatrix& HIPmatrix::min(int axis) {
    return _aggregate(axis, NVMatrixAggs::Min(), NVMatrixBinaryOps::Second());
}

HIPmatrix& HIPmatrix::sumOfSquares(int axis) {
    return _aggregate(axis, NVMatrixAggs::Sum(), NVMatrixOps::Square(), NVMatrixBinaryOps::Second());
}

void HIPmatrix::_sum_setParams(int n, dim3* blocks, dim3* threads) {
    *threads = dim3(DP_BLOCKSIZE);
    *blocks = dim3(std::min(CPUSUM_MAX, DIVUP(n, DP_BLOCKSIZE)));
}

float HIPmatrix::mean() {
    return sum() / getNumElements();
}

float HIPmatrix::sum() {
    return _totalAgg(NVMatrixAggs::Sum());
}

float HIPmatrix::sum(HIPmatrix& tmpbuf) {
    return _totalAgg(NVMatrixAggs::Sum(), tmpbuf, getDefaultStream());
}

float HIPmatrix::max() {
    return _totalAgg(NVMatrixAggs::Max());
}

float HIPmatrix::min() {
    return _totalAgg(NVMatrixAggs::Min());
}

float HIPmatrix::countNan() {
    return _totalAgg(NVMatrixAggs::CountNan());
}

float HIPmatrix::countInf() {
    return _totalAgg(NVMatrixAggs::CountInf());
}

template<class Agg>
float HIPmatrix::_totalAgg(Agg agg) {
    return _totalAgg(agg, getDefaultStream());
}

template<class Agg>
float HIPmatrix::_totalAgg(Agg agg, hipStream_t stream) {
    HIPmatrix tmp;
    return _totalAgg(agg, tmp, stream);
}

template<class Agg>
float HIPmatrix::_totalAgg(Agg agg, HIPmatrix& tmpbuf, hipStream_t stream) {
    assert(isContiguous());
    dim3 blocks, threads;
    // Sum most of it on GPU

    _sum_setParams(getNumElements(), &blocks, &threads);
    tmpbuf.resize(1, blocks.x);
    hipLaunchKernel(HIP_KERNEL_NAME(kTotalAgg), dim3(blocks), dim3(threads), 0, stream, getDevData(), tmpbuf.getDevData(), getNumElements(), agg);
    getLastCudaError("kTotalAgg: Kernel execution failed");
    // Don't need to sync because we copyToHost in the same stream, so it's serialized
//    HIPmatrix::syncStream(stream);
    return tmpbuf.cpuAgg(agg, stream);
}
template<class Agg>
float HIPmatrix::cpuAgg(Agg agg, hipStream_t stream) {
    Matrix bufCPU(getNumRows(), getNumCols());
    copyToHost(bufCPU, false, stream);
    if (getNumElements() > 1) { // Sum remainder on CPU
        if (typeid(Agg) == typeid(NVMatrixAggs::Sum)) {
            return bufCPU.sum();
        } else if (typeid(Agg) == typeid(NVMatrixAggs::Max)) {
            return bufCPU.max();
        } else if (typeid(Agg) == typeid(NVMatrixAggs::Min)) {
            return bufCPU.min();
        } else if (typeid(Agg) == typeid(NVMatrixAggs::CountNan)) {
            return bufCPU.hasNan(); //yea, it's not the same, who cares
        } else if (typeid(Agg) == typeid(NVMatrixAggs::CountInf)) {
            return bufCPU.hasInf();
        } else {
            assert(false);
        }
    }
    return bufCPU(0,0);
}

float HIPmatrix::dotProduct(HIPmatrix& b) {
    return dotProduct(b, getDefaultStream());
}

float HIPmatrix::dotProduct(HIPmatrix& b, hipStream_t stream) {
    HIPmatrix tmp;
    return dotProduct(b, tmp, stream);
}

/*
 * Fast dot product only for matrices with same transposedness.
 */
float HIPmatrix::dotProduct(HIPmatrix& b, HIPmatrix& tmp, hipStream_t stream) {
    assert(isContiguous() && b.isContiguous());
    assert(isSameDims(b));
    assert(isTrans() == b.isTrans()); // see?
    dim3 blocks, threads;
    _sum_setParams(getNumElements(), &blocks, &threads);
//    HIPmatrix target(1, blocks.x);
    tmp.resize(1, blocks.x);
    hipLaunchKernel(HIP_KERNEL_NAME(kDotProduct_r), dim3(blocks), dim3(threads), 0, stream, getDevData(), b.getDevData(), tmp.getDevData(), getNumElements());
    getLastCudaError("kDotProduct_r: Kernel execution failed");
//    hipDeviceSynchronize();
//    syncStream(stream);
//    return tmp._totalAgg(NVMatrixAggs::Sum(), stream);
    return tmp.cpuAgg(NVMatrixAggs::Sum(), stream);
}

float HIPmatrix::norm2() {
    return dotProduct(*this);
}

float HIPmatrix::norm() {
    return sqrt(norm2());
}

void HIPmatrix::print(int startRow, int rows, int startCol, int cols) const {
//    hipDeviceSynchronize();
    syncDevice();
    Matrix hm = Matrix(_numRows, _numCols);
    copyToHost(hm);
    hm.print(startRow, rows, startCol, cols);
}

void HIPmatrix::print(int rows, int cols) const {
    print(0, rows, 0, cols);
}

void HIPmatrix::printShape(const char* name) const {
    printf("%s: %dx%d\n", name, _numRows, _numCols);
}

void HIPmatrix::alloc(int numElements) {
    _memSegment = DEVICE_MEMORY_MANAGER::getInstance(getDeviceID()).malloc(numElements * sizeof(float));
}

void HIPmatrix::dealloc() {
    DEVICE_MEMORY_MANAGER::getInstance(_memSegment->getDeviceID()).free(_memSegment);
    _memSegment = NULL;
    deallocTexture();
}

void HIPmatrix::deallocTexture() {
    if (_texObj != 0) {
        checkCudaErrors(cudaDestroyTextureObject(_texObj));
        _texObj = 0;
    }
}

cudaTextureObject_t HIPmatrix::getTextureObject() {
   if (_texObj == 0) {
       assert(isContiguous());
       //size_t memFree, memTotal;

       struct cudaResourceDesc resDesc;
       memset(&resDesc, 0, sizeof(resDesc));
       resDesc.resType = cudaResourceTypeLinear;
       resDesc.res.linear.devPtr = getDevData();
       resDesc.res.linear.sizeInBytes = getNumDataBytes();
       resDesc.res.linear.desc = hipCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
       struct cudaTextureDesc texDesc;
       memset(&texDesc, 0, sizeof(texDesc));
       checkCudaErrors(cudaCreateTextureObject(&_texObj, &resDesc, &texDesc, NULL));
   }
   assert(_texObj != 0);
   return _texObj;
}

HIPmatrix& HIPmatrix::construct() const {
    return *new HIPmatrix();
}
HIPmatrix& HIPmatrix::construct(bool isTrans) const {
    return *new HIPmatrix(isTrans);
}
HIPmatrix& HIPmatrix::construct(int numRows, int numCols, bool isTrans) const {
    return *new HIPmatrix(numRows, numCols, isTrans);
}
HIPmatrix& HIPmatrix::construct(const Matrix& like, bool copy) const {
    return *new HIPmatrix(like, copy);
}
HIPmatrix& HIPmatrix::construct(const HIPmatrix& like, bool copy) const {
    return *new HIPmatrix(like, copy);
}
HIPmatrix& HIPmatrix::construct(const HIPmatrix& like) const {
    return *new HIPmatrix(like);
}
HIPmatrix& HIPmatrix::construct(const Matrix& like) const {
    return *new HIPmatrix(like);
}
HIPmatrix& HIPmatrix::construct(MemorySegment* mem, int numRows, int numCols, int stride, bool isTrans) const {
    return *new HIPmatrix(mem, numRows, numCols, stride, isTrans);
}

std::pair<size_t, size_t> HIPmatrix::getCudaMemorySize() {
    size_t memFree, memTotal;
    checkCudaErrors(hipMemGetInfo(&memFree, &memTotal));
    return std::pair<size_t,size_t>(memFree, memTotal);
}


/* ================
 * HostNVMatrix
 * ================
 */
HostNVMatrix::~HostNVMatrix() {
    if (_ownsData && _numElements > 0) {
        dealloc();
    } else {
        // dealloc frees the mem segment. But if this is a view,
        // then we need to delete the mem segment object.
//        assert(_memSegment == NULL || _memSegment->getSize() == 0);
        delete _memSegment;
    }
    _deleted = true;
}
HostNVMatrix::HostNVMatrix() : HIPmatrix() {
    _init(false);
}
HostNVMatrix::HostNVMatrix(bool isTrans) {
    _init(isTrans);
}
HostNVMatrix::HostNVMatrix(int numRows, int numCols, bool isTrans)  {
    _init(isTrans);
    resize(numRows, numCols);
}
HostNVMatrix::HostNVMatrix(const Matrix& like, bool copy)  {
    _init(like.isTrans());
    resize(like.getNumRows(), like.getNumCols());
    if (copy) {
        copyFromHost(like);
    }
}
HostNVMatrix::HostNVMatrix(const HIPmatrix& like, bool copy)  {
    _init(like.isTrans());
    resize(like.getNumRows(), like.getNumCols());
    if (copy) {
        like.copy(*this);
    }
}
HostNVMatrix::HostNVMatrix(const HIPmatrix& like)  {
    _init(like.isTrans());
    resize(like.getNumRows(), like.getNumCols());
}
HostNVMatrix::HostNVMatrix(const Matrix& like) {
    _init(false);
    resize(like.getNumRows(), like.getNumCols());
}
HostNVMatrix::HostNVMatrix(MemorySegment* mem, int numRows, int numCols, int stride, bool isTrans)
    : HIPmatrix(mem, numRows, numCols, stride, isTrans) {
}

HIPmatrix& HostNVMatrix::construct() const {
    return *new HostNVMatrix();
}
HIPmatrix& HostNVMatrix::construct(bool isTrans) const {
    return *new HostNVMatrix(isTrans);
}
HIPmatrix& HostNVMatrix::construct(int numRows, int numCols, bool isTrans) const {
    return *new HostNVMatrix(numRows, numCols, isTrans);
}
HIPmatrix& HostNVMatrix::construct(const Matrix& like, bool copy) const {
    return *new HostNVMatrix(like, copy);
}
HIPmatrix& HostNVMatrix::construct(const HIPmatrix& like, bool copy) const {
    return *new HostNVMatrix(like, copy);
}
HIPmatrix& HostNVMatrix::construct(const HIPmatrix& like) const {
    return *new HostNVMatrix(like);
}
HIPmatrix& HostNVMatrix::construct(const Matrix& like) const {
    return *new HostNVMatrix(like);
}
HIPmatrix& HostNVMatrix::construct(MemorySegment* mem, int numRows, int numCols, int stride, bool isTrans) const {
    return *new HostNVMatrix(mem, numRows, numCols, stride, isTrans);
}

void HostNVMatrix::copyFromHost(const Matrix& hostMatrix, bool resizeTarget, hipStream_t stream) {
    if (resizeTarget) {
        resize(hostMatrix);
    } else {
        assert(isSameDims(hostMatrix));
    }
    setTrans(hostMatrix.isTrans());
    if (getNumElements() > 0) {
        checkCudaErrors(cudaMemcpy2D(getDevData(), _stride * sizeof(float), hostMatrix.getData(),
                                     hostMatrix.getLeadingDim() * sizeof(float), getLeadingDim() * sizeof(float),
                                     getFollowingDim(), hipMemcpyHostToHost));
//        syncStream(stream);
    }
}

void HostNVMatrix::copyFromHost(const Matrix& hostMatrix, bool resizeTarget) {
    copyFromHost(hostMatrix, resizeTarget, 0);
}

void HostNVMatrix::copyFromHost(const Matrix& hostMatrix) {
    copyFromHost(hostMatrix, false, 0);
}

void HostNVMatrix::copyToHost(Matrix& hostMatrix, bool resizeTarget, hipStream_t stream) const {
    if (resizeTarget) {
        hostMatrix.resize(getNumRows(), getNumCols());
    } else {
        assert(isSameDims(hostMatrix));
    }
    hostMatrix.setTrans(_isTrans);
    if (getNumElements() > 0) {
        checkCudaErrors(cudaMemcpy2D(hostMatrix.getData(), hostMatrix.getLeadingDim() * sizeof(float),
                                     getDevData(), _stride * sizeof(float), getLeadingDim() * sizeof(float),
                                     getFollowingDim(), hipMemcpyHostToHost));
//        syncStream(stream);
    }
}

void HostNVMatrix::copyToHost(Matrix& hostMatrix, bool resizeTarget) const {
    copyToHost(hostMatrix, resizeTarget, 0);
}

void HostNVMatrix::copyToHost(Matrix& hostMatrix) const {
    copyToHost(hostMatrix, false, 0);
}

void HostNVMatrix::alloc(int numElements) {
//    checkCudaErrors(cudaHostAlloc(&_devData, numElements * sizeof(float), cudaHostAllocPortable));
    _memSegment = HOST_MEMORY_MANAGER::getInstance().malloc(numElements * sizeof(float));
//    _memSegment = FastHostMemoryManager::getInstance().malloc(numElements * sizeof(float));
}

void HostNVMatrix::dealloc() {
//    FastHostMemoryManager::getInstance().free(_memSegment);
    HOST_MEMORY_MANAGER::getInstance().free(_memSegment);
    _memSegment = NULL;
//    checkCudaErrors(hipFreeHost(_devData));
}

cudaTextureObject_t HostNVMatrix::getTextureObject() {
    assert(false);
    return 0;
}
