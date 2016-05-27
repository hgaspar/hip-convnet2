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

#ifndef NVMATRIX_H_
#define NVMATRIX_H_

#include <map>
#include <vector>
#include <hipblas.h>
#include <cuda.h>
#include <curand.h>
#include <time.h>
#include <curand_kernel.h>

#include <helper_cuda.h>
#include "../../util/include/matrix.h"
#include "hipmatrix_kernels.hpp"
#include "hipmatrix_operators.hpp"
#include "memory.hpp"

#ifdef WARNINGS
#define WARN(msg) printf("WARN: File %s, line %d: %s\n", __FILE__, __LINE__, msg);
#else
#define WARN(msg) ;
#endif

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
                            printf("CURAND Error at %s:%d\n",__FILE__,__LINE__);\
                            exit(EXIT_FAILURE);}} while(0)

#define CUBLAS_CALL(x) do { if((x) != HIPBLAS_STATUS_SUCCESS) { \
                            printf("CUBLAS Error at %s:%d\n",__FILE__,__LINE__);\
                            exit(EXIT_FAILURE);}} while(0)

/*
 * Memory manager to use for GPU memory allocations.
 *
 * CUDAMemoryManager: Default Nvidia memory manager; just calls hipMalloc / hipFree.
 *                    Allocating and freeing memory is slow.
 * FastMemoryManager: A GPU memory manager with very fast (constant time)
 *                    alloc / free, but possibly more wasteful of memory.
 */
#define DEVICE_MEMORY_MANAGER       CUDAMemoryManager

/*
 * Memory manager to use for host memory allocations.
 *
 * CUDAHostMemoryManager: Default Nvidia memory manager; just calls cudaHostAlloc / hipFreeHost.
 *                        Allocating and freeing memory is slow.
 * FastHostMemoryManager: A host memory manager with very fast (constant time)
 *                        alloc / free, but possibly more wasteful of memory.
 */
#define HOST_MEMORY_MANAGER         CUDAHostMemoryManager

class HIPMatrix;
typedef std::vector<HIPMatrix*> NVMatrixV;

class HIPMatrix {
protected:
    int _numCols, _numRows;
    int _numElements;
    int _stride;
//    float* getDevData();
    MemorySegment* _memSegment;
    bool _isTrans;
    bool _ownsData;
    // This flag makes sure that the HIPMatrix destructor does nothing
    // when called on HostNVMatrix instance.
    bool _deleted;
    cudaTextureObject_t _texObj;

//    static std::map<int,curandGenerator_t> rndGen;
    static std::map<int,MemorySegment*> _rndDevStates;
    static std::map<int,hipblasHandle_t> _cublasHandles;
    // Map from device id --> # of random streams initialized on that device
    static std::map<int,int> _rndDevThreads;
    static pthread_mutex_t *_rndMutex, *_cublasMutex, *_streamMutex;
    // Map from device id --> default stream
    static std::map<int,hipStream_t> _defaultStreams;

    hipblasOperation_t getTransChar() const {
        /*
         * not a typo! return opposite character because a
         * non-transposed hipmatrix is in row-major order while a non-transposed
         * cublas matrix is in column-major order.
         */
        return _isTrans ? HIPBLAS_OP_N : HIPBLAS_OP_T;
    }

    void _init(bool isTrans);
    void _sum_setParams(int n, dim3* blocks, dim3* threads);
    template<class Agg> float cpuAgg(Agg agg, hipStream_t stream);
    template<class Agg> float _totalAgg(Agg agg);
    template<class Agg> float _totalAgg(Agg agg, hipStream_t stream);
    template<class Agg> float _totalAgg(Agg agg, HIPMatrix& tmpbuf, hipStream_t stream);
    template<class Agg, class UnaryOp, class BinaryOp> void _aggregate(int axis, HIPMatrix& target, Agg agg, UnaryOp uop, BinaryOp bop, hipStream_t stream, HIPMatrix* tmp);
    template<class Agg, class UnaryOp, class BinaryOp> void _aggregate(int axis, HIPMatrix& target, Agg agg, UnaryOp uop, BinaryOp bop, hipStream_t stream);
    template<class Agg, class UnaryOp, class BinaryOp> void _aggregate(int axis, HIPMatrix& target, Agg agg, UnaryOp uop, BinaryOp bop);
    template<class Agg, class BinaryOp> void _aggregate(int axis, HIPMatrix& target, Agg agg, BinaryOp bop, hipStream_t stream);
    template<class Agg, class BinaryOp> void _aggregate(int axis, HIPMatrix& target, Agg agg, BinaryOp bop);
    template<class Agg, class BinaryOp> HIPMatrix& _aggregate(int axis, Agg agg, BinaryOp bop, hipStream_t stream);
    template<class Agg, class BinaryOp> HIPMatrix& _aggregate(int axis, Agg agg, BinaryOp bop);
    template<class Agg, class UnaryOp, class BinaryOp> HIPMatrix& _aggregate(int axis, Agg agg, UnaryOp, BinaryOp bop, hipStream_t stream);
    template<class Agg, class UnaryOp, class BinaryOp> HIPMatrix& _aggregate(int axis, Agg agg, UnaryOp, BinaryOp bop);

    template<class Agg, class UnaryOp, class BinaryOp> void _aggregate(int axis, HIPMatrix& target, Agg agg, UnaryOp uop, BinaryOp bop, HIPMatrix& tmp);
    template<class Agg, class BinaryOp> void _aggregate(int axis, HIPMatrix& target, Agg agg, BinaryOp bop, hipStream_t stream, HIPMatrix& tmp);
    template<class Agg, class BinaryOp> void _aggregate(int axis, HIPMatrix& target, Agg agg, BinaryOp bop, HIPMatrix& tmp);
    template<class Agg, class BinaryOp> HIPMatrix& _aggregate(int axis, Agg agg, BinaryOp bop, hipStream_t stream, HIPMatrix& tmp);
    template<class Agg, class BinaryOp> HIPMatrix& _aggregate(int axis, Agg agg, BinaryOp bop, HIPMatrix& tmp);
    template<class Agg, class UnaryOp, class BinaryOp> HIPMatrix& _aggregate(int axis, Agg agg, UnaryOp, BinaryOp bop, hipStream_t stream, HIPMatrix& tmp);
    template<class Agg, class UnaryOp, class BinaryOp> HIPMatrix& _aggregate(int axis, Agg agg, UnaryOp, BinaryOp bop, HIPMatrix& tmp);

    template <class Randomizer> void _unaryRandomize(HIPMatrix& target, Randomizer rnd, hipStream_t stream);
    template <class Randomizer> void _unaryRandomize(HIPMatrix& target, Randomizer rnd);
    template <class Randomizer> void _binaryRandomize(HIPMatrix& data2, HIPMatrix& target, Randomizer rnd);
    template <class Randomizer> void _binaryRandomize(HIPMatrix& data2, HIPMatrix& target, Randomizer rnd, hipStream_t stream);

    virtual void alloc(int numElements);
    virtual void dealloc();
    void deallocTexture();
    virtual HIPMatrix& construct() const;
    virtual HIPMatrix& construct(bool isTrans) const;
    virtual HIPMatrix& construct(int numRows, int numCols, bool isTrans=false) const;
    virtual HIPMatrix& construct(const Matrix& like, bool copy) const;
    virtual HIPMatrix& construct(const HIPMatrix& like, bool copy) const;
    virtual HIPMatrix& construct(const HIPMatrix& like) const;
    virtual HIPMatrix& construct(const Matrix& like) const;
    virtual HIPMatrix& construct(MemorySegment* mem, int numRows, int numCols, int stride, bool isTrans) const;
    static hipblasHandle_t getCublasHandle();
    static hipblasHandle_t getCublasHandle(int deviceID);
public:
    HIPMatrix();
    HIPMatrix(bool isTrans);
    HIPMatrix(int numRows, int numCols, bool isTrans=false);
    HIPMatrix(const Matrix& like, bool copy);
    HIPMatrix(const HIPMatrix& like, bool copy);
    HIPMatrix(const HIPMatrix& like);
    HIPMatrix(const Matrix& like);
    HIPMatrix(MemorySegment* mem, int numRows, int numCols, int stride, bool isTrans);
    virtual ~HIPMatrix();

    // Returns the device ID on which the data pointer is allocated
    int getDataDeviceID() const;
    static void initRandom(unsigned long long seed, int numStreams, hipStream_t stream);
    static void initRandom(unsigned long long seed, int numStreams);
    static void initRandom(unsigned long long seed);
    static void initRandom();
    static void initCublas();
    static void destroyCublas();
    static std::pair<size_t, size_t> getCudaMemorySize();

    // Returns the currently-active device ID for calling thread
    static int getDeviceID();
    static void setDeviceID(int d);
    static bool canAccessPeer(int srcDevice, int tgtDevice);
    static bool isRndInitialized();
    static bool isRndInitialized(bool haveLock);
    static curandState* getCurandState();
    static curandState* getCurandState(int numStreams);
    static void destroyRandom();
    static pthread_mutex_t* makeMutex();
    static hipStream_t getDefaultStream(int deviceID);
    static hipStream_t getDefaultStream();
    static void syncDevice();
    static void syncStream();
    static void syncStream(hipStream_t stream);

    /*
     * DO NOT DEREFERENCE IN HOST CODE! This is a device memory pointer.
     */
    float* getCellPtr(int i, int j) const {
        if (_isTrans) {
            return &getDevData()[j * _numRows + i];
        }
        return &getDevData()[i * _numCols + j];
    }

    bool isSameDims(const Matrix& m) const {
        return m.getNumRows() == _numRows && m.getNumCols() == _numCols;
    }

    bool isSameDims(const HIPMatrix& m) const {
        return m.getNumRows() == _numRows && m.getNumCols() == _numCols;
    }

    int getNumRows() const {
        return _numRows;
    }

    int getNumCols() const {
        return _numCols;
    }

    int getStride() const {
        return _stride;
    }

    int getLeadingDim() const {
        return _isTrans ? _numRows : _numCols;
    }

    int getFollowingDim() const {
        return !_isTrans ? _numRows : _numCols;
    }

    /*
     * FALSE:    Row-major order.
     * TRUE:     Column-major order.
     */
    bool isTrans() const {
        return _isTrans;
    }

    bool isView() const {
        return !_ownsData;
    }

    float* getDevData() const {
        return _memSegment == NULL ? NULL : _memSegment->getData<float>();
    }

    MemorySegment& getMemorySegment() const {
        return *_memSegment;
    }

    int getNumElements() const {
        return _numElements;
    }

    size_t getNumDataBytes() const {
        return size_t(_numElements) * 4;
    }

    /*
     * Only use if you know what you're doing!
     * Does not actually transpose matrix.
     */
    void setTrans(bool trans) {
        if (trans != _isTrans) {
            assert(isContiguous());
            _isTrans = trans;
            _stride = getLeadingDim();
        }
    }

    /*
     * Only use if you know what you're doing!
     * This toggles whether this object will free its GPU memory when it's destroyed.
     */
    void setIsView(bool isView) {
        _ownsData = !isView;
    }

    bool isContiguous() const {
        return _stride == getLeadingDim() || getFollowingDim() == 1;
    }

    void truncate() {
        resize(0,0);
    }

    virtual cudaTextureObject_t getTextureObject();

    virtual void copyFromHost(const Matrix& hostMatrix);
    virtual void copyFromHost(const Matrix& hostMatrix, bool resizeTarget);
    virtual void copyFromHost(const Matrix& hostMatrix, bool resizeTarget, hipStream_t stream);
    virtual void copyToHost(Matrix& hostMatrix) const;
    virtual void copyToHost(Matrix& hostMatrix, bool resizeTarget) const;
    virtual void copyToHost(Matrix& hostMatrix, bool resizeTarget, hipStream_t stream) const;
    void copy(HIPMatrix& dest) const;
    void copy(HIPMatrix& dest, hipStream_t stream) const;
    HIPMatrix& copy() const;
    void addProduct(HIPMatrix& a, HIPMatrix &b, float scaleThis, float scaleAB, hipStream_t stream);
    void addProduct(HIPMatrix& a, HIPMatrix &b, float scaleThis, float scaleAB);
    void addProduct(HIPMatrix& a, HIPMatrix &b);
    void rightMult(HIPMatrix &b, float scaleAB, HIPMatrix &target, hipStream_t stream);
    void rightMult(HIPMatrix &b, float scaleAB, HIPMatrix &target);
    void rightMult(HIPMatrix &b, HIPMatrix &target);
    void rightMult(HIPMatrix &b, float scaleAB);
    void randomizeUniform();
    void addGaussianNoise(HIPMatrix& stdevs, bool var, HIPMatrix& target);
    void addGaussianNoise(float stdev, HIPMatrix& target);
    void addGaussianNoise(HIPMatrix& stdevs, bool var);
    void addGaussianNoise(HIPMatrix& stdevs);
    void addGaussianNoise(float stdev);
    void addGaussianNoise();
    void randomizeGaussian();
    void randomizeGaussian(float stdev);
    void randomizeGaussian(float mean, float stdev);
    void randomizeGaussian(float mean, HIPMatrix& stdevs);
    void randomizeGaussian(float mean, float stdevMult, HIPMatrix& stdevs);
    void randomizeGaussian(HIPMatrix& stdevs);
    void randomizeGaussian(HIPMatrix& stdevs, HIPMatrix& target);
    void binarizeProbs();
    void binarizeProbs(HIPMatrix& target);

    void biggerThan(HIPMatrix& m, HIPMatrix& target);
    void biggerThan(HIPMatrix& m);
    void biggerThanVector(HIPMatrix& vec, HIPMatrix& target);
    void biggerThanVector(HIPMatrix& vec);
    void equals(HIPMatrix& m, HIPMatrix& target);
    void equals(HIPMatrix& m);

    void _checkBounds(int startRow, int endRow, int startCol, int endCol) const;
    HIPMatrix& slice(int startRow, int endRow, int startCol, int endCol) const;
    void slice(int startRow, int endRow, int startCol, int endCol, HIPMatrix& target) const;
    HIPMatrix& sliceRows(int startRow, int endRow) const;
    void sliceRows(int startRow, int endRow, HIPMatrix& target) const;
    HIPMatrix& sliceCols(int startCol, int endCol) const;
    void sliceCols(int startCol, int endCol, HIPMatrix& target) const;

    NVMatrixV& splitRows(int numParts);
    NVMatrixV& splitCols(int numParts);

    template <class Op> void apply(Op op, HIPMatrix& target, hipStream_t stream) {
        if (!target.isSameDims(*this)) {
            target.resize(*this);
        }
        if (getNumElements() > 0) {
            int height = target.getFollowingDim(), width = target.getLeadingDim();

            if (target.isTrans() == isTrans()) {
                if (!isContiguous() || !target.isContiguous()) {
                    dim3 blocks(std::min(NUM_BLOCKS_MAX, DIVUP(width, ELTWISE_THREADS_X)),
                            std::min(NUM_BLOCKS_MAX, DIVUP(height, ELTWISE_THREADS_Y)));
                    dim3 threads(ELTWISE_THREADS_X, ELTWISE_THREADS_Y);
                    hipLaunchKernel(HIP_KERNEL_NAME(kEltwiseUnaryOp<Op>), dim3(blocks), dim3(threads), 0, stream, getDevData(), target.getDevData(), height, width, getStride(), target.getStride(), op);
                    getLastCudaError("kEltwiseUnaryOp: Kernel execution failed");
                } else {
                    dim3 threads = dim3(ELTWISE_FLAT_THREADS_X);
                    dim3 blocks = dim3(std::min(128, DIVUP(_numElements, ELTWISE_FLAT_THREADS_X)));
                    hipLaunchKernel(HIP_KERNEL_NAME(kEltwiseUnaryOpFlat<Op>), dim3(blocks), dim3(threads), 0, stream, getDevData(), target.getDevData(), _numElements, op);
                    getLastCudaError("kEltwiseUnaryOpFlat: Kernel execution failed");
                }
            } else {
                dim3 blocks(std::min(NUM_BLOCKS_MAX, DIVUP(width, ELTWISE_THREADS_X)),
                        std::min(NUM_BLOCKS_MAX, DIVUP(height, ELTWISE_THREADS_Y)));
                dim3 threads(ELTWISE_THREADS_X, ELTWISE_THREADS_Y);
                bool checkBounds = !(width % ELTWISE_THREADS_X == 0 && height % ELTWISE_THREADS_X == 0);
    //            printf("height: %d, width: %d, stride: %d, target stride: %d, check bounds: %d, threads.x: %d, threads.y: %d, blocks.x: %d, blocks.y: %d\n",
    //                    height, width, getStride(), target.getStride(), checkBounds, threads.x, threads.y, blocks.x, blocks.y);
                if (checkBounds) {
                    hipLaunchKernel(HIP_KERNEL_NAME(kEltwiseUnaryOpTrans<Op, true>), dim3(blocks), dim3(threads), 0, stream, getDevData(), target.getDevData(), height, width, getStride(), target.getStride(), op);
                } else {
                    hipLaunchKernel(HIP_KERNEL_NAME(kEltwiseUnaryOpTrans<Op, false>), dim3(blocks), dim3(threads), 0, stream, getDevData(), target.getDevData(), height, width, getStride(), target.getStride(), op);
                }
                getLastCudaError("kEltwiseUnaryOpTrans: Kernel execution failed");
            }
        }
    }

    template <class Op> void apply(Op op, hipStream_t stream) {
        apply(op, *this, stream);
    }

    template <class Op> void apply(Op op, HIPMatrix& target) {
        apply(op, target, getDefaultStream());
    }

    template <class Op> void apply(Op op) {
        apply(op, *this);
    }

    template <class Op> void applyBinary(Op op, HIPMatrix& b) {
        applyBinary(op, b, *this);
    }

    template <class Op> void applyBinary(Op op, HIPMatrix& b, HIPMatrix& target) {
        applyBinary(op, b, target, getDefaultStream());
    }

    template <class Op> void applyBinary(Op op, HIPMatrix& b, HIPMatrix& target, hipStream_t stream) {
        assert(this->isSameDims(b));

        if (!target.isSameDims(*this)) {
            target.resize(*this);
        }

        if (getNumElements() > 0) {
            int height = target.getFollowingDim(), width = target.getLeadingDim();
            if (target.isTrans() == isTrans() && target.isTrans() == b.isTrans()) {
                if (!isContiguous() || !b.isContiguous() || !target.isContiguous()) {
                    dim3 blocks(std::min(128, DIVUP(width, ELTWISE_THREADS_X)),
                                std::min(128, DIVUP(height, ELTWISE_THREADS_Y)));
                    dim3 threads(ELTWISE_THREADS_X, ELTWISE_THREADS_Y);
                    hipLaunchKernel(HIP_KERNEL_NAME(kEltwiseBinaryOp<Op>), dim3(blocks), dim3(threads), 0, stream, getDevData(), b.getDevData(), target.getDevData(), height, width, getStride(),
                                                              b.getStride(), target.getStride(), op);
                } else {
                    dim3 threads = dim3(ELTWISE_FLAT_THREADS_X);
                    dim3 blocks = dim3(std::min(128, DIVUP(_numElements, ELTWISE_FLAT_THREADS_X)));
                    hipLaunchKernel(HIP_KERNEL_NAME(kEltwiseBinaryOpFlat<Op>), dim3(blocks), dim3(threads), 0, stream, getDevData(), b.getDevData(), target.getDevData(), _numElements, op);
                }
                getLastCudaError("kEltwiseBinaryOp: Kernel execution failed");
            } else {

                dim3 blocks(std::min(128, DIVUP(width, ELTWISE_THREADS_X)),
                            std::min(128, DIVUP(height, ELTWISE_THREADS_Y)));
                dim3 threads(ELTWISE_THREADS_X, ELTWISE_THREADS_Y);
                //  both x here since y divides x
                bool checkBounds = !(width % ELTWISE_THREADS_X == 0 && height % ELTWISE_THREADS_X == 0);
                if (target.isTrans() == isTrans() && target.isTrans() != b.isTrans()) {
                    if (checkBounds) {
                        hipLaunchKernel(HIP_KERNEL_NAME(kEltwiseBinaryOpTrans<Op,true,false,false>), dim3(blocks), dim3(threads), 0, stream, getDevData(), b.getDevData(), target.getDevData(), height, width,getStride(),
                                                                   b.getStride(), target.getStride(), op);
                    } else {
                        hipLaunchKernel(HIP_KERNEL_NAME(kEltwiseBinaryOpTrans<Op,false,false,false>), dim3(blocks), dim3(threads), 0, stream, getDevData(), b.getDevData(), target.getDevData(), height, width,getStride(),
                                                                   b.getStride(), target.getStride(), op);
                    }
                } else if (target.isTrans() != isTrans() && target.isTrans() != b.isTrans()) {
                    if (checkBounds) {
                        hipLaunchKernel(HIP_KERNEL_NAME(kEltwiseBinaryOpTrans<Op,true,true,false>), dim3(blocks), dim3(threads), 0, stream, getDevData(), b.getDevData(), target.getDevData(), height, width,getStride(),
                                                                   b.getStride(), target.getStride(), op);
                    } else {
                        hipLaunchKernel(HIP_KERNEL_NAME(kEltwiseBinaryOpTrans<Op,false,true,false>), dim3(blocks), dim3(threads), 0, stream, getDevData(), b.getDevData(), target.getDevData(), height, width,getStride(),
                                                                   b.getStride(), target.getStride(), op);
                    }
                } else if (target.isTrans() != isTrans() && target.isTrans() == b.isTrans()) {
                    if (checkBounds) {
                        hipLaunchKernel(HIP_KERNEL_NAME(kEltwiseBinaryOpTrans<Op,true,false,true>), dim3(blocks), dim3(threads), 0, stream, b.getDevData(), getDevData(), target.getDevData(), height, width,b.getStride(),
                                                                   getStride(), target.getStride(), op);
                    } else {
                        hipLaunchKernel(HIP_KERNEL_NAME(kEltwiseBinaryOpTrans<Op,false,false,true>), dim3(blocks), dim3(threads), 0, stream, b.getDevData(), getDevData(), target.getDevData(), height, width, b.getStride(),
                                                                   getStride(), target.getStride(), op);
                    }
                }
                getLastCudaError("kEltwiseBinaryOpTrans: Kernel execution failed");
            }
        }
    }

    template <class Op> void applyTernary(Op op, HIPMatrix& b, HIPMatrix& c, HIPMatrix& target) {
        applyTernary(op, b, c, target, getDefaultStream());
    }

    template <class Op> void applyTernary(Op op, HIPMatrix& b, HIPMatrix& c, HIPMatrix& target, hipStream_t stream) {
        assert(isSameDims(b));
        assert(isSameDims(c));
        // For now ternary ops are only supported for matrices of same transposedness
        assert(isTrans() == b.isTrans());
        assert(isTrans() == c.isTrans());
        if (!target.isSameDims(*this) || target.isTrans() != isTrans()) {
            target.resize(*this);
        }
        if (getNumElements() > 0) {
            int height = target.getFollowingDim(), width = target.getLeadingDim();
            if (!isContiguous() || !b.isContiguous() || !c.isContiguous() || !target.isContiguous()) {
                dim3 blocks(std::min(512, DIVUP(width, ELTWISE_THREADS_X)),
                            std::min(512, DIVUP(height, ELTWISE_THREADS_Y)));
                dim3 threads(ELTWISE_THREADS_X, ELTWISE_THREADS_Y);
                hipLaunchKernel(HIP_KERNEL_NAME(kEltwiseTernaryOp<Op>), dim3(blocks), dim3(threads), 0, stream, getDevData(), b.getDevData(), c.getDevData(), target.getDevData(), height, width,
                                                                       getStride(), b.getStride(), c.getStride(), target.getStride(), op);
                getLastCudaError("kEltwiseTernaryOp: Kernel execution failed");
            } else {
                dim3 threads = dim3(ELTWISE_FLAT_THREADS_X);
                dim3 blocks = dim3(std::min(128, DIVUP(_numElements, ELTWISE_FLAT_THREADS_X)));
                hipLaunchKernel(HIP_KERNEL_NAME(kEltwiseTernaryOpFlat<Op>), dim3(blocks), dim3(threads), 0, stream, getDevData(), b.getDevData(), c.getDevData(), target.getDevData(), _numElements, op);
                getLastCudaError("kEltwiseTernaryOpFlat: Kernel execution failed");
            }
        }
    }

    bool resize(int numRows, int numCols, bool trans);
    bool resize(int numRows, int numCols);
    bool resize(const HIPMatrix &like);
    bool resize(const Matrix &like);
    void reshape(int numRows, int numCols);
    HIPMatrix& reshaped(int numRows, int numCols) const;
    void copy(HIPMatrix &dest, int srcStartRow, int srcEndRow, int srcStartCol, int srcEndCol, int destStartRow, int destStartCol) const;
    void copy(HIPMatrix &dest, int srcStartRow, int srcEndRow, int srcStartCol, int srcEndCol, int destStartRow, int destStartCol, hipStream_t stream) const;
    void add(HIPMatrix& b, float scaleA, float scaleB, HIPMatrix& target, hipStream_t stream);
    void add(HIPMatrix& b, float scaleA, float scaleB, HIPMatrix& target);
    void add(HIPMatrix& b, float scaleB, HIPMatrix& target);
    void add(HIPMatrix& b, HIPMatrix& target);
    void add(HIPMatrix& b, float scaleB);
    void add(HIPMatrix& b, float scaleA, float scaleB);
    void add(HIPMatrix& b);
    void eltwiseMult(HIPMatrix& b);
    void eltwiseMult(HIPMatrix& b, HIPMatrix& target);
    void eltwiseDivide(HIPMatrix& b);
    void eltwiseDivide(HIPMatrix& b, HIPMatrix& target);
    void squaredDiff(HIPMatrix& b);
    void squaredDiff(HIPMatrix& b, HIPMatrix& target);
    void subtract(HIPMatrix& b, HIPMatrix& target);
    void subtract(HIPMatrix& b);
    void addVector(HIPMatrix& vec, float scaleVec, HIPMatrix& target, hipStream_t stream);
    void addVector(HIPMatrix& vec, float scaleVec, HIPMatrix& target);
    void addVector(HIPMatrix& vec);
    void addVector(HIPMatrix& vec, float scaleVec);
    void addVector(HIPMatrix& vec, HIPMatrix& target);
    void equalsVector(HIPMatrix& vec, HIPMatrix& target);
    void equalsVector(HIPMatrix& vec);
    void eltwiseMultByVector(HIPMatrix& vec, HIPMatrix& target, hipStream_t stream);
    void eltwiseMultByVector(HIPMatrix& vec, HIPMatrix& target);
    void eltwiseMultByVector(HIPMatrix& vec);
    void eltwiseMultByVector(HIPMatrix& vec, hipStream_t stream);
    void eltwiseDivideByVector(HIPMatrix& vec, HIPMatrix& target);
    void eltwiseDivideByVector(HIPMatrix& vec);
    void tile(int timesY, int timesX, HIPMatrix& target);
    void tile(int timesY, int timesX, HIPMatrix& target, hipStream_t stream);

    void addSum(HIPMatrix& a, int axis, float scaleThis, float scaleSum);
    void addSum(HIPMatrix& a, int axis, float scaleThis, float scaleSum, hipStream_t stream);
    void addMax(HIPMatrix& a, int axis, float scaleThis, float scaleMax);
    void addMax(HIPMatrix& a, int axis, float scaleThis, float scaleMax, hipStream_t stream);
    void sum(int axis, HIPMatrix& target, hipStream_t stream);
    void sum(int axis, HIPMatrix& target);
    void sum(int axis, HIPMatrix& target, hipStream_t stream, HIPMatrix& tmp);
    void sum(int axis, HIPMatrix& target, HIPMatrix& tmp);
    HIPMatrix& sum(int axis);
    void max(int axis, HIPMatrix& target);
    void max(int axis, HIPMatrix& target, HIPMatrix& tmp);
    HIPMatrix& max(int axis);
    void min(int axis, HIPMatrix& target);
    HIPMatrix& min(int axis);
    void sumOfSquares(int axis, HIPMatrix& target, hipStream_t stream);
    void sumOfSquares(int axis, HIPMatrix& target);
    HIPMatrix& sumOfSquares(int axis);
    float mean();
    float sum();
    float sum(HIPMatrix& tmpbuf);
    float max();
    float min();
    float countInf();
    float countNan();
    float norm2();
    float norm();

    void inRangeInc(float lower, float upper);
    void inRangeInc(float lower, float upper, HIPMatrix& target);
    void inRangeExc(float lower, float upper);
    void inRangeExc(float lower, float upper, HIPMatrix& target);
    void biggerThanScalar(float scalar);
    void biggerThanScalar(float scalar, HIPMatrix& target);
    void smallerThanScalar(float scalar);
    void smallerThanScalar(float scalar, HIPMatrix& target);
    void addScalar(float scaleThis, float scalar, HIPMatrix& target);
    void addScalar(float scalar, HIPMatrix& target);
    void addScalar(float scalar);
    void minWithScalar(float scalar, HIPMatrix& target);
    void minWithScalar(float scalar);
    void maxWithScalar(float scalar, HIPMatrix& target);
    void maxWithScalar(float scalar);
    void pow(float p, HIPMatrix& target);
    void pow(float p);
    void scale(float _scale);
    void scale(float _scale, HIPMatrix& target);
    void scale(float _scale, HIPMatrix& target, hipStream_t stream);
    void scale(float _scale, hipStream_t stream);
    void zero();
    void zero(HIPMatrix& like);

    float dotProduct(HIPMatrix& b, HIPMatrix& tmp, hipStream_t stream);
    float dotProduct(HIPMatrix& b, hipStream_t stream);
    float dotProduct(HIPMatrix& b);

    /*
     * Does SOFT transpose and returns result, leaving this matrix unchanged
     */
    HIPMatrix& getTranspose();
    HIPMatrix& getClone();

    /*
     * Does HARD transpose and puts result in target
     */
    void transpose(HIPMatrix& target);

    /*
     * Does SOFT transpose
     */
    void transpose();
    bool transpose(bool trans);

    void flipTrans(HIPMatrix& target, hipStream_t stream);
    void flipTrans(HIPMatrix& target);
    HIPMatrix& flipTrans();

    void print(int startRow, int rows, int startCol, int cols) const;
    void print(int rows, int cols) const;
    void printShape(const char* name) const;

    template <class Op> void applyBinaryV(Op op, HIPMatrix& vec, HIPMatrix& target) {
        applyBinaryV(op, vec, target, getDefaultStream());
    }

    template <class Op> void applyBinaryV(Op op, HIPMatrix& vec, HIPMatrix& target, hipStream_t stream) {
        assert(&target != &vec); // for now
        if (isSameDims(vec)) {
            applyBinary(op, vec, target, stream);
            return;
        }
        assert(vec.getNumRows() == 1 || vec.getNumCols() == 1);
        assert(vec.getNumRows() == _numRows || vec.getNumCols() == _numCols);
        assert(vec.isContiguous());

        target.resize(*this); // target must be same orientation as me for now
        int width = getLeadingDim(); //_isTrans ? _numRows : _numCols;
        int height = getFollowingDim(); //_isTrans ? _numCols : _numRows;
        dim3 threads(ADD_VEC_THREADS_X, ADD_VEC_THREADS_Y);

        if ((vec.getNumRows() == _numRows && !isTrans()) || (vec.getNumCols() == _numCols && isTrans())) {
            dim3 blocks(std::min(512, DIVUP(width, ADD_VEC_THREADS_X)), std::min(NUM_BLOCKS_MAX, DIVUP(height, ADD_VEC_THREADS_Y)));
            hipLaunchKernel(HIP_KERNEL_NAME(kColVectorOp<Op>), dim3(blocks), dim3(threads), 0, stream, getDevData(), vec.getDevData(), target.getDevData(), width, height, getStride(), target.getStride(), op);
        } else {
            dim3 blocks(std::min(NUM_BLOCKS_MAX, DIVUP(width, ADD_VEC_THREADS_X)), std::min(NUM_BLOCKS_MAX, DIVUP(height, ADD_VEC_THREADS_Y)));
            hipLaunchKernel(HIP_KERNEL_NAME(kRowVectorOp<Op>), dim3(blocks), dim3(threads), 0, stream, getDevData(), vec.getDevData(), target.getDevData(), width, height, getStride(), target.getStride(), op);
        }
        getLastCudaError("Kernel execution failed");
    //    hipDeviceSynchronize();
    }

    template<class UnaryOperator> float argMax(UnaryOperator u) {
       return _totalAgg(NVMatrixAggs::ArgMax<UnaryOperator>(u));
    }
    static void batchedMatrixMultiply(NVMatrixV& a, NVMatrixV& b, NVMatrixV& target, float scaleTarget, float scaleAB, hipStream_t stream, const float** aPtrsDev, const float** bPtrsDev, float** tgtPtrsDev);
    static void batchedMatrixMultiply(NVMatrixV& a, NVMatrixV& b, NVMatrixV& target, float scaleTarget, float scaleAB, hipStream_t stream);
    static void batchedMatrixMultiply(NVMatrixV& a, NVMatrixV& b, NVMatrixV& target, float scaleTarget, float scaleAB, const float** aPtrsDev, const float** bPtrsDev, float** tgtPtrsDev);
    static void batchedMatrixMultiply(NVMatrixV& a, NVMatrixV& b, NVMatrixV& target, float scaleTarget, float scaleAB);

    static void assertSame(NVMatrixV& a);
};

class HostNVMatrix : public HIPMatrix {
protected:
    void alloc(int numElements);
    void dealloc();
    HIPMatrix& construct() const;
    HIPMatrix& construct(bool isTrans) const;
    HIPMatrix& construct(int numRows, int numCols, bool isTrans=false) const;
    HIPMatrix& construct(const Matrix& like, bool copy) const;
    HIPMatrix& construct(const HIPMatrix& like, bool copy) const;
    HIPMatrix& construct(const HIPMatrix& like) const;
    HIPMatrix& construct(const Matrix& like) const;
    HIPMatrix& construct(MemorySegment* mem, int numRows, int numCols, int stride, bool isTrans) const;
public:
    ~HostNVMatrix();
    HostNVMatrix();
    HostNVMatrix(bool isTrans);
    HostNVMatrix(int numRows, int numCols, bool isTrans=false);
    HostNVMatrix(const Matrix& like, bool copy);
    HostNVMatrix(const HIPMatrix& like, bool copy);
    HostNVMatrix(const HIPMatrix& like);
    HostNVMatrix(const Matrix& like);
    HostNVMatrix(MemorySegment* mem, int numRows, int numCols, int stride, bool isTrans);
    void copyFromHost(const Matrix& hostMatrix);
    void copyFromHost(const Matrix& hostMatrix, bool resizeTarget);
    void copyFromHost(const Matrix& hostMatrix, bool resizeTarget, hipStream_t stream);
    void copyToHost(Matrix& hostMatrix) const;
    void copyToHost(Matrix& hostMatrix, bool resizeTarget) const;
    void copyToHost(Matrix& hostMatrix, bool resizeTarget, hipStream_t stream) const;
    cudaTextureObject_t getTextureObject();
};

#endif /* NVMATRIX_H_ */
