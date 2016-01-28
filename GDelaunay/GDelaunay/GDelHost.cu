/*
Author: Ashwin Nanjappa
Filename: GDelHost.cu

Copyright (c) 2013, School of Computing, National University of Singapore. 
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution. 

Neither the name of the National University of Singapore nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission from the National University of Singapore. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES 
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.

*/

////////////////////////////////////////////////////////////////////////////////
//                               Star Host Code
////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////// Headers //

// Project
#include "Config.h"
#include "Pba.h"
#include "Geometry.h"
#include "PerfTimer.h"
#include "GDelInternal.h"
#include "GDelKernels.h"

using namespace std;

const int ThreadsPerBlock   = MAX_THREADS_PER_BLOCK;
const int BlocksPerGrid     = 512;
const int ThreadNum         = ThreadsPerBlock * BlocksPerGrid;

const int PredThreadsPerBlock   = MAX_PRED_THREADS_PER_BLOCK;
const int PredBlocksPerGrid     = 16;
const int PredThreadNum         = PredThreadsPerBlock * PredBlocksPerGrid;

const RealType DataExpansionFactor = 1.3f;

// Do NOT set these here!
// These are set by application.
bool LogVerbose = false;
bool LogStats   = false;
bool LogTiming  = false;
bool LogMemory  = false;

HostTimer containerTimer;

inline void cuPrintMemory( const string& inStr )
{
    size_t free;
    size_t total;
    const int MegaByte = ( 1 << 20 );
    CudaSafeCall( cudaMemGetInfo( &free, &total ) );
    cout << "[" << inStr << "] Memory used (MB): " << ( total - free ) / MegaByte << endl;
    return;
}

///////////////////////////////////////////////////////////////// Memory Pool //

int PoolBlockSize = -1; 

thrust::host_vector< void * > PoolMemory; 

void PoolInit( int blockSize ) 
{
    PoolBlockSize = blockSize; 
    return;
}

void PoolDeinit() 
{
    if ( LogVerbose )
        cout << "Peak pool size: " << PoolMemory.size() << endl; 

    for ( int i = 0; i < ( int ) PoolMemory.size(); ++i ) 
        CudaSafeCall( cudaFree( PoolMemory[i] ) ); 

    PoolMemory.clear(); 

    PoolBlockSize = -1; 

    return;
}

///////////////////////////////////////////////////////////// DeviceContainer //

template< typename T > 
class DeviceContainer
{
private:
    thrust::device_ptr< T > _ptr; 
    int                     _size; 
    int                     _capacity; 

public: 
    typedef thrust::device_ptr< T > iterator; 

    DeviceContainer( ) : _size( 0 ), _capacity( 0 ) { }; 
    
    DeviceContainer( int n ) : _size( 0 ), _capacity( 0 )
    {
        resize( n ); 
        return;
    }

    DeviceContainer( int n, T value ) : _size( 0 ), _capacity( 0 )
    {
        resize( n, value );
        return;
    }

    ~DeviceContainer()
    {
        free();
        return;
    }

    void free() 
    {
        if ( _capacity > 0 )
        {
            if ( _capacity * sizeof(T) == PoolBlockSize ) 
                PoolMemory.push_back( _ptr.get() ); 
            else 
                CudaSafeCall( cudaFree( _ptr.get() ) ); 
        }

        _size       = 0; 
        _capacity   = 0; 

        return;
    }

    const thrust::device_ptr< T >& get_device_ptr() const
    {
        return _ptr; 
    }

    void resize( int n )
    {
        if ( _capacity >= n )
        {
            _size = n; 
            return;
        }

        free(); 

        _size       = n; 
        _capacity   = ( n == 0 ) ? 1 : n; 

        if ( ( _capacity * sizeof( T ) == PoolBlockSize ) &&
            PoolMemory.size() > 0 ) 
        {
            _ptr = thrust::device_ptr< T >( ( T* ) PoolMemory.back() );
            PoolMemory.pop_back(); 
        }
        else
        {
            if ( LogMemory )
            {
                containerTimer.start(); 
            }

            _ptr = thrust::device_malloc< T >( _capacity );

            if ( LogMemory )
            {
                containerTimer.stop(); 
                cout << "DeviceContainer allocated," << _capacity * sizeof( T ) << ", Time," << containerTimer.value() << endl; 
                cuPrintMemory( "" );
            }
        }

        return;
    }

    void resize( int n, const T& value )
    {
        resize( n ); 
        thrust::fill_n( begin(), n, value );
        return;
    }

    int size() const { return _size; }

    thrust::device_reference< T > operator[] ( const int index ) const
    {
        return _ptr[ index ]; 
    }

    const iterator begin() const { return _ptr; }

    const iterator end() const { return _ptr + _size; }

    void erase( const iterator& first, const iterator& last )
    {
        if ( last == end() )
        {
            _size -= (last - first);
        }
        else
        {
            assert( false && "Not supported right now!" );
        }

        return;
    }

    void swap( DeviceContainer< T >& arr ) 
    {
        int tempSize    = _size; 
        int tempCap     = _capacity; 
        T* tempPtr      = ( _capacity > 0 ) ? _ptr.get() : 0; 

        _size       = arr._size; 
        _capacity   = arr._capacity; 

        if ( _capacity > 0 ) 
            _ptr = thrust::device_ptr< T >( arr._ptr.get() ); 

        arr._size       = tempSize; 
        arr._capacity   = tempCap; 

        if ( tempCap >= 0 )
            arr._ptr = thrust::device_ptr< T >( tempPtr );

        return;
    }
    
    void swapAndFree( DeviceContainer< T >& inArr )
    {
        swap( inArr );
        inArr.free();
        return;
    }

    void copyFrom( const DeviceContainer< T >& inArr )
    {
        resize( inArr._size );
        thrust::copy( inArr._ptr, inArr._ptr + inArr._size, _ptr );
        return;
    }

    void assign( const int n, const T& value ) 
    {
        thrust::fill_n( _ptr, n, value );
        return;
    }

    void copyToHost( thrust::host_vector< T >& dest )
    {
        dest.insert( dest.begin(), begin(), end() );
        return;
    }

    DeviceContainer& operator=( DeviceContainer& src )
    {
        resize( src._size ); 

        if ( src._size > 0 ) 
            thrust::copy( src.begin(), src.end(), begin() ); 

        return *this; 
    }
}; 

// DeviceContainer to kernelArray converter
template < typename T >
KerArray< T > toKernelArray( DeviceContainer< T >& dVec )
{
    KerArray< T > tArray;
    tArray._arr = thrust::raw_pointer_cast( &dVec[0] );
    tArray._num = ( int ) dVec.size();

    return tArray;
}

// We move all triangle arrays EXCEPT triStar, which should already be updated!
template< typename T >
__global__ void kerMoveTriangleArray
(
int             oldTriNum,
KerIntArray     oldNewMap,
KerArray< T >   oldArr,
KerArray< T >   newArr
)
{
    // Iterate through triangles
    for ( int oldTriIdx = getCurThreadIdx(); oldTriIdx < oldTriNum; oldTriIdx += getThreadNum() )
    {
        // Skip free triangles

        const int newTriIdx = oldNewMap._arr[ oldTriIdx ];

        if ( -1 == newTriIdx )
            continue;

        // Copy old to new 
        newArr._arr[ newTriIdx ] = oldArr._arr[ oldTriIdx ];
    }

    return;
}

// Delete DeviceContainer pointer safely
template < typename T >
void safeDeleteDevConPtr( T** ptr )
{
    if ( *ptr )
    {
        delete *ptr;
        *ptr = NULL;
    }

    CudaCheckError();

    return;
}

/////////////////////////////////////////////////////////////////////// Types //

typedef DeviceContainer< short >            ShortDVec;
typedef DeviceContainer< int >              IntDVec;
typedef DeviceContainer< Triangle >         TriDVec;
typedef DeviceContainer< TriangleOpp >      TriOppDVec;
typedef DeviceContainer< TriOppTetra >      LocOppTetraDVec;
typedef DeviceContainer< TriangleStatus >   TriStatusDVec;
typedef DeviceContainer< Segment >          SegmentDVec;
typedef DeviceContainer< Tetrahedron >      TetraDVec;
typedef DeviceContainer< TriPosition >      TriPositionDVec;
typedef DeviceContainer< DeathCert >        DeathCertDVec;

typedef thrust::host_vector< TriangleStatus > TriStatusHVec;
typedef thrust::host_vector< DeathCert >      DeathCertHVec;

typedef IntDVec::iterator                       IntDIter;
typedef thrust::tuple< int, int >               IntTuple2;
typedef thrust::tuple< IntDIter, IntDIter >     IntDIterTuple2;
typedef thrust::zip_iterator< IntDIterTuple2 >  ZipDIter;

struct PointData
{
    Point3DVec* _pointVec;
    WeightDVec* _weightVec;
    int         _bitsPerIndex;

    void init( const Point3HVec& pointHVec, const WeightHVec& weightHVec )
    {
        _pointVec       = new Point3DVec( pointHVec );
        _weightVec      = new WeightDVec( weightHVec );
        _bitsPerIndex   = ( int ) ceil( log( ( double ) _pointVec->size() ) / log( 2.0 ) ); 

        return;
    }

    void deinit()
    {
        safeDeleteDevConPtr( &_pointVec );
        safeDeleteDevConPtr( &_weightVec );

        _bitsPerIndex = -1;

        return;
    }

    KerPointData toKernel()
    {
        KerPointData pData;

        pData._pointArr     = thrust::raw_pointer_cast( &(*_pointVec)[0] );
        pData._weightArr    = thrust::raw_pointer_cast( &(*_weightVec)[0] );
        pData._num          = ( int ) _pointVec->size();

        return pData;
    }
};

struct TriangleData
{
    TriDVec*            _triVec[2];         // Triangles of stars
    TriOppDVec*         _triOppVec[2];
    IntDVec*            _triStarVec[2];     // Star containing triangle
    TriStatusDVec*      _triStatusVec[2];   // Triangle status
    LocOppTetraDVec*    _triOppTetra[2]; 

    void init()
    {
        for ( int i = 0; i < 2; ++i )
        {
            _triVec[i]          = new TriDVec();
            _triOppVec[i]       = new TriOppDVec();
            _triStarVec[i]      = new IntDVec();
            _triStatusVec[i]    = new TriStatusDVec();
            _triOppTetra[i]     = new LocOppTetraDVec();
        }

        return;
    }

    void deinit()
    {
        for ( int i = 0; i < 2; ++i )
        {
            safeDeleteDevConPtr( &_triVec[i] );
            safeDeleteDevConPtr( &_triOppVec[i] );
            safeDeleteDevConPtr( &_triStarVec[i] );
            safeDeleteDevConPtr( &_triStatusVec[i] );
            safeDeleteDevConPtr( &_triOppTetra[i] );
        }

        return;
    }

    void resize( int newSize, int arrId, const TriangleStatus& triStatus ) 
    {
        _triVec[ arrId ]->resize( newSize ); 
        _triOppVec[ arrId ]->resize( newSize ); 
        _triStarVec[ arrId ]->resize( newSize ); 
        _triStatusVec[ arrId ]->resize( newSize, triStatus ); 
        _triOppTetra[ arrId ]->resize( newSize ); 

        return;
    }

    int size( int vecId ) const
    {
        return _triVec[ vecId ]->size();
    }

    int totalSize() const
    {
        return size( 0 ) + size( 1 );
    }

    KerTriangleData toKernel()
    {
        KerTriangleData kData;

        for ( int i = 0; i < 2; ++i )
        {
            kData._triArr[i]        = thrust::raw_pointer_cast( &(*_triVec[i])[0] );
            kData._triOppArr[i]     = thrust::raw_pointer_cast( &(*_triOppVec[i])[0] );
            kData._triStarArr[i]    = thrust::raw_pointer_cast( &(*_triStarVec[i])[0] );
            kData._triStatusArr[i]  = thrust::raw_pointer_cast( &(*_triStatusVec[i])[0] );
            kData._triOppTetra[i]   = thrust::raw_pointer_cast( &(*_triOppTetra[i])[0] );
        }

        return kData;
    }
};

struct StarData
{
    int                 _starNum;       // Number of stars
    TriangleData        _triData; 
    IntDVec*            _starTriMap[2]; // Index into triangle array for each star
    IntDVec*            _pointNumVec;   // Number of points in each star
    TriPositionDVec*    _insStatusVec;  // Status of each star
    IntDVec*            _insCountVec;   // Number of insertions of each star
    DeathCertDVec*      _deathCertVec;
    IntDVec*            _flagVec;

    void init( int pointNum )
    {
        // Preallocate these per-star vectors
        _starTriMap[0]  = new IntDVec( pointNum );
        _starTriMap[1]  = new IntDVec( pointNum );
        _pointNumVec    = new IntDVec( pointNum, 0 );
        _insStatusVec   = new TriPositionDVec( pointNum );
        _insCountVec    = new IntDVec( pointNum );
        _deathCertVec   = new DeathCertDVec( pointNum );
        _flagVec        = new IntDVec( FlagNum );

        _triData.init();

        return;
    }

    void deInit()
    {
        _triData.deinit(); 

        safeDeleteDevConPtr( &_starTriMap[0] );
        safeDeleteDevConPtr( &_starTriMap[1] );
        safeDeleteDevConPtr( &_pointNumVec );
        safeDeleteDevConPtr( &_insStatusVec );
        safeDeleteDevConPtr( &_insCountVec );
        safeDeleteDevConPtr( &_deathCertVec );
        safeDeleteDevConPtr( &_flagVec );

        return;
    }

    KerStarData toKernel()
    {
        KerStarData sData;

        sData._starNum = _starNum;

        for ( int i = 0; i < 2; ++i )
        {
            sData._triNum[i]        = ( int ) _triData.size( i );
            sData._triArr[i]        = thrust::raw_pointer_cast( &(*_triData._triVec[i])[0] );
            sData._triOppArr[i]     = thrust::raw_pointer_cast( &(*_triData._triOppVec[i])[0] );
            sData._triOppTetra[i]   = thrust::raw_pointer_cast( &(*_triData._triOppTetra[i])[0] );
            sData._triStarArr[i]    = thrust::raw_pointer_cast( &(*_triData._triStarVec[i])[0] );
            sData._triStatusArr[i]  = thrust::raw_pointer_cast( &(*_triData._triStatusVec[i])[0] );
            sData._starTriMap[i]    = thrust::raw_pointer_cast( &(*_starTriMap[i])[0] );
        }

        sData._totalTriNum  = sData._triNum[0] + sData._triNum[1];
        sData._pointNumArr  = thrust::raw_pointer_cast( &(*_pointNumVec)[0] );
        sData._insStatusArr = thrust::raw_pointer_cast( &(*_insStatusVec)[0] );
        sData._insCountArr  = thrust::raw_pointer_cast( &(*_insCountVec)[0] );
        sData._deathCertArr = thrust::raw_pointer_cast( &(*_deathCertVec)[0] );
        sData._flagArr      = thrust::raw_pointer_cast( &(*_flagVec)[0] );

        return sData;
    }

    // Expands input vector to hold old data
    template< typename T >
    void expandData( int oldSize, int newSize, IntDVec& oldNewMap, DeviceContainer< T >& inVec )
    {
        DeviceContainer< T > tmpVec( newSize ); 

        if ( oldSize > 0 ) 
        {
            kerMoveTriangleArray<<< BlocksPerGrid, ThreadsPerBlock >>>( oldSize, toKernelArray( oldNewMap ), toKernelArray( inVec ), toKernelArray( tmpVec ) );
            CudaCheckError();
        }

        inVec.swap( tmpVec );

        return;
    }

    void expandTriangles( int newSize, IntDVec& newTriMap )
    {
        //cout << __FUNCTION__ << endl;

        const int oldSize = ( int ) _triData.size( 1 );    // Grab old size before it is replaced

        ////
        // Create old-to-new triangle index map
        // *and* also update triStar and triStatus
        ////

        IntDVec newStarVec( newSize );
        TriStatusDVec newStatusVec( newSize, Free );
        IntDVec oldNewMap( oldSize, -1 );

        if ( oldSize > 0 ) 
        {
            kerMakeOldNewTriMap<<< BlocksPerGrid, ThreadsPerBlock >>>(
                toKernel(),
                oldSize,
                toKernelArray( newTriMap ),
                toKernelArray( oldNewMap ),
                toKernelArray( newStarVec ),
                toKernelArray( newStatusVec ) );
            CudaCheckError();
        }

        _starTriMap[1]->swap( newTriMap );
        _triData._triStarVec[1]->swapAndFree( newStarVec );
        _triData._triStatusVec[1]->swapAndFree( newStatusVec );

        // Move rest of triangle arrays
        expandData( oldSize, newSize, oldNewMap, *_triData._triVec[1] ); 
        expandData( oldSize, newSize, oldNewMap, *_triData._triOppVec[1] ); 
        expandData( oldSize, newSize, oldNewMap, *_triData._triOppTetra[1] ); 

        return;
    }
};

struct FacetData
{
    IntDVec*        _fromStarVec;
    IntDVec*        _vertStarVec;
    IntDVec*        _fromTriVec;
    SegmentDVec*    _segVec;
    IntDVec*        _insertMapVec;
    int             _drownedFacetNum; 

    void init()
    {
        _fromStarVec    = new IntDVec();
        _vertStarVec    = new IntDVec();
        _fromTriVec     = new IntDVec();
        _segVec         = new SegmentDVec();
        _insertMapVec   = new IntDVec();

        return;
    }

    void deInit()
    {
        safeDeleteDevConPtr( &_fromStarVec );
        safeDeleteDevConPtr( &_vertStarVec );
        safeDeleteDevConPtr( &_fromTriVec );
        safeDeleteDevConPtr( &_segVec );
        safeDeleteDevConPtr( &_insertMapVec );

        return;
    }

    KerFacetData toKernel()
    {
        KerFacetData fData;
        fData._fromStarArr  = thrust::raw_pointer_cast( &(*_fromStarVec)[0] );
        fData._toStarArr    = thrust::raw_pointer_cast( &(*_vertStarVec)[0] );
        fData._fromTriArr   = thrust::raw_pointer_cast( &(*_fromTriVec)[0] );
        fData._segArr       = thrust::raw_pointer_cast( &(*_segVec)[0] );
        fData._insertMapArr = thrust::raw_pointer_cast( &(*_insertMapVec)[0] );
        fData._num          = ( int ) _fromStarVec->size();

        fData._drownedFacetNum = _drownedFacetNum; 

        return fData;
    }
};

struct InsertionData
{
    IntDVec*    _vertVec;
    IntDVec*    _vertStarVec;
    IntDVec*    _starVertMap;

    void init()
    {
        _vertVec        = new IntDVec();
        _vertStarVec    = new IntDVec();
        _starVertMap    = new IntDVec();

        return;
    }

    void deInit() 
    {
        safeDeleteDevConPtr( &_vertVec );
        safeDeleteDevConPtr( &_vertStarVec );
        safeDeleteDevConPtr( &_starVertMap );

        return;
    }

    KerInsertionData toKernel()
    {
        KerInsertionData iData;

        iData._vertArr      = thrust::raw_pointer_cast( &(*_vertVec)[0] );
        iData._vertStarArr  = thrust::raw_pointer_cast( &(*_vertStarVec)[0] );
        iData._starVertMap  = thrust::raw_pointer_cast( &(*_starVertMap)[0] );

        iData._vertNum  = ( int ) _vertVec->size();
        iData._starNum  = ( int ) _starVertMap->size();

        return iData;
    }
};

struct TetraData
{
    TetraDVec* _vec;

    void init()
    {
        _vec = new TetraDVec();
        return;
    }

    void deInit() 
    {
        safeDeleteDevConPtr( &_vec );
        return;
    }

    KerTetraData toKernel()
    {
        KerTetraData tData;
        tData._arr      = thrust::raw_pointer_cast( &(*_vec)[0] );
        tData._num      = ( int ) _vec->size();

        return tData;
    }
};

typedef IntHVec LoopStatHVec;

///////////////////////////////////////////////////////////////////// Globals //

int GridWidth   = 0;
int* DGrid      = NULL;

PredicateInfo   DPredicateInfo;
PointData       DPointData;
FacetData       DFacetData;
InsertionData   DInsertData;
StarData        DStarData;
TetraData       DTetraData;
IntDVec*        DTriBufVec;
TetraHVec       HTetraVec;

int WorksetSizeMax  = -1;
int InsertPointMax  = -1;
int LoopNum         = -1;
int FacetMax        = -1;

// Stats
int InsertSum           = 0;
int StitchSum           = 0;
int PointInconsCount    = 0;
int PointDrownedCount   = 0;
int FacetInconsCount    = 0;
//AllStatHVec StatVec;
IntHVec StarLoopVec;

// Timing
HostTimer loopTimer;
HostTimer insertTimer;
HostTimer expandTimer;
HostTimer facetTimer;
HostTimer sortTimer;
double expandTime   = 0;
double facetTime    = 0;
double insertTime   = 0;
double sortTime     = 0;

// Debugging
int DeadStarNum = 0;

////////////////////////////////////////////////////////////////////////////////

// Replace input vector with its map and also calculate the sum of input vector
// Input:  [ 4 2 0 5 ]
// Output: [ 0 4 6 6 ] Sum: 11
int makeInPlaceMapAndSum( IntDVec& inVec )
{
    const int lastValue = inVec[ inVec.size() - 1 ]; 

    // Make map
    thrust::exclusive_scan( inVec.begin(), inVec.end(), inVec.begin() );

    // Sum
    const int sum = inVec[ inVec.size() - 1 ] + lastValue; 

    return sum;
}


// Convert input vector to map and also calculate the sum of input array
// Input:  [ 4 2 0 5 ]
// Output: [ 0 4 6 6 ] Sum: 11
template< typename T >
int makeMapAndSum( const DeviceContainer< T >& inVec, IntDVec& mapVec )
{
    // Resize map vector
    mapVec.resize( inVec.size() );

    // Make map
    thrust::exclusive_scan( inVec.begin(), inVec.end(), mapVec.begin() );

    // Sum
    const int sum = inVec[ inVec.size() - 1 ] + mapVec[ mapVec.size() - 1 ];

    return sum;
}

// Given an input list of sorted stars (with duplicates and missing stars)
// creates a map for all stars
void compactAndMakeAllStarMap( IntDVec& inVec, IntDVec& mapVec, int starNum )
{
    // Expand map to input vector size
    mapVec.resize( starNum, -1 );

    kerMakeAllStarMap<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelArray( inVec ),
        toKernelArray( mapVec ),
        starNum );
    CudaCheckError();

    return;
}

struct isNegative
{
    __host__ __device__ bool operator() ( const int x )
    {
        return ( x < 0 );
    }
};

struct isZero
{
    __host__ __device__ bool operator() ( const int x )
    {
        return ( x == 0 );
    }
};

// Check if second value in tuple2 is negative
struct isTuple2Negative
{
    __host__ __device__ bool operator() ( const IntTuple2& tup )
    {
        const int y = thrust::get<1>( tup );
        return ( y < 0 );
    }
};

// Remove negative elements in input vector
template < typename T >
int compactIfNegative( DeviceContainer< T >& inVec )
{
    inVec.erase(    thrust::remove_if( inVec.begin(), inVec.end(), isNegative() ),
                    inVec.end() );

    return ( int ) inVec.size();
}

// Remove from first vector when element in second vector is negative
template < typename T >
int compactIfNegative( DeviceContainer< T >& inVec, const IntDVec& checkVec )
{
    assert( ( inVec.size() == checkVec.size() ) && "Vectors should be equal size!" );

    inVec.erase(    thrust::remove_if( inVec.begin(), inVec.end(), checkVec.begin(), isNegative() ),
                    inVec.end() );

    return ( int ) inVec.size();
}

template < typename T >
int compactIfZero( DeviceContainer< T >& inVec, const IntDVec& checkVec )
{
    assert( ( inVec.size() == checkVec.size() ) && "Vectors should be equal size!" );

    inVec.erase(    thrust::remove_if( inVec.begin(), inVec.end(), checkVec.begin(), isZero() ),
                    inVec.end() );

    return ( int ) inVec.size();
}

// Remove from *both* input vectors when element in second vector is negative
void compactBothIfNegative( IntDVec& vec0, IntDVec& vec1 )
{
    assert( ( vec0.size() == vec1.size() ) && "Vectors should be equal size!" );

    ZipDIter newEnd = thrust::remove_if(    thrust::make_zip_iterator( thrust::make_tuple( vec0.begin(), vec1.begin() ) ),
                                            thrust::make_zip_iterator( thrust::make_tuple( vec0.end(), vec1.end() ) ),
                                            isTuple2Negative() );

    IntDIterTuple2 endTuple = newEnd.get_iterator_tuple();

    vec0.erase( thrust::get<0>( endTuple ), vec0.end() );
    vec1.erase( thrust::get<1>( endTuple ), vec1.end() );

    return;
}

// Remove duplicates from the input key-value vector pair
void makePairVectorUnique( IntDVec& vec0, IntDVec& vec1 )
{
    assert( ( vec0.size() == vec1.size() ) && "Invalid size vectors!" );

    kerMarkDuplicates<<< BlocksPerGrid, ThreadsPerBlock >>>( toKernelArray( vec0 ), toKernelArray( vec1 ) ); 
    CudaCheckError();

    compactBothIfNegative( vec0, vec1 ); 

    return;
}

////////////////////////////////////////////////////////////////////////////////

void initPredicate()
{
    DPredicateInfo.init();

    // Predicate constants
    DPredicateInfo._consts = cuNew< RealType >( DPredicateVarNum );

    // Predicate arrays
    DPredicateInfo._data = cuNew< RealType >( PredicateDataTotalSize * PredThreadNum );

    // Set predicate constants
    kerInitPredicate<<< 1, 1 >>>( DPredicateInfo._consts );
    CudaCheckError();

    return;
}

Point3DVec* starsInit
(
const Config&       config,
const Point3HVec&   pointHVec,
const WeightHVec&   weightHVec
)
{
    GridWidth   = config._gridSize;
    FacetMax    = config._facetMax;
    LogVerbose  = config._logVerbose;
    LogStats    = config._logStats;
    LogTiming   = config._logTiming;

    initPredicate();

    // Set cache configuration
    CudaSafeCall( cudaFuncSetCacheConfig( kerCheckStarConsistency, cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerCheckValidTetras, cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerCountPointsOfStar, cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerFindDrownedFacetInsertionsExact, cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerFindDrownedFacetInsertionsFast, cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerGetPerTriangleInsertion, cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerGrabTetrasFromStars, cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerMarkBeneathTrianglesExact, cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerMarkBeneathTrianglesFast, cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerMakeInitialConeFast, cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerMakeInitialConeExact, cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerMarkOwnedTriangles, cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerGetFacetInsertCount, cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerReadGridPairs, cudaFuncCachePreferL1 ) ); 
    CudaSafeCall( cudaFuncSetCacheConfig( kerSetBeneathToFree, cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerStitchPointToHole, cudaFuncCachePreferL1 ) ); 

    // Move points to device first
    DPointData.init( pointHVec, weightHVec );

    PoolInit( DPointData._pointVec->size() * sizeof( int ) ); 

    DTetraData.init();
    DStarData.init( DPointData._pointVec->size() );
    DInsertData.init();

    DTriBufVec = new IntDVec();

    DeadStarNum = 0;
    
    // Used by PBA
    return DPointData._pointVec;
}

void starsDeinit()
{
    DFacetData.deInit();
    DInsertData.deInit();
    DPointData.deinit();
    DPredicateInfo.deInit();
    DStarData.deInit();
    DTetraData.deInit();

    safeDeleteDevConPtr( &DTriBufVec );
    
    PoolDeinit(); 

    return;
}

////////////////////////////////////////////////////////////////////////////////
//                         makeStarsFromGrid()
////////////////////////////////////////////////////////////////////////////////

void _readGridPairs()
{
    if ( LogVerbose )
    {
        cout << endl << __FUNCTION__ << endl;
    }

    ////
    // Count pairs
    ////

    const int BlocksPerGrid     = GridWidth + 2; 
    const int ThreadsPerBlock   = GridWidth; 
    const int ThreadNum         = BlocksPerGrid * ThreadsPerBlock; 

    DInsertData._vertVec->resize( ThreadNum ); 

    // Get per-thread pair count
    kerReadGridPairs<<< BlocksPerGrid, ThreadsPerBlock >>>( DGrid, GridWidth, DInsertData.toKernel(), CountPerThreadPairs );
    CudaCheckError();

    // Convert count to per-thread map
    const int worksetNum = makeMapAndSum( *DInsertData._vertVec, *DInsertData._starVertMap );

    ////
    // Grab pairs
    ////

    // Prepare workset array
    DInsertData._vertVec->resize( worksetNum );
    DInsertData._vertStarVec->resize( worksetNum );

    if ( LogVerbose )
    {
        cout << "Workset pairs: " << DInsertData._vertVec->size() << endl;
    }

    // Read pairs from grid
    kerReadGridPairs<<< BlocksPerGrid, ThreadsPerBlock >>>( DGrid, GridWidth, DInsertData.toKernel(), GrabPerThreadPairs );
    CudaCheckError();

    return;
}

void _removeWorksetDuplicates()
{
    if ( LogVerbose )
    {
        cout << endl << __FUNCTION__ << endl;
    }

    kerAppendValueToKey<<< BlocksPerGrid, ThreadsPerBlock >>>( DInsertData.toKernel(), 31 - DPointData._bitsPerIndex ); 

    const int vertSize = DInsertData._vertStarVec->size();

    if ( LogVerbose )
    {
        cuPrintMemory( "sort_by_key" );
    }

    thrust::sort_by_key( DInsertData._vertStarVec->begin(), DInsertData._vertStarVec->end(), DInsertData._vertVec->begin() );

    // Remove duplicates from both vectors
    makePairVectorUnique( *DInsertData._vertStarVec, *DInsertData._vertVec );

    kerRemoveValueFromKey<<< BlocksPerGrid, ThreadsPerBlock >>>( DInsertData.toKernel(), 31 - DPointData._bitsPerIndex ); 
    CudaCheckError();

    if ( LogVerbose )
    {
        cout << "Workset pairs: " << DInsertData._vertVec->size() << endl;
    }

    // Make star-vert map for all stars
    compactAndMakeAllStarMap( *DInsertData._vertStarVec, *DInsertData._starVertMap, DPointData._pointVec->size() ); 

    // Stars of initial insertions not needed
    DInsertData._vertStarVec->free();

    return;
}

// Pump working sets with <4 points.
// Since working sets for each star are derived from tetra, the smallest working
// set can be 3.
// 1. Find working sets with 3 points
// 2. Allocate more memory for these working sets
// 3. Copy over old working sets to new AND add new point to size-3 sets
void _pumpWorksets()
{
    if ( LogVerbose )
    {
        cout << endl << __FUNCTION__ << endl;
    }

    ////
    // Look for starving worksets
    ////

    // Prepare mark vector
    IntDVec markVec( DInsertData._starVertMap->size() );

    // Prepare workset size vector
    DStarData._insCountVec->resize( DInsertData._starVertMap->size() );

    // Mark starving working sets
    kerMarkStarvingWorksets<<< BlocksPerGrid, ThreadsPerBlock >>>(
        DInsertData.toKernel(),
        toKernelArray( markVec ),
        toKernelArray( *DStarData._insCountVec ) );
    CudaCheckError();

    // Note size of largest workset (needed later)
    WorksetSizeMax = *( thrust::max_element( DStarData._insCountVec->begin(), DStarData._insCountVec->end() ) );

    if ( LogVerbose )
    {
        cout << "Largest working set: " << WorksetSizeMax << endl;
    }

    // Prepare new vertex vector
    IntDVec newMap; 

    const int newWorksetSize = makeMapAndSum( *DStarData._insCountVec, newMap ); 

    ////
    // Pump starving worksets, if any
    ////

    if ( newWorksetSize > DInsertData._vertVec->size() )
    {
        IntDVec vertVec( newWorksetSize );

        // Pump worksets
        kerPumpWorksets<<< BlocksPerGrid, ThreadsPerBlock >>>(
            DInsertData.toKernel(),
            toKernelArray( newMap ),
            toKernelArray( vertVec ) );
        CudaCheckError();

        // Copy new vert and map arrays back
        DInsertData._vertVec->swap( vertVec );
        DInsertData._starVertMap->swap( newMap );

        if ( LogVerbose )
            cout << "Workset pairs after pumping: " << DInsertData._vertVec->size() << endl;
    }

    return;
}

void _makeWorksetForMissingPoints()
{
    if ( LogVerbose )
        cout << endl << __FUNCTION__ << endl;

    // Prepare vectors
    IntDVec fromStarVec( DInsertData._starVertMap->size() );
    IntDVec newWsetSizeVec( DInsertData._starVertMap->size() );

    // Calculate insertions required
    kerMissingPointWorksetSize<<< BlocksPerGrid, ThreadsPerBlock >>>(
        DPointData.toKernel(),
        DInsertData.toKernel(),
        DGrid,
        GridWidth,
        toKernelArray( fromStarVec ),
        toKernelArray( newWsetSizeVec ) );
    CudaCheckError();

    // Output Voronoi diagram is no longer useful, so free it
    CudaSafeCall( cudaFree( DGrid ) );

    ////
    // Prepare star arrays
    ////

    // Note: Triangle arrays are updated later
    DStarData._starNum  = ( int ) DInsertData._starVertMap->size();
    DStarData._starTriMap[0]->resize( DStarData._starNum );
    DStarData._starTriMap[1]->resize( DStarData._starNum );
    DStarData._pointNumVec->resize( DStarData._starNum, 0 );
    DStarData._insStatusVec->resize( DStarData._starNum );
    DStarData._insCountVec->copyFrom( newWsetSizeVec ); // Transfer workset number as insertion point number to stars
    const DeathCert cert = { Alive, Alive, Alive, Alive };
    DStarData._deathCertVec->resize( DStarData._starNum, cert );

    // Convert insertions to map
    IntDVec newMap;
    const int newVertNum = makeMapAndSum( newWsetSizeVec, newMap );
    newWsetSizeVec.free();

    // Prepare new vertex vector
    IntDVec newVertVec( newVertNum );

    // Copy over worksets for missing points
    kerMakeMissingPointWorkset<<< BlocksPerGrid, ThreadsPerBlock >>>(
        DInsertData.toKernel(),
        toKernelArray( fromStarVec ),
        toKernelArray( newMap ),
        toKernelArray( newVertVec ) );
    CudaCheckError();

    ////
    // Update workset with new map and verts
    ////

    DInsertData._starVertMap->swap( newMap );
    DInsertData._vertVec->swap( newVertVec );

    if ( LogVerbose )
    {
        cout << "Workset pairs: " << DInsertData._vertVec->size() << endl;
        cout << "Average workset per star: " << DInsertData._vertVec->size() / DInsertData._starVertMap->size() << endl;
    }

    return;
}

// Create 4-simplex for every star
void _makeInitialCones()
{
    if ( LogVerbose )
        cout << endl << __FUNCTION__ << endl;

    ////
    // Prepare star arrays
    ////

    const int triNum = get2SphereTriangleNum( DStarData._starNum, DInsertData._vertVec->size() );
    DStarData._triData.resize( triNum, 0, Free );   // Allocate only array[0] in the beginning
    DStarData._flagVec->resize( FlagNum, 0 );

    // Buffer to reuse for triangle related arrays
    const int expTriNum = ( int ) ( triNum * DataExpansionFactor );
    DTriBufVec->resize( expTriNum );

    if ( LogStats )
        cout << "Allocating triangles: " << triNum << endl;

    ////
    // Create initial 4-simplex for each star
    ////

    // Handle exact check triangles
    IntDVec exactVertVec( DStarData._starNum );

    kerMakeInitialConeFast<<< BlocksPerGrid, ThreadsPerBlock >>>(
        DPredicateInfo,
        DPointData.toKernel(),
        DStarData.toKernel(),
        DInsertData.toKernel(),
        toKernelArray( exactVertVec )
        );
    CudaCheckError();

    kerMakeInitialConeExact<<< PredBlocksPerGrid, PredThreadsPerBlock >>>(
        DPredicateInfo,
        DPointData.toKernel(),
        DStarData.toKernel(),
        DInsertData.toKernel(),
        toKernelArray( exactVertVec )
        ); 
    CudaCheckError(); 

    return;
}

void _debugCountDeath()
{
    DeathCertHVec   deathVec;
    IntHVec         deadStarVec;

    DStarData._deathCertVec->copyToHost( deathVec );

    const int starNum = ( int ) deathVec.size();

    for ( int si = 0; si < starNum; ++si )
    {
        const DeathCert& deathCert = deathVec[ si ];
        if ( Alive != getDeathStatus( deathCert ) )
            deadStarVec.push_back( si );
    }

    const int curDeadNum = ( int ) deadStarVec.size();

    assert( curDeadNum >= DeadStarNum );

    if ( curDeadNum > DeadStarNum )
    {
        cout << "Dead stars: " << deadStarVec.size() << endl;
        cout << "Dead in this insertion: " << curDeadNum - DeadStarNum << endl;
        DeadStarNum = curDeadNum;
    }

    return;
}

void _makeStars()
{
    if ( LogVerbose )
        cout << endl << __FUNCTION__ << endl;

    ////
    // Prepare arrays
    ////

    // Resize vertStarVec to the same size as vertVec
    // Initialized to -1 to differentiate from stars that will die during insertion
    DInsertData._vertStarVec->resize( DInsertData._vertVec->size(), -1 );

    // Handle exact check triangles
    TriPositionDVec exactTriVec( ExactTriangleMax );
    DStarData._flagVec->assign( DStarData._flagVec->size(), 0 );

    ////
    // Prepare to work only on active triangles
    ////

    IntDVec activeStarVec( DStarData._starNum );
    IntDVec activeTriCountVec( DStarData._starNum ); 

    thrust::sequence( activeStarVec.begin(), activeStarVec.end() );
    activeTriCountVec.copyFrom( *DStarData._starTriMap[0] ); 

    const int activeTriNum = DStarData._triData.totalSize(); 

    DTriBufVec->resize( activeTriNum ); 
    IntDVec& activeTriVec = *DTriBufVec;
    ShortDVec triInsNumVec( activeTriNum );

    kerGetPerTriangleInsertion<<<  BlocksPerGrid, ThreadsPerBlock >>>(
        DStarData.toKernel(),
        DInsertData.toKernel(),
        toKernelArray( activeStarVec ), 
        toKernelArray( activeTriCountVec ), 
        toKernelArray( activeTriVec ), 
        toKernelArray( triInsNumVec ) );
    CudaCheckError();

    ////
    // Insert workset points into stars
    ////

    for ( int idx = 4; idx < WorksetSizeMax; ++idx )
    {
        DStarData._insStatusVec->assign( DStarData._insStatusVec->size(), -1 );

        kerMarkBeneathTrianglesFast<<< BlocksPerGrid, ThreadsPerBlock >>>(
            DPredicateInfo,
            DPointData.toKernel(),
            DStarData.toKernel(),
            DInsertData.toKernel(),
            toKernelArray( activeTriVec ), 
            toKernelArray( triInsNumVec ),
            toKernelArray( exactTriVec ),
            idx );
        CudaCheckError(); 

        kerMarkBeneathTrianglesExact<<< PredBlocksPerGrid, PredThreadsPerBlock >>>(
            DPredicateInfo,
            DPointData.toKernel(),
            DStarData.toKernel(),
            DInsertData.toKernel(),
            toKernelArray( activeTriVec ), 
            toKernelArray( triInsNumVec ),
            toKernelArray( exactTriVec ),
            idx );
        CudaCheckError();

        kerStitchPointToHole<<< BlocksPerGrid, ThreadsPerBlock >>>(
            DStarData.toKernel(),
            DInsertData.toKernel(),
            toKernelArray( activeStarVec ),
            idx );
        CudaCheckError(); 
    }

    kerSetBeneathToFree<<< BlocksPerGrid, ThreadsPerBlock >>>(
        DStarData.toKernel(),
        toKernelArray( activeTriVec ), 
        toKernelArray( triInsNumVec ) );
    CudaCheckError();

    kerCountPointsOfStar<<< BlocksPerGrid, ThreadsPerBlock >>>( DStarData.toKernel() ); 
    CudaCheckError(); 

    if ( LogStats )
    {
        _debugCountDeath();
    }

    return;
}

// Initialize link vertices of each star from tetra.
void makeStarsFromGrid( int* grid )
{
    DGrid = grid;

    _readGridPairs();
    _removeWorksetDuplicates();
    _pumpWorksets();
    _makeWorksetForMissingPoints();
    _makeInitialCones();
    _makeStars();

    return;
}

////////////////////////////////////////////////////////////////////////////////
//                            processFacets()
////////////////////////////////////////////////////////////////////////////////

// Clear out the drowned-dead points
void _clearDrownedPoints()
{
    DInsertData._vertStarVec->resize( 0 );
    DInsertData._vertVec->resize( 0 );

    return;
}

int __makeFacetList()
{
    if ( LogVerbose )
        cout << endl << __FUNCTION__ << endl;

    ////
    // Count drowned facets and affected facets
    ////

    IntDVec& facetMap = *DTriBufVec;
    facetMap.resize( DStarData._triData.totalSize() );

    kerGetValidFacetCount<<< BlocksPerGrid, ThreadsPerBlock >>>(
        DStarData.toKernel(),
        toKernelArray( facetMap ) );
    CudaCheckError();

    // vertStarVec entries not -1 are drowned, collect them
    compactBothIfNegative( *DInsertData._vertVec, *DInsertData._vertStarVec );

    DFacetData._drownedFacetNum = DInsertData._vertStarVec->size();

    ////
    // Get facet map
    ////

    int facetNum    = DFacetData._drownedFacetNum + makeInPlaceMapAndSum( facetMap ); 
    facetNum        = max( min( facetNum, FacetMax ), DFacetData._drownedFacetNum );

    // Make the number of non-drowned facet divisable by 3,
    // so each link triangle can either generate 0 or all (3) facets. 
    facetNum = DFacetData._drownedFacetNum + ( ( facetNum - DFacetData._drownedFacetNum + 2 ) / 3 ) * 3; 

    if ( LogVerbose )
    {
        cout << "Facet number: " << facetNum << endl;
        cout << "Drowned: " << DFacetData._drownedFacetNum << endl; 
    }

    // Check if star splaying is over
    if ( 0 == facetNum )
        return 0;

    /////
    // Prepare facet arrays
    ////

    DFacetData._fromStarVec->resize( facetNum );
    DFacetData._vertStarVec->resize( facetNum );
    DFacetData._fromTriVec->resize( facetNum );
    DFacetData._segVec->resize( facetNum );
    DFacetData._insertMapVec->resize( facetNum );

    kerMakeFacetFromDrownedPoint<<< BlocksPerGrid, ThreadsPerBlock >>>(
        DStarData.toKernel(),
        DInsertData.toKernel(),
        DFacetData.toKernel() );
    CudaCheckError();

    kerGetValidFacets<<< BlocksPerGrid, ThreadsPerBlock >>>(
        DStarData.toKernel(),
        toKernelArray( facetMap ),
        DFacetData.toKernel() );
    CudaCheckError();

    ////
    // Sort facet list by to-star to speed up facets processing
    ////

    if ( facetNum > ThreadNum )
        thrust::sort_by_key(    DFacetData._vertStarVec->begin(),  DFacetData._vertStarVec->end(),
                                thrust::make_zip_iterator( make_tuple(  DFacetData._fromStarVec->begin(),
                                                                        DFacetData._fromTriVec->begin(),
                                                                        DFacetData._segVec->begin() ) ) );

    return facetNum;
}

int _makeFacetList()
{
    if ( LogTiming )
    {
        facetTimer.start();
    }

    const int facetNum = __makeFacetList();

    if ( LogTiming )
    {
        facetTimer.stop();
        facetTime += facetTimer.value();
    }

    return facetNum;
}

// Find number of insertions required per facet
int __makePerFacetInsertCount()
{
    if ( LogVerbose )
    {
        cout << __FUNCTION__ << endl;
    }

    if ( LogStats )
    {
        IntHVec fromVec;
        DFacetData._fromStarVec->copyToHost( fromVec );

        int certCount = 0;

        for ( int i = 0; i < ( int ) fromVec.size(); ++i )
        {
            if ( fromVec[i] < 0 )
                ++certCount;
        }

        if ( certCount > 0 )
            cout << "Dead stars needing certificate: " << certCount << endl;
    }

    kerGetFacetInsertCount<<< BlocksPerGrid, ThreadsPerBlock >>>(
        DStarData.toKernel(),
        DFacetData.toKernel() );
    CudaCheckError();

    if ( LogStats )
    {
        IntHVec toVec;
        DFacetData._vertStarVec->copyToHost( toVec );

        int deadFacetCount = 0;

        for ( int i = DFacetData._drownedFacetNum; i < ( int ) toVec.size(); ++i )
        {
            if ( toVec[i] < 0 )
                ++deadFacetCount;
        }

        if ( deadFacetCount > 0 )
            cout << "Facets with dead to-verts: " << deadFacetCount << endl;
    }

    // Find sum of insertions and update insertion map
    const int insertNum = makeInPlaceMapAndSum( *DFacetData._insertMapVec );

    if ( LogVerbose )
    {
        cout << "Insertion points: " << insertNum << endl;
    }

    kerGetDeathCertificateFast<<< BlocksPerGrid, ThreadsPerBlock >>>(
        DPredicateInfo,
        DPointData.toKernel(),
        DStarData.toKernel(),
        DFacetData.toKernel() );
    CudaCheckError();

    kerGetDeathCertificateExact<<< PredBlocksPerGrid, PredThreadsPerBlock >>>(
        DPredicateInfo,
        DPointData.toKernel(),
        DStarData.toKernel(),
        DFacetData.toKernel() );
    CudaCheckError();

    return insertNum;
}

int _makePerFacetInsertCount()
{
    if ( LogTiming )
    {
        facetTimer.start();
    }

    const int insertCount = __makePerFacetInsertCount();

    if ( LogTiming )
    {
        facetTimer.stop();
        facetTime += facetTimer.value();
    }

    return insertCount;
}

// Find the insertion points for each facet inconsistency
void __makeInsertPointList( int insertNum )
{
    if ( LogVerbose )
        cout << endl << __FUNCTION__ << endl;

    ////
    // Find insertion points for facets
    ////

    // Prepare insertion array
    DInsertData._vertStarVec->resize( insertNum );
    DInsertData._vertVec->resize( insertNum );

    kerFindDrownedFacetInsertionsFast<<< BlocksPerGrid, ThreadsPerBlock >>>(
        DPredicateInfo,
        DPointData.toKernel(),
        DStarData.toKernel(),
        DFacetData.toKernel(),
        DInsertData.toKernel() );
    CudaCheckError();

    kerFindDrownedFacetInsertionsExact<<< PredBlocksPerGrid, PredThreadsPerBlock >>>(
        DPredicateInfo,
        DPointData.toKernel(),
        DStarData.toKernel(),
        DFacetData.toKernel(),
        DInsertData.toKernel() );
    CudaCheckError();

    kerGetPerFacetInsertions<<< BlocksPerGrid, ThreadsPerBlock >>>(
        DStarData.toKernel(),
        DFacetData.toKernel(),
        DInsertData.toKernel() );
    CudaCheckError();

    kerCheckCertInsertions<<< BlocksPerGrid, ThreadsPerBlock >>>(
        DStarData.toKernel(),
        DInsertData.toKernel() );
    CudaCheckError();

    return;
}

void _makeInsertPointList( int insertCount )
{
    if ( LogTiming )
    {
        facetTimer.start();
    }

    __makeInsertPointList( insertCount );

    if ( LogTiming )
    {
        facetTimer.stop();
        facetTime += facetTimer.value();
    }

    return;
}

void __sortInsertPointList()
{
    if ( LogVerbose )
        cout << endl << __FUNCTION__ << endl;

    // Remove the -ve insertions, caused by death certificate insertions
    compactBothIfNegative( *DInsertData._vertStarVec, *DInsertData._vertVec );

    kerAppendValueToKey<<< BlocksPerGrid, ThreadsPerBlock >>>( DInsertData.toKernel(), 31 - DPointData._bitsPerIndex );
    CudaCheckError();

    thrust::sort_by_key( DInsertData._vertStarVec->begin(), DInsertData._vertStarVec->end(), DInsertData._vertVec->begin() );

    // Remove duplicates from both vectors
    makePairVectorUnique( *DInsertData._vertStarVec, *DInsertData._vertVec );

    kerRemoveValueFromKey<<< BlocksPerGrid, ThreadsPerBlock >>>( DInsertData.toKernel(), 31 - DPointData._bitsPerIndex ); 
    CudaCheckError();

    if ( LogVerbose )
        cout << "Insertion points after dup removal: " << DInsertData._vertVec->size() << endl;

    ////
    // Make map
    ////

    compactAndMakeAllStarMap( *DInsertData._vertStarVec, *DInsertData._starVertMap, DStarData._starNum );

    ////
    // Find insertion point count for each star
    ////

    // Go back and update the point count of each star (current count + intended insertion count)
    kerCountPerStarInsertions<<< BlocksPerGrid, ThreadsPerBlock >>>(
        DStarData.toKernel(),
        DInsertData.toKernel() );
    CudaCheckError();

    ////
    // Find largest insertion point count
    ////

    InsertPointMax = *( thrust::max_element( DStarData._insCountVec->begin(), DStarData._insCountVec->end() ) );

    if ( LogVerbose )
        cout << "Largest insertion set: " << InsertPointMax << endl;

    if ( LogStats )
        InsertSum += ( int ) DInsertData._vertVec->size();

    return;
}

void _sortInsertPointList()
{
    if ( LogTiming )
        sortTimer.start();

    __sortInsertPointList();

    if ( LogTiming )
    {
        sortTimer.stop();
        sortTime += sortTimer.value();
    }

    return;
}

void __expandStarsForInsertion()
{
    if ( LogVerbose )
        cout << endl << __FUNCTION__ << endl;

    ////
    // Calculate triangle/segment count for insertion
    ////

    // Prepare triangle count array
    IntDVec dTriNumVec( DStarData._starNum );

    // Estimate triangles needed for insertion
    kerComputeTriangleCount<<< BlocksPerGrid, ThreadsPerBlock >>>(
        DStarData.toKernel(),
        toKernelArray( dTriNumVec ) );
    CudaCheckError();

    ////
    // Expand triangles *only* if needed
    ////

    // Compute new triangles map and sum
    IntDVec newTriMap;
    const int newTriNum = makeMapAndSum( dTriNumVec, newTriMap );
    dTriNumVec.free();

    // Check if triangle array 2 needs to be expanded

    const int curTriNum = ( int ) DStarData._triData.size( 1 );

    if ( curTriNum < newTriNum )
    {
        if ( LogStats )
            cout << "Expanding triangles From: " << curTriNum << " To: " << newTriNum << endl;

        DStarData.expandTriangles( newTriNum, newTriMap );
    }

    return;
}

void _expandStarsForInsertion()
{
    if ( LogTiming )
    {
        expandTimer.start();
    }

    __expandStarsForInsertion();

    if ( LogTiming )
    {
        expandTimer.stop();
        expandTime += expandTimer.value();
    }

    return;
}

void __insertPointsToStars()
{
    if ( LogVerbose )
        cout << __FUNCTION__ << endl;

    // Handle exact check triangles
    IntDVec exactTriVec( ExactTriangleMax );

    ////
    // Prepare to work only on active triangles
    // "Active" triangles/stars are those that have some insertion
    ////

    IntDVec activeStarVec( DStarData._starNum );
    IntDVec activeTriCountVec( DStarData._starNum, 0 ); 

    // Find active stars
    thrust::sequence( activeStarVec.begin(), activeStarVec.end() );
    compactIfZero( activeStarVec, *DStarData._insCountVec );

    // Find triangle count for each active star
    kerGetActiveTriCount<<< BlocksPerGrid, ThreadsPerBlock >>>( 
        DStarData.toKernel(), 
        toKernelArray( activeStarVec ),
        toKernelArray( activeTriCountVec ) ); 
    CudaCheckError();

    // Number of active triangles and get triangle map
    const int activeTriNum = makeInPlaceMapAndSum( activeTriCountVec );

    // Store triangle index of active triangles for reuse
    IntDVec& activeTriVec = *DTriBufVec;
    activeTriVec.resize( activeTriNum );

    ShortDVec triInsNumVec( activeTriNum );

    // Find insertion count per triangle of active stars
    kerGetPerTriangleInsertion<<<  BlocksPerGrid, ThreadsPerBlock >>>(
        DStarData.toKernel(),
        DInsertData.toKernel(),
        toKernelArray( activeStarVec ), 
        toKernelArray( activeTriCountVec ), 
        toKernelArray( activeTriVec ), 
        toKernelArray( triInsNumVec ) );
    CudaCheckError();

    DStarData._flagVec->assign( DStarData._flagVec->size(), 0 );

    for ( int idx = 0; idx < InsertPointMax; ++idx )
    {
        DStarData._insStatusVec->assign( DStarData._insStatusVec->size(), -1 );

        kerMarkBeneathTrianglesFast<<< BlocksPerGrid, ThreadsPerBlock >>>(
            DPredicateInfo,
            DPointData.toKernel(),
            DStarData.toKernel(),
            DInsertData.toKernel(),
            toKernelArray( activeTriVec ), 
            toKernelArray( triInsNumVec ),
            toKernelArray( exactTriVec ),
            idx );
        CudaCheckError();

        kerMarkBeneathTrianglesExact<<< PredBlocksPerGrid, PredThreadsPerBlock >>>(
            DPredicateInfo,
            DPointData.toKernel(),
            DStarData.toKernel(),
            DInsertData.toKernel(),
            toKernelArray( activeTriVec ), 
            toKernelArray( triInsNumVec ),
            toKernelArray( exactTriVec ),
            idx );
        CudaCheckError();

        kerStitchPointToHole<<< BlocksPerGrid, ThreadsPerBlock >>>(
            DStarData.toKernel(),
            DInsertData.toKernel(),
            toKernelArray( activeStarVec ),
            idx );
        CudaCheckError(); 
    }

    kerSetBeneathToFree<<< BlocksPerGrid, ThreadsPerBlock >>>(
        DStarData.toKernel(),
        toKernelArray( activeTriVec ), 
        toKernelArray( triInsNumVec ) );
    CudaCheckError();

    kerCountPointsOfStar<<< BlocksPerGrid, ThreadsPerBlock >>>( DStarData.toKernel() ); 
    CudaCheckError(); 

    if ( LogStats )
        _debugCountDeath();

    return;
}

void _insertPointsToStars()
{
    if ( LogTiming )
        insertTimer.start();

    __insertPointsToStars();

    if ( LogTiming )
    {
        insertTimer.stop();
        insertTime += insertTimer.value();
    }

    return;
}

bool areStarsConsistent()
{
    if ( LogVerbose )
    {
        cout << endl << __FUNCTION__ << endl;
    }

    // Clear the flags
    DStarData._flagVec->assign( DStarData._flagVec->size(), 0 );
    
    kerInvalidateFreeTriangles<<< BlocksPerGrid, ThreadsPerBlock >>>( DStarData.toKernel() );
    CudaCheckError();

    kerCheckStarConsistency<<< BlocksPerGrid, ThreadsPerBlock >>>( DStarData.toKernel() );
    CudaCheckError();

    return ( 0 == (*DStarData._flagVec)[ ExactTriNum ] );
}

void processFacets()
{
    DFacetData.init(); 

    LoopNum = 0;

    int insertCount = -1; 

    do
    {
        if ( LogVerbose )
            cout << endl << "Loop: " << LoopNum << endl;

        if ( LogTiming )
            loopTimer.start();

        const int facetNum  = _makeFacetList();
        insertCount         = ( facetNum > 0 ) ? _makePerFacetInsertCount() : -1;

        if ( insertCount <= 0 ) 
        {
            if ( areStarsConsistent() )
            {
                break; 
            }
            else
            {
                ++LoopNum;  // Begin another loop

                _clearDrownedPoints();  // Drowned-dead points have already been handled

                if ( LogVerbose )
                {
                    cout << "Stars are not consistent! Repeating loop!" << endl;
                }

                continue; 
            }
        }

        _makeInsertPointList( insertCount );
        _sortInsertPointList();
        _expandStarsForInsertion();
        _insertPointsToStars();

        if ( LogTiming )
        {
            loopTimer.stop();
            loopTimer.print( "Loop" );
        }

        ++LoopNum;

    } while ( true );

    if ( LogTiming )
    {
        cout << "Expand time: " << expandTime << endl;
        cout << "Facet time: " << facetTime << endl;
        cout << "Sort time: " << sortTime << endl;
        cout << "Insert time: " << insertTime << endl;
    }

    DFacetData.deInit();
    DInsertData.deInit();

    return;
}

////////////////////////////////////////////////////////////////////////////////
//                           makeTetraFromStars()
////////////////////////////////////////////////////////////////////////////////

void _getTetraFromStars()
{
    // Prepare
    IntDVec& dTetraTriMapVec = *DTriBufVec;
    dTetraTriMapVec.resize( DStarData._triData.totalSize(), -1 );

    // Mark triangles owned by star
    kerMarkOwnedTriangles<<< BlocksPerGrid, ThreadsPerBlock >>>(
        DStarData.toKernel(),
        toKernelArray( dTetraTriMapVec ) );
    CudaCheckError();

    // Create map from tetra to triangles
    const int tetraNum = compactIfNegative( dTetraTriMapVec );

    if ( LogVerbose )
    {
        cout << "Tetra number: " << tetraNum << endl;
    }

    // Memory for tetra
    IntDVec validVec( tetraNum, 0 );
    IntDVec dTriTetraMapVec( DStarData._triData.totalSize(), -1 );

    // Check tetra orientations and map tri to tetra
    kerCheckValidTetras<<< BlocksPerGrid, ThreadsPerBlock >>>(
        DPredicateInfo,
        DPointData.toKernel(),
        DStarData.toKernel(),
        toKernelArray( validVec ),
        toKernelArray( dTetraTriMapVec ),
        toKernelArray( dTriTetraMapVec ),
        tetraNum );
    CudaCheckError();

    IntDVec compressedIndex; 
    const int validTetraNum = makeMapAndSum( validVec, compressedIndex );

    if ( LogVerbose )
    {
        cout << "After upper hull removal: " << validTetraNum << endl;
    }

    DTetraData._vec->resize( validTetraNum );

    // Write tetra adjacencies
    kerGrabTetrasFromStars<<< BlocksPerGrid, ThreadsPerBlock >>>(
        DStarData.toKernel(),
        DTetraData.toKernel(),
        toKernelArray( validVec ),
        toKernelArray( dTriTetraMapVec ),
        toKernelArray( dTetraTriMapVec ),
        toKernelArray( compressedIndex ),
        tetraNum
        );
    CudaCheckError();

    return;
}

void _readTetraToHost()
{
    HTetraVec.clear(); 
    DTetraData._vec->copyToHost( HTetraVec ); 

    return;
}

void makeTetraFromStars()
{
    _getTetraFromStars();
    _readTetraToHost();

    return;
}

const TetraHVec& getHostTetra()
{
    return HTetraVec;
}

////////////////////////////////////////////////////////////////////////////////
