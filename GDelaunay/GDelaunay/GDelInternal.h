/*
Author: Ashwin Nanjappa
Filename: GDelInternal.h

Copyright (c) 2013, School of Computing, National University of Singapore. 
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution. 

Neither the name of the National University of University nor the names of its contributors
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
//                   Internal Header for all GPU code
////////////////////////////////////////////////////////////////////////////////

#pragma once

///////////////////////////////////////////////////////////////////// Headers //

#include "STLWrapper.h"
#include "Geometry.h"

// Launch configuration for complex kernels
#define MAX_THREADS_PER_BLOCK       256
#define MAX_PRED_THREADS_PER_BLOCK  32
#define MIN_BLOCKS_PER_MP           2

const int Marker                = -1;
const int MinWorksetSize        = 8; 
const int ProofPointsPerStar    = 4;
const int ExactTriangleMax      = 10240;

/////////////////////////////////////////////////////////////////// Functions //

template< typename T >
T* cuNew( int num )
{
    T* loc              = NULL;
    const size_t space  = num * sizeof( T );
    CudaSafeCall( cudaMalloc( &loc, space ) );

    return loc;
}

template< typename T >
void cuDelete( T** loc )
{
    CudaSafeCall( cudaFree( *loc ) );
    *loc = NULL;
    return;
}

/////////////////////////////////////////////////////////////////////// Types //

enum DPredicateVar
{
    Splitter,       /* = 2^ceiling(p / 2) + 1.  Used to split floats in half. */
    Epsilon,        /* = 2^(-p).  Used to estimate roundoff errors. */
    /* A set of coefficients used to calculate maximum roundoff errors.          */
    Resulterrbound,
    CcwerrboundA,
    CcwerrboundB,
    CcwerrboundC,
    O3derrboundA,
    O3derrboundB,
    O3derrboundC,
    IccerrboundA,
    IccerrboundB,
    IccerrboundC,
    IsperrboundA,
    IsperrboundB,
    IsperrboundC,
    Ispwerrbound,   // Added for weighted orient4
    DPredicateVarNum
};

enum DPredicateDataInfo
{
    // Size of each array
    Temp96Size  = 96,
    Temp192Size = 192,
    Det384xSize = 384,
    Det384ySize = 384,
    Det384zSize = 384,
    Det192wSize = 192,
    DetxySize   = 768,
    DetxyzSize  = 1152,
    AdetSize    = 1152,
    AbdetSize   = 2304,
    CdedetSize  = 3456,
    DeterSize   = 5760,

    // Total size
    PredicateDataTotalSize = 0
    + Temp96Size
    + Temp192Size
    + Det384xSize
    + Det384ySize
    + Det384zSize
    + Det192wSize
    + DetxyzSize
    + DetxySize
    + AdetSize
    + AbdetSize
    + CdedetSize
    + DeterSize
};

enum KernelMode
{
    KernelModeInvalid,

    // Read quad from grid
    CountPerThreadPairs,
    GrabPerThreadPairs,
};

enum FlagStatus
{
    ExactTriNum,
    FlagNum,    // Note: Keep this as last one
};

// Empty triangles are Free
// Triangles are Valid or ValidAndUnchecked rest of the time
typedef char TriangleStatus;
const TriangleStatus Valid                  = 0;
const TriangleStatus ValidAndUnchecked      = 1;
const TriangleStatus NewValidAndUnchecked   = 2;
const TriangleStatus Free                   = 3;
const TriangleStatus DoExactOnValid         = 4;
const TriangleStatus DoExactOnUnchecked     = 5;
const TriangleStatus Beneath                = 6;

__forceinline__ __host__ __device__ bool triNeedsExactCheck( TriangleStatus _status ) 
{
    return ( ( DoExactOnValid == _status ) || ( DoExactOnUnchecked == _status ) );
}

struct TriOppTetra
{
    LocTriIndex  _opp[3]; 
};

struct KerPointData
{
    Point3* _pointArr;
    Weight* _weightArr;
    int     _num;
};

struct KerTetraData
{
    Tetrahedron*  _arr;
    int           _num;
};

typedef int TriPosition;

struct TriPositionEx
{
    int _arrId;
    int _triIdx;
};

__forceinline__ __device__ TriPositionEx decodeTriPos( const TriPosition triPos ) 
{
    TriPositionEx triPosEx; 

    triPosEx._arrId     = ( triPos & 1 ); 
    triPosEx._triIdx    = ( triPos >> 1 ); 

    return triPosEx;
}

__forceinline__ __device__ TriPositionEx makeTriPosEx( int triIdx, int arrId ) 
{
    TriPositionEx triPosEx; 

    triPosEx._arrId     = arrId; 
    triPosEx._triIdx    = triIdx; 

    return triPosEx;
}

__forceinline__ __device__ TriPosition encodeTriPos( const TriPositionEx triPosEx )
{
    return ( ( triPosEx._triIdx << 1 ) + triPosEx._arrId ); 
}

/////////////////////////////////////////////////////////// Death Certificate //

// Death certificate status
const int Alive         = -1;
const int DeadNeedCert  = -2;

const int DeathCertSize = 5;

// _v[0] holds status (while alive) and killer once dead
// _v[1-4] are certificate
struct DeathCert
{
    int _v[ DeathCertSize ];
};

__forceinline__ __host__ __device__ int getDeathStatus( const DeathCert& cert )
{
    return cert._v[0];
}

__forceinline__ __device__ void markStarDeath( DeathCert& cert, int killer, TriPosition validTriPos )
{
    CudaAssert( ( Alive == cert._v[0] ) && "Star is not alive before its death!" );

    cert._v[0] = DeadNeedCert;
    cert._v[1] = validTriPos;

    return;
}

__forceinline__ __device__ int flipVertex( int inVert )
{
    return ( - inVert - 1 );
}

// Indices and sizes of triangles of a star
struct StarInfo
{
    int _beg0; 
    int _beg1_size0;    
    int _triNum0; 
    int _size0; 
    int _totalSize;

    __forceinline__ __device__ int toGlobalTriIdx( int locTriIdx ) const
    {
        return ( locTriIdx < _size0 ) ? _beg0 + locTriIdx : _triNum0 + _beg1_size0 + locTriIdx; 
    }

    __forceinline__ __device__ void moveNextTri( TriPositionEx& triPosEx ) const
    {
        ++triPosEx._triIdx; 

        if ( ( 0 == triPosEx._arrId ) & ( triPosEx._triIdx == _beg0 + _size0 ) )
        {
            triPosEx._triIdx    = _beg1_size0 + _size0; 
            triPosEx._arrId     = 1; 
        }

        return;
    }

    __forceinline__ __device__ LocTriIndex toLocTriIdx( TriPositionEx triPosEx ) const
    {
        return ( triPosEx._arrId == 0 ) ? ( triPosEx._triIdx - _beg0 ) : ( triPosEx._triIdx - _beg1_size0 ); 
    }    

    __forceinline__ __device__ TriPositionEx locToTriPos( int locIdx ) const
    {
        TriPositionEx triPosEx; 

        triPosEx._arrId     = ( locIdx >= _size0 ); 
        triPosEx._triIdx    = ( locIdx >= _size0 ) ? _beg1_size0 + locIdx : _beg0 + locIdx; 

        return triPosEx; 
    }
};

struct KerStarData
{
    int _starNum;
    int _totalTriNum;
    int _triNum[2];

    Triangle*       _triArr[2];
    TriangleOpp*    _triOppArr[2];
    TriOppTetra*    _triOppTetra[2]; 
    int*            _triStarArr[2]; 
    TriangleStatus* _triStatusArr[2]; 

    int*            _starTriMap[2];
    int*            _pointNumArr;
    TriPosition*    _insStatusArr;
    int*            _insCountArr;
    DeathCert*      _deathCertArr;
    int*            _flagArr;

    __forceinline__ __device__ StarInfo getStarInfo( int star ) const
    {
        const int triIdxBeg0    = _starTriMap[0][ star ];
        const int triIdxBeg1    = _starTriMap[1][ star ]; 

        const int triIdxEnd0    = ( star < ( _starNum - 1 ) ) ? _starTriMap[0][ star + 1 ] : _triNum[0];
        const int triIdxEnd1    = ( star < ( _starNum - 1 ) ) ? _starTriMap[1][ star + 1 ] : _triNum[1];

        CudaAssert( ( -1 != triIdxBeg0 ) && ( -1 != triIdxBeg1 ) && ( -1 != triIdxEnd0 ) && ( -1 != triIdxEnd1 ) ); 

        StarInfo starInfo; 

        starInfo._beg0          = triIdxBeg0; 
        starInfo._size0         = triIdxEnd0 - triIdxBeg0; 
        starInfo._triNum0       = _triNum[0]; 
        starInfo._beg1_size0    = triIdxBeg1 - starInfo._size0; 
        starInfo._totalSize     = triIdxEnd0 - triIdxBeg0 + triIdxEnd1 - triIdxBeg1; 

        return starInfo; 
    }

    __forceinline__ __device__ TriPositionEx globToTriPos( int idx ) const
    {
        CudaAssert( ( idx >= 0 ) && ( idx < _totalTriNum ) && "Invalid index!" );

        TriPositionEx triPosEx; 

        triPosEx._arrId     = ( idx >= _triNum[0] );        
        triPosEx._triIdx    = idx - _triNum[0] * triPosEx._arrId;

        return triPosEx; 
    }

    __forceinline__ __device__ Triangle& triangleAt( TriPositionEx loc )
    {
        return _triArr[ loc._arrId ][ loc._triIdx ];
    }

    __forceinline__ __device__ TriangleOpp& triOppAt( TriPositionEx loc )
    {
        return _triOppArr[ loc._arrId ][ loc._triIdx ];
    }

    __forceinline__ __device__ TriOppTetra& triOppTetraAt( TriPositionEx loc )
    {
        return _triOppTetra[ loc._arrId ][ loc._triIdx ];
    }

    __forceinline__ __device__ int& triStarAt( TriPositionEx loc )
    {
        return _triStarArr[ loc._arrId ][ loc._triIdx ];
    }

    __forceinline__ __device__ TriangleStatus& triStatusAt( TriPositionEx loc )
    {
        return _triStatusArr[ loc._arrId ][ loc._triIdx ];
    }
};

struct PredicateInfo
{
    RealType*  _consts;
    RealType*  _data;

    void init()
    {
        _consts = NULL;
        _data   = NULL;
        return;
    }

    void deInit()
    {
        cuDelete( &_consts );
        cuDelete( &_data );

        return;
    }
};

struct KerFacetData
{
    int*        _fromStarArr;
    int*        _toStarArr;
    int*        _fromTriArr;
    Segment*    _segArr;

    int*    _insertMapArr;
    int     _num;
    int     _drownedFacetNum; 
};

struct KerTriangleData
{
    Triangle*       _triArr[2];         // Triangles of stars
    TriangleOpp*    _triOppArr[2];
    int*            _triStarArr[2];     // Star containing triangle
    TriOppTetra*    _triOppTetra[2];    // The local triangle index of the opposite tetrahedron
    TriangleStatus* _triStatusArr[2];   // Triangle status
}; 

struct KerInsertionData
{
    int* _vertArr;
    int* _vertStarArr;
    int* _starVertMap;

    int  _starNum;
    int  _vertNum;
};

struct KerDrownedData
{
    int* _arr;
    int* _mapArr;
    int* _indexArr;
    int  _num;
};

template < typename T >
struct KerArray
{
    T*  _arr;
    int _num;
};

template < typename T >
KerArray< T > toKernelArray( thrust::device_vector< T >& dVec )
{
    KerArray< T > tArray;
    tArray._arr = thrust::raw_pointer_cast( &dVec[0] );
    tArray._num = ( int ) dVec.size();

    return tArray;
}

typedef KerArray< short >           KerShortArray;
typedef KerArray< int >             KerIntArray;
typedef KerArray< Triangle >        KerTriangleArray;
typedef KerArray< TriangleStatus >  KerTriStatusArray;
typedef KerArray< TriPosition >     KerTriPositionArray;

////////////////////////////////////////////////////////////////////////////////
