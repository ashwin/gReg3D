/*
Author: Ashwin Nanjappa
Filename: GDelKernels.h

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
//                               Star Device Code
////////////////////////////////////////////////////////////////////////////////

#pragma once

///////////////////////////////////////////////////////////////////// Kernels //

// Forward Declarations
struct KerFacetData;
struct KerInsertionData;
struct KerPointData;
struct KerStarData;
struct KerTetraData;
struct PredicateInfo;

// Predicate kernels
__global__ void kerMakeInitialConeFast( PredicateInfo, KerPointData, KerStarData, KerInsertionData, KerIntArray );
__global__ void kerMakeInitialConeExact( PredicateInfo, KerPointData, KerStarData, KerInsertionData, KerIntArray );
__global__ void kerMarkBeneathTrianglesFast( PredicateInfo, KerPointData, KerStarData, KerInsertionData, KerIntArray, KerShortArray, KerTriPositionArray, int );
__global__ void kerMarkBeneathTrianglesExact( PredicateInfo, KerPointData, KerStarData, KerInsertionData, KerIntArray, KerShortArray, KerTriPositionArray, int );
__global__ void kerFindDrownedFacetInsertionsFast( PredicateInfo, KerPointData, KerStarData, KerFacetData, KerInsertionData );
__global__ void kerFindDrownedFacetInsertionsExact( PredicateInfo, KerPointData, KerStarData, KerFacetData, KerInsertionData );
__global__ void kerCheckValidTetras( PredicateInfo, KerPointData, KerStarData, KerIntArray, KerIntArray, KerIntArray, int );

// Other kernels
__global__ void kerInitPredicate( RealType* );
__global__ void kerReadGridPairs( const int*, int, KerInsertionData, KernelMode );
__global__ void kerMarkStarvingWorksets( KerInsertionData, KerIntArray, KerIntArray );
__global__ void kerPumpWorksets( KerInsertionData, KerIntArray, KerIntArray );
__global__ void kerMakeAllStarMap( KerIntArray, KerIntArray, int );
__global__ void kerMissingPointWorksetSize( KerPointData, KerInsertionData, const int*, int, KerIntArray, KerIntArray );
__global__ void kerMakeMissingPointWorkset( KerInsertionData, KerIntArray, KerIntArray, KerIntArray );
__global__ void kerGetPerTriangleInsertion( KerStarData, KerInsertionData, KerIntArray, KerIntArray, KerIntArray, KerShortArray );
__global__ void kerStitchPointToHole( KerStarData, KerInsertionData, KerIntArray, int );
__global__ void kerCountPointsOfStar( KerStarData ); 
__global__ void kerGetValidFacetCount( KerStarData, KerIntArray );
__global__ void kerMakeFacetFromDrownedPoint( KerStarData, KerInsertionData, KerFacetData );
__global__ void kerGetValidFacets( KerStarData, KerIntArray, KerFacetData );
__global__ void kerGetFacetInsertCount( KerStarData, KerFacetData );
__global__ void kerGetPerFacetInsertions( KerStarData, KerFacetData, KerInsertionData );
__global__ void kerCheckCertInsertions( KerStarData, KerInsertionData );
__global__ void kerCountPerStarInsertions( KerStarData, KerInsertionData );
__global__ void kerComputeTriangleCount( KerStarData, KerIntArray );
__global__ void kerMarkOwnedTriangles( KerStarData, KerIntArray );
__global__ void kerGrabTetrasFromStars( KerStarData, KerTetraData, KerIntArray, KerIntArray, KerIntArray, KerIntArray, int );
__global__ void kerInvalidateFreeTriangles( KerStarData );
__global__ void kerCheckStarConsistency( KerStarData );
__global__ void kerAppendValueToKey( KerInsertionData, int );
__global__ void kerRemoveValueFromKey( KerInsertionData, int );
__global__ void kerMarkDuplicates( KerIntArray, KerIntArray ); 
__global__ void kerMakeOldNewTriMap( KerStarData, int, KerIntArray, KerIntArray, KerIntArray, KerTriStatusArray );
__global__ void kerGetActiveTriCount( KerStarData, KerIntArray, KerIntArray );
__global__ void kerGetActiveTriStatus( KerStarData, KerIntArray, KerIntArray ); 
__global__ void kerGetDeathCertificateFast( PredicateInfo, KerPointData, KerStarData, KerFacetData );
__global__ void kerGetDeathCertificateExact( PredicateInfo, KerPointData, KerStarData, KerFacetData );
__global__ void kerSetBeneathToFree( KerStarData, KerIntArray, KerShortArray );

/////////////////////////////////////////////////////////////////// Functions //

template< typename T >
__device__ void cuSwap( T& v0, T& v1 )
{
    const T tmp = v0;
    v0          = v1;
    v1          = tmp;

    return;
}

// Calculate number of triangles for set of starNum 2-spheres
// Sum of number of points in all 2-spheres is pointNum
__forceinline__ __host__ __device__ int get2SphereTriangleNum( int starNum, int pointNum )
{
    // From 2-sphere Euler we have ( t = 2n - 4 )
    return ( 2 * pointNum ) - ( starNum * 4 );
}

// Convert 3D coordinate to index
__forceinline__ __host__ __device__ int coordToIdx( int gridWidth, int3 coord )
{
    return ( ( coord.z * ( gridWidth * gridWidth ) ) + ( coord.y * gridWidth ) + coord.x );
}

__forceinline__ __device__ int getCurThreadIdx()
{
    const int threadsPerBlock   = blockDim.x;
    const int curThreadIdx      = ( blockIdx.x * threadsPerBlock ) + threadIdx.x;
    return curThreadIdx;
}

__forceinline__ __device__ int getThreadNum()
{
    const int blocksPerGrid     = gridDim.x;
    const int threadsPerBlock   = blockDim.x;
    const int threadNum         = blocksPerGrid * threadsPerBlock;
    return threadNum;
}

__forceinline__ __device__ PredicateInfo getCurThreadPredInfo( PredicateInfo predInfo )
{
    const int curPredDataIdx          = getCurThreadIdx() * PredicateDataTotalSize;
    RealType* curPredData             = &( predInfo._data[ curPredDataIdx ] );
    const PredicateInfo curPredInfo   = { predInfo._consts, curPredData };
    return curPredInfo;
}

////////////////////////////////////////////////////////////////////////////////
