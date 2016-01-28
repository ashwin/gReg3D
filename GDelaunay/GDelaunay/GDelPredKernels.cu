/*
Author: Ashwin Nanjappa
Filename: GDelPredKernels.cu

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

///////////////////////////////////////////////////////////////////// Headers //

#include "GDelInternal.h"
#include "GDelKernels.h"
#include "Geometry.h"
#include "GDelPredDevice.h"

///////////////////////////////////////////////////////////////////// Kernels //

__global__ void 
__launch_bounds__( MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP )
kerMakeInitialConeFast
(
PredicateInfo       predInfo,
KerPointData        pointData,
KerStarData         starData,
KerInsertionData    worksetData,
KerIntArray         exactVertArr
)
{
    // For pentachoron of the orientation 0123v, the following are
    // the 4 link-triangles orientation as seen from v.
    // Opposite triangle indices are also the same!
    const int LinkTri[4][3] = {
        { 1, 2, 3 },
        { 0, 3, 2 },
        { 0, 1, 3 },
        { 0, 2, 1 },    };

    const PredicateInfo curPredInfo = getCurThreadPredInfo( predInfo );

    // Iterate through stars
    for ( int star = getCurThreadIdx(); star < starData._starNum; star += getThreadNum() )
    {
        ////
        // Compute and store triangle map
        ////

        const int curWsetIdx            = worksetData._starVertMap[ star ];
        const int triIdxBeg             = get2SphereTriangleNum( star, curWsetIdx );
        starData._starTriMap[0][ star ] = triIdxBeg;    
        starData._starTriMap[1][ star ] = 0;

        ////
        // Read 4 points of star
        ////
        
        const int nextWsetIdx   = ( star < ( starData._starNum - 1 ) ) ? worksetData._starVertMap[ star + 1 ] : worksetData._vertNum;
        const int wsetSize      = nextWsetIdx - curWsetIdx;

        CudaAssert( ( wsetSize >= 4 ) && "Working set too small!" );

        // 4 points
        int linkPtIdx[4];
        for ( int pi = 0; pi < 4; ++pi )
            linkPtIdx[ pi ] = worksetData._vertArr[ curWsetIdx + pi ];

        ////
        // Form 4-simplex with 4 points and star point
        ////

        // Orientation
        const Orient ord = orientation4Fast_w(  curPredInfo, pointData,
                                                linkPtIdx[0], linkPtIdx[1], linkPtIdx[2], linkPtIdx[3], star );

        if ( OrientZero == ord ) 
        {
            // Need exact check
            const int exactListIdx              = atomicAdd( &starData._flagArr[ ExactTriNum ], 1 ); 
            exactVertArr._arr[ exactListIdx ]   = star;

            continue; 
        }

        // Swap for -ve order
        if ( OrientNeg == ord )
            cuSwap( linkPtIdx[0], linkPtIdx[1] );

        ////
        // Write 4 triangles of 4-simplex
        ////

        for ( int ti = 0; ti < 4; ++ti )
        {
            Triangle tri;
            TriangleOpp triOpp;

            for ( int vi = 0; vi < 3; ++vi )
            {
                tri._v[ vi ]        = linkPtIdx[ LinkTri[ ti ][ vi ] ];
                triOpp._opp[ vi ]   = LinkTri[ ti ][ vi ];
            }

            CudaAssert( ( star != tri._v[ 0 ] ) && ( star != tri._v[ 1 ] ) && ( star != tri._v[ 2 ] )
                        && "Star vertex same as one of its cone vertices!" ); 

            const TriPositionEx triPos      = makeTriPosEx( triIdxBeg + ti, 0 );
            starData.triangleAt( triPos )   = tri;
            starData.triOppAt( triPos )     = triOpp;
            starData.triStarAt( triPos )    = star;
            starData.triStatusAt( triPos )  = ValidAndUnchecked;
        }
    }

    return;
}

__global__ void 
__launch_bounds__( MAX_PRED_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP )
kerMakeInitialConeExact
(
PredicateInfo       predInfo,
KerPointData        pointData,
KerStarData         starData,
KerInsertionData    worksetData,
KerIntArray         exactVertArr
)
{
    // For pentachoron of the orientation 0123v, the following are
    // the 4 link-triangles orientation as seen from v.
    // Opposite triangle indices are also the same!
    const int LinkTri[4][3] = {
        { 1, 2, 3 },
        { 0, 3, 2 },
        { 0, 1, 3 },
        { 0, 2, 1 },    };

    // Exact check not needed

    const int exactVertNum = starData._flagArr[ ExactTriNum ]; 

    if ( 0 == exactVertNum )
        return; 

    const PredicateInfo curPredInfo = getCurThreadPredInfo( predInfo );

    // Iterate through stars
    for ( int idx = getCurThreadIdx(); idx < exactVertNum; idx += getThreadNum() )
    {
        const int star = exactVertArr._arr[ idx ];

        ////
        // Compute and store triangle map
        ////

        const int curWsetIdx    = worksetData._starVertMap[ star ];
        const int triIdxBeg     = get2SphereTriangleNum( star, curWsetIdx );

        ////
        // Read 4 points of star
        ////       
        const int nextWsetIdx   = ( star < ( starData._starNum - 1 ) ) ? worksetData._starVertMap[ star + 1 ] : worksetData._vertNum;
        const int wsetSize      = nextWsetIdx - curWsetIdx;

        CudaAssert( ( wsetSize >= 4 ) && "Working set too small!" );

        // 4 points
        int linkPtIdx[4];
        for ( int pi = 0; pi < 4; ++pi )
            linkPtIdx[ pi ] = worksetData._vertArr[ curWsetIdx + pi ];

        ////
        // Form 4-simplex with 4 points and star point
        ////

        // Orientation
        const Orient ord = orientation4SoS_w(   curPredInfo, pointData,
                                                linkPtIdx[0], linkPtIdx[1], linkPtIdx[2], linkPtIdx[3], star );

        CudaAssert( ( OrientZero != ord ) && "Orientation is zero!" );

        // Swap for -ve order
        if ( OrientNeg == ord )
            cuSwap( linkPtIdx[0], linkPtIdx[1] );

        ////
        // Write 4 triangles of 4-simplex
        ////

        for ( int ti = 0; ti < 4; ++ti )
        {
            Triangle tri;
            TriangleOpp triOpp;

            for ( int vi = 0; vi < 3; ++vi )
            {
                tri._v[ vi ]        = linkPtIdx[ LinkTri[ ti ][ vi ] ];
                triOpp._opp[ vi ]   = LinkTri[ ti ][ vi ];
            }

            CudaAssert( star != tri._v[ 0 ] && star != tri._v[ 1 ] && star != tri._v[ 2 ] ); 

            const TriPositionEx triPos      = makeTriPosEx( triIdxBeg + ti, 0 );
            starData.triangleAt( triPos )   = tri;
            starData.triOppAt( triPos )     = triOpp;
            starData.triStarAt( triPos )    = star;
            starData.triStatusAt( triPos )  = ValidAndUnchecked;
        }
    }

    return;
}

__global__ void 
__launch_bounds__( MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP )
kerMarkBeneathTrianglesFast
(
PredicateInfo       predInfo,
KerPointData        pointData,
KerStarData         starData,
KerInsertionData    insertData,
KerIntArray         activeTriArr,
KerShortArray       triInsNumArr,
KerTriPositionArray exactTriArr,
int                 insIdx
)
{
    const PredicateInfo curThreadPredInfo = getCurThreadPredInfo( predInfo );

    // Iterate triangles
    for ( int idx = getCurThreadIdx(); idx < activeTriArr._num; idx += getThreadNum() )
    {
        // Check if any insertion for triangle
        if ( triInsNumArr._arr[ idx ] <= insIdx )
            continue;

        ////
        // Ignore free triangle
        ////

        const int triIdx            = activeTriArr._arr[ idx ]; 
        const TriPositionEx triPos  = starData.globToTriPos( triIdx );
        TriangleStatus& triStatus   = starData.triStatusAt( triPos ); 

        if ( ( Free == triStatus ) || ( Beneath == triStatus ) )
            continue;

        const int star = starData.triStarAt( triPos );

        CudaAssert(     ( ( Valid == triStatus ) || ( ValidAndUnchecked == triStatus ) || ( NewValidAndUnchecked == triStatus ) )
                    &&  "Invalid triangle status for fast-exact check!" );
        CudaAssert( ( Alive == getDeathStatus( starData._deathCertArr[ star ] ) ) && "Star has to be alive for valid triangle!" );

        // Insertion point
        const int insBeg = insertData._starVertMap[ star ];
        const int insEnd = ( star < ( starData._starNum - 1 ) ) ? insertData._starVertMap[ star + 1 ] : insertData._vertNum;
        const int insLoc = insBeg + insIdx;

        CudaAssert( ( insLoc < insEnd ) && "Invalid insertion index!" );

        ////
        // Check if triangle beneath point
        ////

        const Triangle tri          = starData.triangleAt( triPos );
        const TriPosition triPosInt = encodeTriPos( triPos ); 
        const int insVert           = insertData._vertArr[ insLoc ];
        const Orient ord            = orientation4Fast_w( curThreadPredInfo, pointData, tri._v[0], tri._v[1], tri._v[2], star, insVert );

        if ( OrientZero == ord )
        {
            // Embed original status in exact status
            triStatus = ( Valid == triStatus ) ? DoExactOnValid : DoExactOnUnchecked;

            ////
            // Exact check processing
            ////

            const int exactListIdx = atomicAdd( &starData._flagArr[ ExactTriNum ], 1 ); 

            if ( exactListIdx < ExactTriangleMax ) 
                exactTriArr._arr[ exactListIdx ] = triPosInt;
        }
        else if ( OrientNeg == ord )
        {
            starData._insStatusArr[ star ] = triPosInt;
            triStatus                      = Beneath;
        }
        // Orient positive
        else if ( NewValidAndUnchecked == triStatus )
        {
            // New triangle created in the last insertion
            // Set it to a normal triangle
            triStatus = ValidAndUnchecked; 
        }        
    }

    return;
}

__global__ void
__launch_bounds__( MAX_PRED_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP )
kerMarkBeneathTrianglesExact
(
PredicateInfo       predInfo,
KerPointData        pointData,
KerStarData         starData,
KerInsertionData    insertData,
KerIntArray         activeTriArr,
KerShortArray       triInsNumArr,
KerTriPositionArray exactTriArr,
int                 insIdx
)
{
    ////
    // Check if exact needed at all
    ////

    int exactTriNum = starData._flagArr[ ExactTriNum ];

    if ( 0 == exactTriNum )
        return; 

    const PredicateInfo curThreadPredInfo   = getCurThreadPredInfo( predInfo );
    const bool exactCheckAll                = ( exactTriNum >= ExactTriangleMax ); 

    if ( exactCheckAll )
        exactTriNum = activeTriArr._num; 

    // Iterate triangles
    for ( int idx = getCurThreadIdx(); idx < exactTriNum; idx += getThreadNum() )
    {
        ////
        // Check if any insertion for triangle
        ////

        if ( exactCheckAll && ( triInsNumArr._arr[ idx ] <= insIdx ) )
            continue;

        const TriPositionEx triPos = exactCheckAll ? starData.globToTriPos( activeTriArr._arr[ idx ] ) : decodeTriPos( exactTriArr._arr[ idx ] );

        ////
        // Check if triangle needs exact check
        ////

        TriangleStatus& status = starData.triStatusAt( triPos );

        // Ignore triangle not needing exact check
        if ( !triNeedsExactCheck( status ) )
            continue;

        ////
        // Read insertion point
        ////

        const int star = starData.triStarAt( triPos );

        CudaAssert( ( Alive == getDeathStatus( starData._deathCertArr[ star ] ) ) && "Star has to be alive for check on valid triangle!" );

        const int insBeg = insertData._starVertMap[ star ];
        const int insEnd = ( star < ( starData._starNum - 1 ) ) ? insertData._starVertMap[ star + 1 ] : insertData._vertNum;
        const int insLoc = insBeg + insIdx;

        CudaAssert( ( insLoc < insEnd ) && "Invalid insertion index!" );

        ////
        // Check if triangle beneath point
        ////

        const int insVert   = insertData._vertArr[ insLoc ];
        const Triangle& tri = starData.triangleAt( triPos );
        const Orient ord    = orientation4SoS_w( curThreadPredInfo, pointData, tri._v[0], tri._v[1], tri._v[2], star, insVert );

        if ( OrientNeg == ord )
        {
            starData._insStatusArr[ star ]  = encodeTriPos( triPos );
            status                          = Beneath;
        }
        else
        {
            status = ( DoExactOnValid == status ) ? Valid : ValidAndUnchecked; // Set back to old status
        }
    }

    return;
}

template< bool doExact >
__device__ void
getDeathCertificate
(
PredicateInfo   curPredInfo,
KerPointData    pointData,
KerStarData     starData,
int             deadStar,
int             killerVert
)
{
    ////
    // Get first proof point from valid triangle stored earlier
    ////

    DeathCert& deathCert                = starData._deathCertArr[ deadStar ];
    const TriPosition validTriPos       = deathCert._v[1]; // Was stored from stitchPointToHole kernel
    const TriPositionEx validTriPosEx   = decodeTriPos( validTriPos );
    const Triangle& validTri            = starData.triangleAt( validTriPosEx );
    const int exVert                    = validTri._v[ 0 ];

    ////
    // Iterate through triangles to find another
    // intersected by plane of (starVert, killerVert, exVert)
    ////

    const StarInfo starInfo = starData.getStarInfo( deadStar );
    int locTriIdx           = 0;

    for ( ; locTriIdx < starInfo._totalSize; ++locTriIdx )
    {
        ////
        // Ignore non-beneath triangles
        ////

        const TriPositionEx triPos      = starInfo.locToTriPos( locTriIdx );
        const TriangleStatus triStatus  = starData.triStatusAt( triPos );

        if ( Beneath != triStatus )
        {
            CudaAssert( ( Free == triStatus ) && "If not beneath, dead star triangle has to be free!" );
            continue;
        }

        ////
        // Ignore triangles containing first proof point
        ////

        const Triangle tri = starData.triangleAt( triPos );
        
        if ( tri.hasVertex( exVert ) )
            continue;

        ////
        // Check if triangle can be proof
        ////

        Orient ord[3];
        int vi = 0; 

        // Iterate through vertices in order
        for ( ; vi < 3; ++vi )
        {
            const int planeVert = tri._v[ vi ];
            const int testVert  = tri._v[ ( vi + 1 ) % 3 ];

            CudaAssert( ( planeVert >= 0 ) && ( planeVert < starData._starNum ) && "Invalid vertex in beneath triangle!" );
            CudaAssert( ( testVert >= 0 ) && ( planeVert < starData._starNum ) && "Invalid vertex in beneath triangle!" );

            // Get order of testVert against the plane formed by (killerVert, starVert, exVert, planeVert)
            Orient order = orientation4Fast_w( curPredInfo, pointData, deadStar, killerVert, exVert, planeVert, testVert );

            if ( OrientZero == order ) 
            {
                if ( doExact ) 
                    order = orientation4SoS_w( curPredInfo, pointData, deadStar, killerVert, exVert, planeVert, testVert );
                else
                    return; // This is in fast check
            }

            ord[ vi ] = order;

            // Check if orders match, they do if plane intersects facet
            if ( ( vi > 0 ) && ( ord[ vi - 1 ] != ord[ vi ] ) )
                break; 
        }

        if ( vi >= 3 )  // All the orders match, we got our proof
            break; 
    }

    CudaAssert( ( locTriIdx < starInfo._totalSize ) && "Not all proof points were found!" );

    ////
    // Store the proof
    ////

    const TriPositionEx proofTriPos = starInfo.locToTriPos( locTriIdx );
    const Triangle proofTri         = starData.triangleAt( proofTriPos );

    deathCert._v[0] = killerVert;
    deathCert._v[1] = exVert;

    for ( int i = 0; i < 3; ++i )
        deathCert._v[ 2 + i ] = proofTri._v[ i ];
    
    return;
}

__global__ void 
__launch_bounds__( MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP )
kerGetDeathCertificateFast
(
PredicateInfo   predInfo,
KerPointData    pointData,
KerStarData     starData,
KerFacetData    facetData
)
{
    const PredicateInfo curThreadPredInfo = getCurThreadPredInfo( predInfo );

    // Iterate only drowned-dead facets
    for ( int facetIdx = getCurThreadIdx(); facetIdx < facetData._drownedFacetNum; facetIdx += getThreadNum() )
    {
        ////
        // Ignore drowned points
        ////

        const int killerVert = facetData._fromStarArr[ facetIdx ];

        if ( killerVert >= 0 )
            continue;

        ////
        // Find death certificate
        ////

        const int posKillerVert = flipVertex( killerVert );
        const int toStar        = facetData._toStarArr[ facetIdx ];

        CudaAssert( ( toStar < 0 ) && "Dead star has to be negative!" );

        const int deadStar = flipVertex( toStar ); // Flip

        CudaAssert( DeadNeedCert == getDeathStatus( starData._deathCertArr[ deadStar ] ) );
        
        getDeathCertificate< false >( curThreadPredInfo, pointData, starData, deadStar, posKillerVert );
    }

    return;
}

__global__ void
__launch_bounds__( MAX_PRED_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP )
kerGetDeathCertificateExact
(
PredicateInfo   predInfo,
KerPointData    pointData,
KerStarData     starData,
KerFacetData    facetData
)
{
    const PredicateInfo curThreadPredInfo = getCurThreadPredInfo( predInfo );

    // Iterate only drowned-dead facets
    for ( int facetIdx = getCurThreadIdx(); facetIdx < facetData._drownedFacetNum; facetIdx += getThreadNum() )
    {
        ////
        // Ignore drowned points
        ////

        const int killerVert = facetData._fromStarArr[ facetIdx ];

        if ( killerVert >= 0 )
            continue;

        ////
        // Ignore if fast certificate obtained
        ////

        const int posKillerVert = flipVertex( killerVert );
        const int toStar        = facetData._toStarArr[ facetIdx ];

        CudaAssert( ( toStar < 0 ) && "Dead star has to be negative!" );

        const int deadStar  = flipVertex( toStar ); // Flip

        if ( DeadNeedCert != getDeathStatus( starData._deathCertArr[ deadStar ] ) )
            continue;

        ////
        // Find death certificate using exact check
        ////

        getDeathCertificate< true >( curThreadPredInfo, pointData, starData, deadStar, posKillerVert );
    }

    return;
}

////
// The containment proof is star plus 4 points from link of star that encloses input point.
// Returns true if exact check is needed. 
////

template< bool doExact >
__device__ bool
findStarContainmentProof
(
PredicateInfo   curPredInfo,
KerPointData    pointData,
KerStarData     starData,
int             star,       // Star that encloses input point
int             inVert,     // Input point that lies inside star
int*            insertVertArr,
int             insertIdxBeg
)
{
    const StarInfo starInfo = starData.getStarInfo( star );

    ////
    // Pick first triangle as facet intersected by plane
    ////

    int locTriIdx = 0;
    
    for ( ; locTriIdx < starInfo._totalSize; ++locTriIdx )
    {
        const TriPositionEx triPos  = starInfo.locToTriPos( locTriIdx );
        const TriangleStatus status = starData.triStatusAt( triPos );

        // Ignore free triangles
        if ( Free != status )
            break;
    }

    // Pick this valid triangle!
    const TriPositionEx triPos  = starInfo.locToTriPos( locTriIdx );
    const Triangle& firstTri    = starData.triangleAt( triPos );
    const int exVert            = firstTri._v[ 0 ];

    CudaAssert( ( locTriIdx < starInfo._totalSize ) && "No valid triangle found!" );

    int proofVerts[4]; 

    // Proof point
    proofVerts[ 0 ] = exVert;

    ////
    // Iterate through triangles to find another
    // intersected by plane of (starVert, inVert, exVert)
    ////

    for ( ; locTriIdx < starInfo._totalSize; ++locTriIdx )
    {
        // Ignore free triangles
        const TriPositionEx triPos  = starInfo.locToTriPos( locTriIdx );
        const TriangleStatus status = starData.triStatusAt( triPos );
        
        if ( Free == status )
            continue;

        // Ignore triangle if has exVert

        const Triangle tri = starData.triangleAt( triPos );
        
        if ( tri.hasVertex( exVert ) )
            continue;

        Orient ord[3];
        int vi = 0; 

        // Iterate through vertices in order
        for ( ; vi < 3; ++vi )
        {
            const int planeVert = tri._v[ vi ];
            const int testVert  = tri._v[ ( vi + 1 ) % 3 ];

            // Get order of testVert against the plane formed by (inVert, starVert, exVert, planeVert)
            Orient order = orientation4Fast_w( curPredInfo, pointData, star, inVert, exVert, planeVert, testVert );

            if ( OrientZero == order ) 
            {
                if ( doExact ) 
                    order = orientation4SoS_w( curPredInfo, pointData, star, inVert, exVert, planeVert, testVert );
                else
                    return true; 
            }

            ord[ vi ] = order;

            // Check if orders match, they do if plane intersects facet
            if ( ( vi > 0 ) && ( ord[ vi - 1 ] != ord[ vi ] ) )
                break; 
        }

        if ( vi >= 3 )  // All the orders match, we got our proof
            break; 
    }

    CudaAssert( ( locTriIdx < starInfo._totalSize ) && "Wrong number of points in proof!" );

    const TriPositionEx proofTriPos = starInfo.locToTriPos( locTriIdx );
    const Triangle proofTri         = starData.triangleAt( proofTriPos );

    // Proof points [1-3]
    for ( int i = 0; i < 3; ++i )
        proofVerts[ 1 + i ] = proofTri._v[ i ]; 

    ////
    // Remove points from proof that are already in destination star
    ////

    const StarInfo inStarInfo = starData.getStarInfo( inVert );

    // Iterate triangles of star
    for ( int locTriIdx = 0; locTriIdx < inStarInfo._totalSize; ++locTriIdx )
    {
        const TriPositionEx triPos  = inStarInfo.locToTriPos( locTriIdx );
        const TriangleStatus status = starData.triStatusAt( triPos );

        if ( Free == status ) 
            continue; 

        const Triangle tri = starData.triangleAt( triPos );

        for ( int i = 0; i < ProofPointsPerStar; ++i ) 
            if ( tri.hasVertex( proofVerts[ i ] ) )
                proofVerts[ i ] = -1; 
    }

    ////
    // Find at least one surviving point
    ////

    int oneVert = -1; 

    for ( int i = 0; i < ProofPointsPerStar; ++i ) 
        if ( -1 != proofVerts[i] ) 
            oneVert = proofVerts[i]; 

    CudaAssert( ( -1 != oneVert ) && "There must be one proof point not there." ); 

    ////
    // Fill in empty slots with surviving point
    ////

    for ( int i = 0; i < ProofPointsPerStar; ++i ) 
        insertVertArr[ insertIdxBeg + i ] = ( -1 == proofVerts[i] ) ? oneVert : proofVerts[i]; 
    
    return false;
}

__global__ void
kerFindDrownedFacetInsertionsFast
(
PredicateInfo       predInfo,
KerPointData        pointData,
KerStarData         starData,
KerFacetData        facetData,
KerInsertionData    insertData
)
{
    const PredicateInfo curThreadPredInfo = getCurThreadPredInfo( predInfo );

    // Iterate through facets
    for ( int facetIdx = getCurThreadIdx(); facetIdx < facetData._drownedFacetNum; facetIdx += getThreadNum() )
    {
        const int insertIdxBeg  = facetData._insertMapArr[ facetIdx ];
        const int insertIdxEnd  = ( ( facetIdx + 1 ) < facetData._num ) ? facetData._insertMapArr[ facetIdx + 1 ] : insertData._vertNum;

        // No insertions for this facet
        if ( insertIdxBeg == insertIdxEnd )
            continue;

        ////
        // Find containment proof to insert
        ////

        const int toStar = facetData._toStarArr[ facetIdx ];

        CudaAssert( ( toStar < 0 ) && "Not a drowned-dead item!" );
        
        const int posToStar = flipVertex( toStar );    // Flip toStar. We had negated it for sorting.
        const int fromStar  = facetData._fromStarArr[ facetIdx ];      

        CudaAssert( ( fromStar >= 0 ) && "Killer vert should not appear here since its insertion is none!" );
        CudaAssert( ( ProofPointsPerStar == ( insertIdxEnd - insertIdxBeg ) ) && "Invalid number of insertions for proof!" );
        CudaAssert( Alive == getDeathStatus( starData._deathCertArr[ posToStar ] ) );
        CudaAssert( Alive == getDeathStatus( starData._deathCertArr[ fromStar ] ) );

        // Find containment proof using only fast check
        const bool needExact = findStarContainmentProof< false >( curThreadPredInfo, pointData, starData, posToStar, fromStar, insertData._vertArr, insertIdxBeg );

        // Write insert stars
        for ( int idx = insertIdxBeg; idx < insertIdxEnd; ++idx )
            insertData._vertStarArr[ idx ] = fromStar;

        if ( needExact )
            facetData._toStarArr[ facetIdx ] = posToStar; 
    }

    return;
}

__global__ void
__launch_bounds__( MAX_PRED_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP )
kerFindDrownedFacetInsertionsExact
(
PredicateInfo       predInfo,
KerPointData        pointData,
KerStarData         starData,
KerFacetData        facetData,
KerInsertionData    insertData
)
{
    const PredicateInfo curThreadPredInfo = getCurThreadPredInfo( predInfo );

    // Iterate through facets
    for ( int facetIdx = getCurThreadIdx(); facetIdx < facetData._drownedFacetNum; facetIdx += getThreadNum() )
    {
        ////
        // Find containment proof to insert
        ////
        
        const int toStar = facetData._toStarArr[ facetIdx ];    // Flip toStar. We had negated it for sorting.

        if ( toStar < 0 )       // No exact check needed
            continue; 

        const int fromStar      = facetData._fromStarArr[ facetIdx ];      
        const int insertIdxBeg  = facetData._insertMapArr[ facetIdx ];

        // Write proof vertices
        findStarContainmentProof< true >( curThreadPredInfo, pointData, starData, toStar, fromStar, insertData._vertArr, insertIdxBeg );
    }

    return;
}

__global__ void kerCheckValidTetras
(
PredicateInfo   predInfo,
KerPointData    pointData,
KerStarData     starData,
KerIntArray     validArr,
KerIntArray     tetraTriMap,
KerIntArray     triTetraMap,
int             tetraNum
)
{
    // Iterate through owned triangles
    for ( int tetraIdx = getCurThreadIdx(); tetraIdx < tetraNum; tetraIdx += getThreadNum() )
    {
        // Owner triangle of tetrahedron
        const int triIdx            = tetraTriMap._arr[ tetraIdx ];
        const TriPositionEx triPos  = starData.globToTriPos( triIdx );

        const int curStar   = starData.triStarAt( triPos );
        const Triangle tri  = starData.triangleAt( triPos );

        CudaAssert( Valid == starData.triStatusAt( triPos ) );

        CudaAssert( ( curStar >= 0 ) && ( curStar < starData._starNum ) && "Invalid star vertex!" );

        ////
        // Orientation of tetra
        // Note: Non-SoS check is enough since flat and -ve tetra are removed
        ////

        const Point3* ptArr = pointData._pointArr;
        const Point3* p[]   = { &( ptArr[ tri._v[0] ] ), &( ptArr[ tri._v[1] ] ), &( ptArr[ tri._v[2] ] ), &( ptArr[ curStar ] ) };
        Orient ord          = shewchukOrient3D( predInfo._consts, p[0]->_p, p[1]->_p, p[2]->_p, p[3]->_p );
        ord                 = flipOrient( ord );

        // Set validity based on order
        if ( OrientPos == ord ) 
        {
            // _validArr is used to compute the new indices for the tetras
            validArr._arr[ tetraIdx ] = 1;

            // Map the owner triangle to the tetra
            triTetraMap._arr[ triIdx ] = tetraIdx;

            ////
            // This tetrahedron exists as triangles in 3 other stars.
            // Set tetrahedron index in those 3 stars.
            ////

            const TriOppTetra oppTetra = starData.triOppTetraAt( triPos ); 

            // Iterate triangle vertices
            for ( int vi = 0; vi < 3; ++vi )
            {
                const int nextStar      = tri._v[ vi ];
                const StarInfo starInfo = starData.getStarInfo( nextStar );
                const int nextGloTriIdx = starInfo.toGlobalTriIdx( oppTetra._opp[ vi ] );

                // Set index
                triTetraMap._arr[ nextGloTriIdx ] = tetraIdx;
            }
        }
    }

    return;
}

////////////////////////////////////////////////////////////////////////////////
