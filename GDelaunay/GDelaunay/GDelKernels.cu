/*
Author: Ashwin Nanjappa
Filename: GDelKernels.cu

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

///////////////////////////////////////////////////////////////////// Kernels //

__forceinline__ __device__ void grabPair
( 
KerInsertionData    worksetData, 
int                 aVal, 
int                 bVal, 
int&                curPairIdx
)
{
    CudaAssert( aVal != bVal ); 

    if ( aVal == Marker ) 
        return; 

    worksetData._vertStarArr[ curPairIdx ] = aVal; 
    worksetData._vertArr    [ curPairIdx ] = bVal; 

    worksetData._vertStarArr[ curPairIdx + 1 ] = bVal; 
    worksetData._vertArr    [ curPairIdx + 1 ] = aVal; 
    
    curPairIdx += 2;

    return;
}

// Do NOT change this to inline device function.
// Inline device function was found to be comparitively slower!
#define READ_GRID_VALUE( dGrid, gridWidth, loc, value )     \
    /* Outer layer */                                       \
    if (   ( loc.x == -1 ) || ( loc.x == gridWidth )        \
        || ( loc.y == -1 ) || ( loc.y == gridWidth )        \
        || ( loc.z == -1 ) || ( loc.z == gridWidth ) )      \
    {                                                       \
        value = Marker;                                     \
    }                                                       \
    else                                                    \
    /* Inner region */                                      \
    {                                                       \
        const int curIdx    = coordToIdx( gridWidth, loc ); \
        value               = dGrid[ curIdx ];              \
    }

// Read pairs from grid, one thread per row
// Invoked twice:
// 1: Count tetra
// 2: Read tetra
__global__ void kerReadGridPairs
(
const int*          dGrid,
int                 gridWidth,
KerInsertionData    worksetData, 
KernelMode          mode
)
{
    // 8 voxels and their Voronoi vertices
    const int Root  = 0; 
    const int Opp   = 7; 
    const int LinkNum                   = 6;
    const int LinkVertex[ LinkNum + 1 ] = { 6, 2, 3, 1, 5, 4, 6 }; 

    int3 loc =   ( blockIdx.x <= gridWidth )
                ? make_int3( threadIdx.x - 1, blockIdx.x - 1, -1 )
                : make_int3( gridWidth - 1, threadIdx.x - 1, -1 ); // Read row on other side of grid

    const int curThreadIdx  = getCurThreadIdx();
    const int pairIdxBeg    =   ( CountPerThreadPairs == mode )
                                ? 0
                                : worksetData._starVertMap[ curThreadIdx ];
    int curPairIdx          = pairIdxBeg;

    int vals[8];
    int valIdx = 0;

    ////
    // Read one plane (4 voxels) in this row
    ////

    for ( int iy = 0; iy <= 1; ++iy ) { for ( int ix = 0; ix <= 1; ++ix )
    {
        const int3 curLoc = make_int3( loc.x + ix, loc.y + iy, loc.z );
        
        READ_GRID_VALUE( dGrid, gridWidth, curLoc, vals[ valIdx ] );

        ++valIdx;
    } } 

    ////
    // Move along row, using plane (4 voxels) from last read
    ////

    // Move along row
    for ( ; loc.z < gridWidth; ++loc.z )
    {     
        // Read 8 voxels around this voxel
        valIdx = 4;

        for ( int iy = 0; iy <= 1; ++iy ) { for ( int ix = 0; ix <= 1; ++ix )
        {
            const int3 curLoc = make_int3( loc.x + ix, loc.y + iy, loc.z + 1 );

            READ_GRID_VALUE( dGrid, gridWidth, curLoc, vals[ valIdx ] );

            ++valIdx;
        } } 

        // Check the main diagonal 
        const int rootVal       = vals[ Root ]; 
        const int oppVal        = vals[ Opp ]; 
        const bool hasMarker    = ( rootVal == Marker ) || ( oppVal == Marker );
        
        if ( rootVal != oppVal ) 
        {
            // Check 6 link pairs
            bool hasQuad    = false; 
            int aVal        = vals[ LinkVertex[ 0 ] ]; 

            for ( int vi = 0; vi < LinkNum; ++vi )
            {
                const int bVal = vals[ LinkVertex[ vi + 1 ] ]; 

                if (    ( aVal != bVal ) 
                     && ( aVal != rootVal ) && ( aVal != oppVal )
                     && ( bVal != rootVal ) && ( bVal != oppVal ) )
                {
                    // Just count
                    if ( CountPerThreadPairs == mode ) 
                    {
                        // 10 pairs normally, 6 pairs if either Root or Opp is Marker
                        curPairIdx += ( hasMarker ? 6 : 10 ); 
                    }
                    // Just read
                    else
                    {
                        grabPair( worksetData, rootVal, aVal, curPairIdx ); 
                        grabPair( worksetData, oppVal, aVal, curPairIdx ); 
                        grabPair( worksetData, rootVal, bVal, curPairIdx ); 
                        grabPair( worksetData, oppVal, bVal, curPairIdx ); 
                        grabPair( worksetData, aVal, bVal, curPairIdx ); 
                    }

                    hasQuad = true; 
                }

                aVal = bVal; 
            }
            
            if ( hasQuad && !hasMarker )         // Has a quad
            {
                if ( CountPerThreadPairs == mode ) 
                    curPairIdx += 2; 
                else
                    grabPair( worksetData, rootVal, oppVal, curPairIdx ); 
            }           
        }

        // Store plane for next row
        vals[ 0 ] = vals[ 4 ];
        vals[ 1 ] = vals[ 5 ];
        vals[ 2 ] = vals[ 6 ];
        vals[ 3 ] = vals[ 7 ]; 
    }

    // Write count of thread
    if ( CountPerThreadPairs == mode )
        worksetData._vertArr[ curThreadIdx ] = curPairIdx - pairIdxBeg;

    return;
}

// (1) Mark starving worksets
// (2) Estimate workset sizes for all (including starving)
__global__ void kerMarkStarvingWorksets
(
KerInsertionData    worksetData,
KerIntArray         markArr,
KerIntArray         worksetSizeArr
)
{
    // Iterate stars
    for ( int idx = getCurThreadIdx(); idx < worksetData._starNum; idx += getThreadNum() )
    {
        const int setBeg    = worksetData._starVertMap[ idx ];
        int setEnd          = ( ( idx + 1 ) < worksetData._starNum ) ? worksetData._starVertMap[ idx + 1 ] : worksetData._vertNum;
        if ( ( -1 == setBeg ) || ( -1 == setEnd ) )
            setEnd = setBeg;
        const int setSize   = setEnd - setBeg;

        CudaAssert( ( setSize >= 0 ) && "Invalid set size!" );

        //CudaAssert( ( setSize >= 3 ) && "Cannot be less than 3!" );

        // Check if starving
        if ( ( 0 < setSize ) && ( setSize < MinWorksetSize ) )
        {
            markArr._arr[ idx ]         = MinWorksetSize - setSize; // Mark
            worksetSizeArr._arr[ idx ]  = MinWorksetSize;   
        }
        // Not starving
        else
        {
            markArr._arr[ idx ]         = 0;
            worksetSizeArr._arr[ idx ]  = setSize;
        }
    }

    return;
}

__device__ bool isCandidateInSet
( 
int         vert,
KerIntArray vertArr, 
int         vertBeg, 
int         vertNum,
int         candidateVert )
{
    if ( vert == candidateVert ) 
        return true; 

    for ( int idx = 0; idx < vertNum; ++idx ) 
    {
        if ( candidateVert == vertArr._arr[ vertBeg + idx ] )
            return true; 
    }

    return false; 
}

__global__ void kerPumpWorksets
(
KerInsertionData    worksetData,
KerIntArray         newMapArr,
KerIntArray         newVertArr
)
{
    // Iterate stars
    for ( int star = getCurThreadIdx(); star < worksetData._starNum; star += getThreadNum() )
    {
        const int fromVertBeg   = worksetData._starVertMap[ star ];
        int fromVertEnd         = ( star < ( worksetData._starNum - 1 ) ) ? worksetData._starVertMap[ star + 1 ] : worksetData._vertNum;
        if ( ( fromVertBeg == -1 ) || ( fromVertEnd == -1 ) )
            fromVertEnd = fromVertBeg; 
        const int fromVertNum   = fromVertEnd - fromVertBeg;

        const int toVertBeg     = newMapArr._arr[ star ];
        const int toVertEnd     = ( star < ( worksetData._starNum - 1 ) ) ? newMapArr._arr[ star + 1 ] : newVertArr._num;
        const int toVertNum     = toVertEnd - toVertBeg; 

        // Copy over
        int toSize = 0;

        for ( int fromIdx = fromVertBeg; fromIdx < fromVertEnd; ++fromIdx )
        {
            newVertArr._arr[ toVertBeg + toSize ] = worksetData._vertArr[ fromIdx ];
            ++toSize;
        }

        bool isPumping = ( fromVertNum < toVertNum ); 

        if ( !isPumping ) 
            continue; 

        // For worksets that needs pumping, *pump* it!
        for ( int fromIdx = fromVertBeg; fromIdx < fromVertEnd; ++fromIdx ) 
        {
            // Pick from the working set of a neighbor
            const int neighborVert      = worksetData._vertArr[ fromIdx ];
            const int neighborVertBeg   = worksetData._starVertMap[ neighborVert ];
            const int neighborVertEnd   = ( neighborVert < ( worksetData._starNum - 1 ) ) ? worksetData._starVertMap[ neighborVert + 1 ] : worksetData._vertNum;

            for ( int candidateIdx = neighborVertBeg; candidateIdx < neighborVertEnd; ++candidateIdx )
            {
                const int candidateVert = worksetData._vertArr[ candidateIdx ]; 

                // Check if it's already there
                if ( !isCandidateInSet( star, newVertArr, toVertBeg, toSize, candidateVert ) )
                {
                    newVertArr._arr[ toVertBeg + toSize ] = candidateVert; 
                    ++toSize; 

                    if ( toSize >= toVertNum )      // Enough?
                    {
                        isPumping = false; 
                        break; 
                    }
                }
            }

            if ( !isPumping ) 
                break; 
        }

        // If that's still not enough, start considering vertices 0, 1,....
        int starIdx = 0; 

        while ( toSize < toVertNum ) 
        {
            CudaAssert( ( starIdx < worksetData._starNum ) && "Not enough points" ); 

            // Check if it's already there
            if ( !isCandidateInSet( star, newVertArr, toVertBeg, toSize, starIdx ) )
            {
                // I don't want a missing point to get in here
                const int starBeg   = worksetData._starVertMap[ starIdx ];
                const int starEnd   = ( starIdx < ( worksetData._starNum - 1 ) ) ? worksetData._starVertMap[ starIdx + 1 ] : worksetData._vertNum;

                if ( starBeg >= starEnd )   // Missing point
                    continue; 

                newVertArr._arr[ toVertBeg + toSize ] = starIdx; 
                ++toSize; 
            }

            ++starIdx; 
        }
    }

    return;
}

// Given a list of sorted numbers (has duplicates and is non-contiguous) create a map
// Note: The input map *should* already be initialized to -1 !!!
// Guarantees:
// (1) For a number that is in input list, its map value and its next number's map value will be correct
// (2) For a number that is not in input list, either its map value is -1, the next one is -1, or size is 0
__global__ void kerMakeAllStarMap
(
KerIntArray inArr,
KerIntArray allStarMap,
int         starNum
)
{
    const int curThreadIdx = getCurThreadIdx(); 

    // Iterate input list of numbers
    for ( int idx = curThreadIdx; idx < inArr._num; idx += getThreadNum() )
    {
        const int curVal    = inArr._arr[ idx ]; 
        const int nextVal   = ( ( idx + 1 ) < inArr._num ) ? inArr._arr[ idx + 1 ] : starNum - 1; 

        CudaAssert( ( curVal <= nextVal ) && "Input array of numbers is not sorted!" );

        // Number changes at this index
        if ( curVal != nextVal )
        {
            allStarMap._arr[ curVal + 1 ]   = idx + 1; 
            allStarMap._arr[ nextVal ]      = idx + 1;
        }
    }

    if ( ( 0 == curThreadIdx ) && ( inArr._num > 0 ) )
    {
        const int firstVal          = inArr._arr[ 0 ];
        allStarMap._arr[ firstVal ] = 0;    // Zero index for first value in input list
    }

    return;
}

__global__ void kerMissingPointWorksetSize
(
KerPointData        pointData,
KerInsertionData    worksetData,
const int*          grid,
int                 gridWidth,
KerIntArray         fromStarArr,
KerIntArray         newWorksetSizeArr
)
{
    // Iterate through workset stars
    for ( int star = getCurThreadIdx(); star < worksetData._starNum; star += getThreadNum() )
    {
        const int mapVal    = worksetData._starVertMap[ star ];
        int nextMapVal      = ( star < ( worksetData._starNum - 1 ) ) ? worksetData._starVertMap[ star + 1 ] : worksetData._vertNum;
        if ( mapVal == -1 || nextMapVal == -1 )
            nextMapVal = mapVal; 
        const int starVertNum   = nextMapVal - mapVal;

        // Check if non-missing point
        if ( starVertNum > 0 )
        {
            fromStarArr._arr[ star ]        = star;
            newWorksetSizeArr._arr[ star ]  = starVertNum;
        }
        // Missing point
        else
        {
            ////
            // Convert point to grid location
            ////

            const Point3 point      = pointData._pointArr[ star ];
            const int3 gridCoord    = { ( int ) point._p[0], ( int ) point._p[1], ( int ) point._p[2] };
            const int gridIdx       = coordToIdx( gridWidth, gridCoord );

            ////
            // Get star and its workset at grid location
            ////

            CudaAssert( ( gridIdx >= 0 ) && ( gridIdx < ( gridWidth * gridWidth * gridWidth ) )
                        && "Point out of grid!" );

            int winStar         = grid[ gridIdx ];
            const int winMapVal = worksetData._starVertMap[ winStar ];
            int winNextMapVal   = ( winStar < ( worksetData._starNum - 1 ) ) ? worksetData._starVertMap[ winStar + 1 ] : worksetData._vertNum;
            if ( ( -1 == winMapVal ) || ( -1 == winNextMapVal ) )
                winNextMapVal = winMapVal; 
            int winVertNum = winNextMapVal - winMapVal;

            CudaAssert( ( winMapVal >= 0 ) ); 
            CudaAssert( ( winNextMapVal >= 0 ) ); 
            CudaAssert( ( winVertNum > 0 ) && "Winning star workset cannot be empty!" );

            ////
            // Write back new workset size
            ////

            if ( winVertNum == 0 )   // Even the winning star is missing, then pick arbitrary 4 points
            {
                winVertNum  = MinWorksetSize; 
                winStar     = -1; 
            }
        
            fromStarArr._arr[ star ]        = winStar;
            newWorksetSizeArr._arr[ star ]  = winVertNum;
        }
    }

    return;
}

__global__ void kerMakeMissingPointWorkset
(
KerInsertionData    worksetData,
KerIntArray         fromStarArr,
KerIntArray         newMap,
KerIntArray         newVertArr
)
{
    // Iterate through stars
    for ( int star = getCurThreadIdx(); star < worksetData._starNum; star += getThreadNum() )
    {
        const int newVertBeg        = newMap._arr[ star ];
        const int newVertEnd        = ( star < ( worksetData._starNum - 1 ) ) ? newMap._arr[ star + 1 ] : newVertArr._num;
        const int newStarVertNum    = newVertEnd - newVertBeg;

        CudaAssert( ( newStarVertNum >= MinWorksetSize ) && "Invalid size for star workset!" );

        const int fromStar = fromStarArr._arr[ star ];

        if ( fromStar >= 0 )    // We got a winning star
        {
            const int fromVertBeg       = worksetData._starVertMap[ fromStar ];
            const int fromVertEnd       = ( fromStar < ( worksetData._starNum - 1 ) ) ? worksetData._starVertMap[ fromStar + 1 ] : worksetData._vertNum;
            const int fromStarVertNum   = fromVertEnd - fromVertBeg;

            CudaAssert( ( fromStarVertNum == newStarVertNum ) && "Mismatching workset sizes!" );

            // Copy from source to destination star
            for ( int idx = 0; idx < newStarVertNum; ++idx )
            {
                const int fromIdx   = fromVertBeg + idx;
                const int toIdx     = newVertBeg + idx;

                newVertArr._arr[ toIdx ] = worksetData._vertArr[ fromIdx ];
            }
        }
        else
        {
            CudaAssert( false && "Disconnected point." ); 

            CudaAssert( ( newStarVertNum == MinWorksetSize ) && "Invalid size for star workset!" );
 
            int startId = 0; 

            // Create arbitrary workset 
            for ( int idx = 0; idx < newStarVertNum; ++idx )
            {
                if ( startId == star ) 
                    ++startId; 
                
                newVertArr._arr[ newVertBeg + idx ] = startId; 

                ++startId; 
            }
        }
    }

    return;
}

__global__ void kerGetPerTriangleInsertion
(
KerStarData         starData,
KerInsertionData    insertData,
KerIntArray         activeStarArr,
KerIntArray         activeTriCountArr,
KerIntArray         activeTriArr,
KerShortArray       triInsNumArr
)
{
    // Iterate only active stars
    for ( int idx = getCurThreadIdx(); idx < activeStarArr._num; idx += getThreadNum() )
    {
        const int star = activeStarArr._arr[ idx ]; 

        ////
        // Number of insertions
        ////

        const int insBeg    = insertData._starVertMap[ star ];
        int insEnd          = ( star < ( starData._starNum - 1 ) ) ? insertData._starVertMap[ star + 1 ] : insertData._vertNum;
        if ( ( -1 == insBeg ) || ( -1 == insEnd ) )
            insEnd = insBeg; 
        const int insCount  = insEnd - insBeg; 

        ////
        // Walk through the triangles
        ////

        const StarInfo starInfo = starData.getStarInfo( star ); 
        int activeTriIdx        = activeTriCountArr._arr[ idx ]; 

        // Iterate star triangles
        for ( int locTriIdx = 0; locTriIdx < starInfo._totalSize; ++locTriIdx )
        {
            activeTriArr._arr[ activeTriIdx ]  = starInfo.toGlobalTriIdx( locTriIdx );  // Store global index for reuse
            triInsNumArr._arr[ activeTriIdx ]  = insCount;                              // Store insertion count
            ++activeTriIdx; 
        }
    }

    return;
}

__device__ TriPositionEx getNextFreeTri
(
KerStarData     starData,
StarInfo        starInfo,
TriPositionEx   prevTriPos
) 
{
    // There must be a free triangle
    while ( true )  
    {
        starInfo.moveNextTri( prevTriPos );  

        const TriangleStatus status = starData.triStatusAt( prevTriPos );

        if ( ( Free == status ) || ( Beneath == status ) )
            return prevTriPos;
    }

    CudaAssert( false && "No free triangle found!" );

    return prevTriPos; 
}

// Find first valid triangle adjacent to hole boundary
// There must be one!
__device__ void findFirstHoleSegment
(
KerStarData     starData,
StarInfo        starInfo,
TriPositionEx   beneathTriPos,
TriPositionEx&  firstTriPos, 
int&            firstVi,
TriPositionEx&  firstHoleTriPos
)
{
    const TriangleOpp triOppFirst = starData.triOppAt( beneathTriPos );

    ////
    // Use the cached beneath triangle and check if it is on hole boundary
    ////

    for ( int vi = 0; vi < 3; ++vi ) 
    {
        const TriPositionEx oppTriPos   = starInfo.locToTriPos( triOppFirst._opp[ vi ] );
        const TriangleStatus status     = starData.triStatusAt( oppTriPos );

        if ( ( Free != status ) && ( Beneath != status ) )  // Found a hole edge
        {
            firstTriPos     = oppTriPos; 
            firstHoleTriPos = beneathTriPos;

            TriangleOpp& oppTriOpp  = starData.triOppAt( oppTriPos ); 
            firstVi                 = oppTriOpp.indexOfOpp( starInfo.toLocTriIdx( beneathTriPos ) ); 

            return; 
        }                
    }

    ////
    // Iterate through triangles and find one that is on hole boundary
    ////

    for ( int locTriIdx = 0; locTriIdx < starInfo._totalSize; ++locTriIdx )
    {
        const TriPositionEx triPos  = starInfo.locToTriPos( locTriIdx );
        const TriangleStatus status = starData.triStatusAt( triPos );

        // Ignore beneath triangles
        if ( ( Free == status ) || ( Beneath == status ) )
            continue;

        const TriangleOpp triOpp = starData.triOppAt( triPos );

        // Iterate segments of beneath triangle
        for ( int vi = 0; vi < 3; ++vi )
        {
            const TriPositionEx triOppPos   = starInfo.locToTriPos( triOpp._opp[ vi ] );
            const TriangleStatus status     = starData.triStatusAt( triOppPos );

            if ( Beneath == status )  // Found a hole edge
            {
                firstTriPos     = triPos; 
                firstVi         = vi; 
                firstHoleTriPos = triOppPos;      

                return; 
            }                
        }
    }

    return;
}

__global__ void kerStitchPointToHole
(
KerStarData         starData,
KerInsertionData    insertData,
KerIntArray         activeStarArr,
int                 insIdx
)
{
    // Reset exact check needed flag
    if ( ( 0 == threadIdx.x ) && ( 0 == blockIdx.x ) )
    {
        starData._flagArr[ ExactTriNum ] = 0;
    }

    // Iterate stars
    for ( int idx = getCurThreadIdx(); idx < activeStarArr._num; idx += getThreadNum() )
    {
        const int star                  = activeStarArr._arr[ idx ];
        const TriPosition beneathTri    = starData._insStatusArr[ star ];

        // No insertion needed OR this star is dead
        if ( -1 == beneathTri )
            continue;   // Move on

        const TriPositionEx beneathTriPos = decodeTriPos( beneathTri ); 

        // Mark this as successful insertion (not drowned)
        const int insIdxBeg = insertData._starVertMap[ star ];
        const int insLoc    = insIdxBeg + insIdx;
        const int insVert   = insertData._vertArr[ insLoc ];

        ////
        // Find first hole segment
        ////

        TriPositionEx firstTriPos;
        TriPositionEx firstNewTriPos;
        int firstVi = -1; 

        // Use the first hole triangle to store the first new triangle

        const StarInfo starInfo = starData.getStarInfo( star );

        findFirstHoleSegment( starData, starInfo, beneathTriPos, firstTriPos, firstVi, firstNewTriPos );

        ////
        // Check if insertion point killed star
        ////

        if ( -1 == firstVi )
        {
            markStarDeath( starData._deathCertArr[ star ], insVert, beneathTri );
            insertData._vertStarArr[ insLoc ]   = star;                     // Mark so that we get death certificate later
            insertData._vertArr[ insLoc ]       = flipVertex( insVert );    // Flip so that we can distinguish between drowned and dead items
            continue;   // Move on
        }

        // Mark as successful insertion (not drowned)
        insertData._vertStarArr[ insLoc ] = -1;

        ////
        // First stitched triangle
        ////

        // Get the first two vertices of the hole
        TriPositionEx curTriPos = firstTriPos; 
        const Triangle& curTri  = starData.triangleAt( curTriPos ); 
        const int firstVert     = curTri._v[ ( firstVi + 1 ) % 3 ]; 
        int curVi               = ( firstVi + 2 ) % 3; 
        int curVert             = curTri._v[ curVi ]; 

        // Stitch the first triangle
        const Triangle firstNewTri              = { insVert, curVert, firstVert };
        starData.triangleAt( firstNewTriPos )   = firstNewTri;
        starData.triStatusAt( firstNewTriPos )  = NewValidAndUnchecked;

        // Adjancency with opposite triangle
        TriangleOpp& firstNewTriOpp = starData.triOppAt( firstNewTriPos ); 
        firstNewTriOpp._opp[ 0 ]    = starInfo.toLocTriIdx( firstTriPos );

        ////
        // Walk around hole, stitching rest of triangles
        ////

        TriPositionEx prevFreeTriPos  = makeTriPosEx( starInfo._beg0 - 1, 0 );
        TriPositionEx prevNewTriPos   = firstNewTriPos; 

        // Walk outside the hole in cw direction
        while ( curVert != firstVert ) 
        {
            // Opposite triangle
            const TriangleOpp& curTriOpp        = starData.triOppAt( curTriPos );
            const TriPositionEx gloOppTriPos    = starInfo.locToTriPos( curTriOpp._opp[ ( curVi + 2 ) % 3 ] ); 
            const TriangleStatus status         = starData.triStatusAt( gloOppTriPos ); 

            CudaAssert( Free != status );

            // Opposite triangle is valid (not in hole)
            if ( ( Beneath != status ) && ( NewValidAndUnchecked != status ) )
            {
                // Continue moving
                const TriangleOpp& oppTriOpp    = starData.triOppAt( gloOppTriPos ); 
                const int oppVi                 = oppTriOpp.indexOfOpp( starInfo.toLocTriIdx( curTriPos ) ); 
                curVi                           = ( oppVi + 2 ) % 3;                
                curTriPos                       = gloOppTriPos;
            }
            // Hole triangle is found
            else
            {
                TriPositionEx newTriPos; 

                // Find a free triangle
                if ( Beneath == status )
                {
                    newTriPos = gloOppTriPos; 
                }
                else
                {
                    newTriPos       = getNextFreeTri( starData, starInfo, prevFreeTriPos ); 
                    prevFreeTriPos  = newTriPos; 
                }

                // Get the next vertex in the hole boundary
                const int oppVi         = ( curVi + 2 ) % 3;
                const Triangle& curTri  = starData.triangleAt( curTriPos );
                const int nextVert      = curTri._v[ ( curVi + 1 ) % 3 ]; 

                // New triangle
                const int locNewTriIdx  = starInfo.toLocTriIdx( newTriPos );
                const Triangle newTri   = { insVert, nextVert, curVert };

                // Adjancency with opposite triangle
                TriangleOpp& curTriOpp  = starData.triOppAt( curTriPos );            
                curTriOpp._opp[ oppVi ] = locNewTriIdx;
                TriangleOpp& newTriOpp  = starData.triOppAt( newTriPos ); 
                newTriOpp._opp[ 0 ]     = starInfo.toLocTriIdx( curTriPos );

                // Adjacency with previous new triangle
                TriangleOpp& prevTriOpp = starData.triOppAt( prevNewTriPos );
                prevTriOpp._opp[2]      = locNewTriIdx; 
                newTriOpp._opp[1]       = starInfo.toLocTriIdx( prevNewTriPos ); 
                
                // Last hole triangle
                if ( nextVert == firstVert )
                {
                    TriangleOpp& firstTriOpp    = starData.triOppAt( firstNewTriPos );
                    firstTriOpp._opp[1]         = locNewTriIdx; 
                    newTriOpp._opp[2]           = starInfo.toLocTriIdx( firstNewTriPos ); 
                }

                // Store new triangle data
                starData.triangleAt( newTriPos )    = newTri;
                starData.triStarAt( newTriPos )     = star;
                starData.triStatusAt( newTriPos )   = NewValidAndUnchecked;

                // Prepare for next triangle
                prevNewTriPos = newTriPos; 

                // Move to the next vertex
                curVi   = ( curVi + 1 ) % 3; 
                curVert = nextVert; 
            }
        }
    }

    return;
}

__global__ void kerSetBeneathToFree
(
KerStarData     starData,
KerIntArray     activeTriArr,
KerShortArray   triInsNumArr
)
{
    // Iterate triangles
    for ( int idx = getCurThreadIdx(); idx < activeTriArr._num; idx += getThreadNum() )
    {
        // Ignore triangles that had no insertions
        if ( 0 == triInsNumArr._arr[ idx ] )
            continue;

        const int triIdx            = activeTriArr._arr[ idx ]; 
        const TriPositionEx triPos  = starData.globToTriPos( triIdx );
        TriangleStatus& triStatus   = starData.triStatusAt( triPos ); 

        // Check if beneath triangle
        if ( Beneath == triStatus )
        {
            const int star = starData.triStarAt( triPos );

            // Check if star of triangle is alive
            if ( Alive == getDeathStatus( starData._deathCertArr[ star ] ) )
                triStatus = Free;
        }
    }

    return;
}

__global__ void kerCountPointsOfStar( KerStarData starData )
{
    // Iterate through stars
    for ( int star = getCurThreadIdx(); star < starData._starNum; star += getThreadNum() )
    {
        // Ignore dead star
        if ( Alive != getDeathStatus( starData._deathCertArr[ star ] ) ) continue;

        // Ignore star with no insertion in this round
        if ( 0 == starData._insCountArr[ star ] ) continue;

        const StarInfo starInfo = starData.getStarInfo( star );
        int validTriCount       = starInfo._totalSize;

        for ( int locTriIdx = 0; locTriIdx < starInfo._totalSize; ++locTriIdx )
        {
            const TriPositionEx triPos      = starInfo.locToTriPos( locTriIdx );
            const TriangleStatus triStatus  = starData.triStatusAt( triPos );

            CudaAssert( ( Beneath != triStatus ) && "No beneath triangle should have survived outside of insertion iterations!" );

            if ( Free == triStatus )
                --validTriCount;
        }

        // Get point count
        CudaAssert( ( 0 == ( ( validTriCount + 4 ) % 2 ) ) && "2-sphere triangle count not divisible by 2!" );

        starData._pointNumArr[ star ] = ( validTriCount + 4 ) / 2;
    }

    return;
}

__device__ bool _checkStarHasVertex
(
KerStarData starData,
int         star,
int         inVert
)
{
    const StarInfo starInfo = starData.getStarInfo( star );

    // Iterate triangles of star
    for ( int locTriIdx = 0; locTriIdx < starInfo._totalSize; ++locTriIdx )
    {
        const TriPositionEx triPos  = starInfo.locToTriPos( locTriIdx ); 
        const TriangleStatus status = starData.triStatusAt( triPos );
        const Triangle& tri         = starData.triangleAt( triPos );

        CudaAssert( ( Beneath != status ) && "Cannot be a dead star!" );

        if ( ( Free != status ) && tri.hasVertex( inVert ) )
                return true;
    }

    return false;
}

// Check if triangle v0-v1-v2 is in star
// If not found, check if vertex in star
// Update the vertex to -1 if it doesn't need to be inserted.
// Return the number of insertion required, or -1 if consistent
__device__ int _checkStarHasFacetOrPoint
(
KerStarData     starData,
TriPositionEx   fromTriLoc,
int             toIdxInFrom,
int             toStar,
int&            fromVert, 
int&            v1,
int&            v2
)
{
    const StarInfo toStarInfo   = starData.getStarInfo( toStar );
    int hasVert                 = 0; 

    // Iterate triangles of star
    for ( int toLocTriIdx = 0; toLocTriIdx < toStarInfo._totalSize; ++toLocTriIdx )
    {
        const TriPositionEx triPos  = toStarInfo.locToTriPos( toLocTriIdx );
        const TriangleStatus status = starData.triStatusAt( triPos );

        // Ignore free triangle
        if ( Free == status )
            continue;

        CudaAssert( ( Beneath != status ) && "Star is dead! Cannot happen! Should have been ignored during facet generation!" );

        // Check for vertex in triangle
        const Triangle tri      = starData.triangleAt( triPos );
        const int triHasVert    =   ( tri.hasVertex( fromVert ) ) +  
                                    ( tri.hasVertex( v1 ) << 1 ) +  
                                    ( tri.hasVertex( v2 ) << 2 );  

        hasVert |= triHasVert; 

        if ( 0x7 == triHasVert )    // All three bits are 1
        {
            // Consistent, note down the index
            starData.triOppTetraAt( fromTriLoc )._opp[ toIdxInFrom ] = toLocTriIdx; 

            return -1; // Get out!
        }
    }

    if ( hasVert & 0x01 ) fromVert  = -1; 
    if ( hasVert & 0x02 ) v1        = -1; 
    if ( hasVert & 0x04 ) v2        = -1; 

    return 3 - __popc( hasVert );   // __popc gives the number of 1s in the binary representation
}

// First half of the facet processing logic
__global__ void kerGetFacetInsertCount
(
KerStarData     starData,
KerFacetData    facetData
)
{
    // Iterate facet items
    for ( int facetIdx = getCurThreadIdx(); facetIdx < facetData._num; facetIdx += getThreadNum() )
    {
        // Read facet item
        const int toStar    = facetData._toStarArr[ facetIdx ];
        int& fromStar       = facetData._fromStarArr[ facetIdx ];   // Will be modified below
        int insertPointNum  = -1;

        // Drowned point OR dead star
        if ( facetIdx < facetData._drownedFacetNum )
        {
            CudaAssert( ( toStar < 0 ) && "Drowned/dead to-star should be negative value!" );

            // Reset to positive
            const int posToStar = flipVertex( toStar );

            // Check if star was killed and its killer vert is in this pair
            if ( fromStar < 0 )
            {
                CudaAssert( ( DeadNeedCert == getDeathStatus( starData._deathCertArr[ posToStar ] ) ) && "Killed star has to be in NeedCert state!" );

                insertPointNum = 0;    // No insertions, certificate is generated by another kernel
            }
            // Check if drowner star has just died
            else if ( Alive != getDeathStatus( starData._deathCertArr[ posToStar ] ) )
            {
                insertPointNum = 0;
            }
            // Check if drowned point has died
            else if ( Alive != getDeathStatus( starData._deathCertArr[ fromStar ] ) )
            {
                insertPointNum = 0;
            }
            // Both are not dead, check if drowner is still in link of drowned point
            else if ( _checkStarHasVertex( starData, fromStar, posToStar ) )
            {
                insertPointNum = ProofPointsPerStar;
            }
            else
            {
                insertPointNum = 0;
            }
        }
        // Normal facet item
        else
        {
            CudaAssert( ( toStar >= 0 ) && ( fromStar >= 0 ) && "Invalid from or to stars!" );

            Segment& seg    = facetData._segArr[ facetIdx ];    // Will be modified below
            int triIdx      = facetData._fromTriArr[ facetIdx ]; 

            // Decode
            const int triStarIdx    = triIdx & 0x03; 
            triIdx                >>= 2; 

            ////
            // Check if to-star is already dead
            ////
            
            const int toDeathStatus = getDeathStatus( starData._deathCertArr[ toStar ] );

            if ( Alive != toDeathStatus )
            {
                insertPointNum                      = DeathCertSize;
                facetData._toStarArr[ facetIdx ]    = flipVertex( toStar ); // Note down death by negating to-star
            }
            else
            {
                // Check if to-star has facet
                const TriPositionEx triPos  = starData.globToTriPos( triIdx );
                insertPointNum              = _checkStarHasFacetOrPoint( starData, triPos, triStarIdx, toStar, fromStar, seg._v[0], seg._v[1] );
                
                // Triangle not found in to-star
                if ( insertPointNum != -1 )
                {
                    // Raise the corresponding triangle as unchecked
                    starData.triStatusAt( triPos ) = ValidAndUnchecked; 
                }
                // Triangle found, is consistent
                else
                {
                    insertPointNum = 0; 
                }
            }
        }

        // Set status of facet
        facetData._insertMapArr[ facetIdx ] = insertPointNum;
    }

    return;
}


// Generate facet-item count only for valid triangles
// Free and Beneath triangles are ignored
__global__ void kerGetValidFacetCount
(
KerStarData starData,
KerIntArray countArr
)
{
    const int FacetsPerTriangle = 3;

    // Iterate triangles
    for ( int triIdx = getCurThreadIdx(); triIdx < starData._totalTriNum; triIdx += getThreadNum() )
    {
        const TriPositionEx triPos      = starData.globToTriPos( triIdx );
        const TriangleStatus triStatus  = starData.triStatusAt( triPos );
        const bool doCheckTri           = ( ValidAndUnchecked == triStatus ) || ( NewValidAndUnchecked == triStatus ); 
        countArr._arr[ triIdx ]         = ( doCheckTri ? FacetsPerTriangle : 0 );
    }

    return;
}

__global__ void kerMakeFacetFromDrownedPoint
(
KerStarData         starData,
KerInsertionData    insertData,
KerFacetData        facetData
)
{
    // Iterate drowned points
    for ( int idx = getCurThreadIdx(); idx < facetData._drownedFacetNum; idx += getThreadNum() )
    {
        // Insert drowned point disguised as facet item

        // *Swap* from-star and to-star so that later kernels are easy
        facetData._fromStarArr[ idx ] = insertData._vertArr[ idx ];

        // Flip toStar so it appears at front of facet list after sorting
        facetData._toStarArr[ idx ] = flipVertex( insertData._vertStarArr[ idx ] );
    }

    return;
}

__global__ void kerGetValidFacets
(
KerStarData     starData,
KerIntArray     facetMap,
KerFacetData    facetData
)
{
    // Facet count begins here in count array
    const int facetCountBeg = facetData._drownedFacetNum;

    // Iterate triangles
    for ( int triIdx = getCurThreadIdx(); triIdx < starData._totalTriNum; triIdx += getThreadNum() )
    {
        int facetIdxBeg = facetCountBeg + facetMap._arr[ triIdx ];
        int facetIdxEnd =     ( triIdx < ( starData._totalTriNum - 1 ) ) 
                            ? ( facetCountBeg + facetMap._arr[ triIdx + 1 ] ) 
                            : facetData._num;
        int facetIdx    = facetIdxBeg;

        // Get out if beyond facets or FacetMax bound (whichever is smaller)
        if ( facetIdxBeg >= facetData._num ) 
            break; 

        // No need to check this triangle
        if ( facetIdxEnd == facetIdxBeg )
            continue; 

        ////
        // Add 3 facet items for this triangle
        ////
    
        const TriPositionEx triPos  = starData.globToTriPos( triIdx );
        TriangleStatus& triStatus   = starData.triStatusAt( triPos );
        const Triangle tri          = starData.triangleAt( triPos );
        const int star              = starData.triStarAt( triPos ); 

        CudaAssert( ( Free != triStatus ) && ( Beneath != triStatus ) ); 

        // Set as valid since we are going to check it
        // Will be invalid again if found out during the check
        triStatus = Valid; 

        // Iterate triangle vertices
        for ( int vi = 0; vi < 3; ++vi )
        {
            const int toStar  = tri._v[ vi ];
            const Segment seg = { tri._v[ ( vi + 1 ) % 3 ], tri._v[ ( vi + 2 ) % 3 ] };

            // Facet item
            facetData._fromStarArr[ facetIdx ]  = star;
            facetData._toStarArr[ facetIdx ]    = toStar;
            facetData._fromTriArr[ facetIdx ]   = ( triIdx << 2 ) | vi; // Encode orientation 
            facetData._segArr[ facetIdx ]       = seg;

            ++facetIdx;
        }
    }

    return;
}

__global__ void kerGetPerFacetInsertions
(
KerStarData         starData,
KerFacetData        facetData,
KerInsertionData    insertData
)
{
    // Iterate through non-drowned facets
    for (   int facetIdx = facetData._drownedFacetNum + getCurThreadIdx(); 
            facetIdx < facetData._num;
            facetIdx += getThreadNum() )
    {
        int insertIdxBeg        = facetData._insertMapArr[ facetIdx ];
        const int insertIdxEnd  = ( facetIdx < ( facetData._num - 1 ) ) ? facetData._insertMapArr[ facetIdx + 1 ] : insertData._vertNum;

        // No insertions for this facet
        if ( insertIdxBeg == insertIdxEnd )
            continue;

        ////
        // Point insertions
        ////

        const int toStar    = facetData._toStarArr[ facetIdx ];
        const int fromStar  = facetData._fromStarArr[ facetIdx ];   // Set to -1 by kerGetFacetInsertCount if found in to-star
        const Segment seg   = facetData._segArr[ facetIdx ];        // Set to -1 by kerGetFacetInsertCount if found in to-star
        const int triad[3]  = { fromStar, seg._v[0], seg._v[1] };

        // To-star is dead
        if ( toStar < 0 )
        {
            const int posToStar = flipVertex( toStar );

            CudaAssert(     ( getDeathStatus( starData._deathCertArr[ posToStar ] ) >= 0 ) &&  "To-Star must be dead with certificate!" );
            CudaAssert( ( DeathCertSize == ( insertIdxEnd - insertIdxBeg ) ) && "Invalid insertion size for to-star death!" );
            
            const DeathCert& deathCert = starData._deathCertArr[ posToStar ];

            ////
            // Store death certificate as insertions
            ////

            for ( int i = 0; i < DeathCertSize; ++i )
            {
                const int certVert                      = deathCert._v[ i ];
                insertData._vertStarArr[ insertIdxBeg ] = fromStar;
                insertData._vertArr[ insertIdxBeg ]     = flipVertex( certVert );   // Negate to check if in star later

                ++insertIdxBeg;
            }
        }
        // To-star is alive
        else
        {
            for ( int i = 0; i < 3; ++i )
            {
                const int vert = triad[i];

                // Vertex not in to-star
                if ( vert != -1 ) 
                {
                    // Insert this point since it is not there yet
                    insertData._vertStarArr[ insertIdxBeg ] = toStar;
                    insertData._vertArr[ insertIdxBeg ]     = vert;

                    ++insertIdxBeg; 
                }
            }

            CudaAssert( ( insertIdxBeg == insertIdxEnd ) && "Invalid number of insertions for PointInconsistent" ); 
        }
    }

    return;
}

// Insertions that do not pass inspection remain negative, and are removed during compaction
__global__ void kerCheckCertInsertions
(
KerStarData         starData,
KerInsertionData    insertData
)
{
    // Iterate insertions
    for ( int idx = getCurThreadIdx(); idx < insertData._vertNum; idx += getThreadNum() )
    {
        const int insVert = insertData._vertArr[ idx ];

        // Ignore vertices that need no inspection
        // Note: Only death certificate insertions need inspection
        if ( insVert >= 0 )
            continue;

        const int insStar       = insertData._vertStarArr[ idx ];
        const int posInsVert    = flipVertex( insVert );

        // 1: Ensure insert vertex is not same as star
        if ( insStar == posInsVert )
            continue;

        // 2: Ensure insert vertex is not in link of star
        if ( _checkStarHasVertex( starData, insStar, posInsVert ) )
            continue;

        // Make insert vertex positive, so it is not removed during compaction
        insertData._vertArr[ idx ] = posInsVert;
    }

    return;
}

__global__ void kerCountPerStarInsertions
(
KerStarData         starData,
KerInsertionData    insertData
)
{
    for ( int star = getCurThreadIdx(); star < starData._starNum; star += getThreadNum() )
    {
        ////
        // Update point number to be inserted AND drowned number
        ////

        const int insBeg  = insertData._starVertMap[ star ];
        int insEnd        = ( star < ( starData._starNum - 1 ) ) ? insertData._starVertMap[ star + 1 ] : insertData._vertNum;
        if ( ( -1 == insBeg ) || ( -1 == insEnd ) ) 
            insEnd = insBeg; 
        const int insPointNum = insEnd - insBeg;

        CudaAssert( ( insPointNum >= 0 ) && "Invalid indices!" );

        // Insert point count for this star
        starData._insCountArr[ star ] = insPointNum;

        ////
        // Drowned count for this star
        // Given star of n link points and m insertion points, only a maximum
        // of (n + m - 4) points can drown
        ////

        const int starPointNum      = starData._pointNumArr[ star ];
        const int totalPointNum     = starPointNum + insPointNum;

        // Update star point count
        starData._pointNumArr[ star ] = totalPointNum;
    }

    return;
}

__global__ void kerComputeTriangleCount
(
KerStarData starData,
KerIntArray triNumArr
)
{
    const float ExpandFactor = 1.0f;

    // Iterate through stars
    for ( int star = getCurThreadIdx(); star < starData._starNum; star += getThreadNum() )
    {
        // Current number of triangles
        const StarInfo starInfo = starData.getStarInfo( star );
        const int curTriNum     = starInfo._totalSize;

        // Expected number of points (current + expected insertions)
        const int expPointNum = starData._pointNumArr[ star ];

        // Expected number of triangles
        const int insTriNum     = get2SphereTriangleNum( 1, expPointNum );
        const int newTriNum     = insTriNum * ExpandFactor;
        triNumArr._arr[ star ]  = max( newTriNum, curTriNum ) - starInfo._size0;    // Only "expand" second array
    }

    return;
}

__global__ void kerMarkOwnedTriangles
(
KerStarData starData,
KerIntArray tetraTriMap
)
{
    // Iterate triangles
    for ( int triIdx = getCurThreadIdx(); triIdx < starData._totalTriNum; triIdx += getThreadNum() )
    {
        const TriPositionEx triPos = starData.globToTriPos( triIdx );
        const TriangleStatus status = starData.triStatusAt( triPos ); 

        if ( ( Free == status ) || ( Beneath == status ) )
            continue; 

        // Check if star is triangle owner
        const Triangle tri  = starData.triangleAt( triPos );
        const int star      = starData.triStarAt( triPos );

        if ( ( star < tri._v[0] ) & ( star < tri._v[1] ) & ( star < tri._v[2] ) )
            tetraTriMap._arr[ triIdx ] = triIdx;
    }

    return;
}

__global__ void kerGrabTetrasFromStars
(
KerStarData     starData,
KerTetraData    tetraData,
KerIntArray     validArr,
KerIntArray     triTetraMap,
KerIntArray     tetraTriMap,
KerIntArray     compressedIndex,
int             tetraNum
)
{
    // Iterate list of all triangles
    for ( int idx = getCurThreadIdx(); idx < tetraNum; idx += getThreadNum() )
    {
        if ( 0 == validArr._arr[ idx ] )
            continue; 

        // Construct the 4 vertices of the tetra
        const TriPositionEx triPos  = starData.globToTriPos( tetraTriMap._arr[ idx ] );
        const int star              = starData.triStarAt( triPos ); 
        const StarInfo starInfo     = starData.getStarInfo( star );
        const Triangle tri          = starData.triangleAt( triPos );

        CudaAssert( Valid == starData.triStatusAt( triPos ) );

        Tetrahedron tetra;
        const int v0    = tri._v[0]; 
        tetra._v[0]     = v0;
        tetra._v[1]     = tri._v[1];
        tetra._v[2]     = tri._v[2];
        tetra._v[3]     = star;

        const TriangleOpp triOpp = starData.triOppAt( triPos ); 

        for ( int vi = 0; vi < 3; ++vi ) 
        {
            const int gloOppTriIdx  = starInfo.toGlobalTriIdx( triOpp._opp[ vi ] );
            const int oppTetraIdx   = triTetraMap._arr[ gloOppTriIdx ];

            tetra._opp[ vi ] = ( -1 == oppTetraIdx ) ? -1 : compressedIndex._arr[ oppTetraIdx ];
        }

        // Locate the triangle in _v[ 0 ]'s star
        const TriOppTetra& oppTetra     = starData.triOppTetraAt( triPos ); 
        const StarInfo v0Info           = starData.getStarInfo( v0 );   
        const TriPositionEx v0TriPos    = v0Info.locToTriPos( oppTetra._opp[ 0 ] );

        // Find the tetra opposite _v[ 3 ]
        const Triangle v0tri        = starData.triangleAt( v0TriPos ); 
        const int starIdx           = v0tri.indexOfVert( star ); 
        const TriangleOpp& v0triOpp = starData.triOppAt( v0TriPos ); 
        const int starTriOppIdx     = v0Info.toGlobalTriIdx( v0triOpp._opp[ starIdx ] ); 
        const int starOppTetraIdx   = triTetraMap._arr[ starTriOppIdx ]; 

        tetra._opp[ 3 ] = ( -1 == starOppTetraIdx ) ? -1 : compressedIndex._arr[ starOppTetraIdx ]; 

        // Write tetra
        tetraData._arr[ compressedIndex._arr[ idx ] ] = tetra;
    }

    return;
}

__global__ void kerInvalidateFreeTriangles( KerStarData starData )
{
    // Iterate through triangles
    for ( int triIdx = getCurThreadIdx(); triIdx < starData._totalTriNum; triIdx += getThreadNum() )
    {
        const TriPositionEx triPos      = starData.globToTriPos( triIdx );
        const TriangleStatus triStatus  = starData.triStatusAt( triPos ); 

        if ( ( Free == triStatus ) || ( Beneath == triStatus ) )
        {
            Triangle& tri = starData.triangleAt( triPos );

            // Invalidate triangle, so it can be detected during consistency check
            tri._v[0] = -1;
        }
    }

    return;
}

__global__ void kerCheckStarConsistency( KerStarData starData )
{
    // Iterate all triangles
    for ( int triIdx = getCurThreadIdx(); triIdx < starData._totalTriNum; triIdx += getThreadNum() )
    {
        ////
        // Ignore free triangles
        ////

        const TriPositionEx triPos  = starData.globToTriPos( triIdx );
        TriangleStatus& triStatus   = starData.triStatusAt( triPos ); 

        if ( ( Free == triStatus ) || ( Beneath == triStatus ) )
            continue; 

        CudaAssert( ( Valid == triStatus ) && "Triangles should be valid at this stage!" );

        ////
        // Check if triangle is consistent
        ////

        const int star                  = starData.triStarAt( triPos ); 
        const Triangle tri              = starData.triangleAt( triPos ); 
        const TriOppTetra triOppTetra   = starData.triOppTetraAt( triPos ); 

        for ( int vi = 0; vi < 3; ++vi ) 
        {
            const int   toVert              = tri._v[ vi ]; 
            const StarInfo starInfo         = starData.getStarInfo( toVert ); 
            const TriPositionEx oppTriPos   = starInfo.locToTriPos( triOppTetra._opp[ vi ] );
            const Triangle oppTri           = starData.triangleAt( oppTriPos );

            // Check if triangle matches
            // Note: Free-Beneath triangle._v[0] has been set to -1, so they fail
            const bool consistent = oppTri.hasVertex( star ) &&
                                    oppTri.hasVertex( tri._v[ ( vi + 1 ) % 3 ] ) &&
                                    oppTri.hasVertex( tri._v[ ( vi + 2 ) % 3 ] ); 

            if ( !consistent ) 
            {
                starData._flagArr[ ExactTriNum ]    = 1; 
                triStatus                           = ValidAndUnchecked; 
                break; 
            }
        }
    }

    return;
}

__global__ void kerAppendValueToKey( KerInsertionData insertData, int bitsPerValue ) 
{
    const int ValMask = 1 << bitsPerValue;

    // Iterate array
    for ( int idx = getCurThreadIdx(); idx < insertData._vertNum; idx += getThreadNum() )
    {
        const int key = insertData._vertStarArr[ idx ]; 
        const int val = insertData._vertArr[ idx ];

        CudaAssert( ( key >= 0 ) && "Invalid key!" );

        insertData._vertStarArr[ idx ] = ( ( key << bitsPerValue ) | ( val & ( ValMask - 1 ) ) ); 
    }

    return;
}

__global__ void kerRemoveValueFromKey( KerInsertionData insertData, int bitsPerValue ) 
{
    // Iterate array
    for ( int idx = getCurThreadIdx(); idx < insertData._vertNum; idx += getThreadNum() )
    {
        const int keyvalue = insertData._vertStarArr[ idx ]; 

        CudaAssert( ( keyvalue >= 0 ) && "Key-Value is invalid!" );

        insertData._vertStarArr[ idx ] = ( keyvalue >> bitsPerValue ); 
    }

    return;
}

__global__ void kerMarkDuplicates( KerIntArray keyArr, KerIntArray valueArr )
{
    // Iterate array
    for ( int idx = getCurThreadIdx(); idx < keyArr._num; idx += getThreadNum() )
    {
        const int key = keyArr._arr[ idx ]; 
        const int val = valueArr._arr[ idx ]; 

        int nextIdx = idx + 1; 

        while ( nextIdx < keyArr._num ) 
        {
            const int nextKey = keyArr._arr[ nextIdx ]; 

            if ( nextKey != key ) 
                break; 

            const int nextVal = valueArr._arr[ nextIdx ]; 

            if ( ( val == nextVal ) || ( val == flipVertex( nextVal ) ) ) 
            {
                valueArr._arr[ idx ] = flipVertex( val ); 
                break; 
            }

            ++nextIdx; 
        }
    }

    return;
}

// Make map AND also update triStar and triStatus array
__global__ void kerMakeOldNewTriMap
(
KerStarData         oldStarData,
int                 oldTriNum,
KerIntArray         newTriMap,
KerIntArray         oldNewMap,
KerIntArray         newTriStar,
KerTriStatusArray   newTriStatus
)
{
    // Iterate through triangles
    for ( int oldTriIdx = getCurThreadIdx(); oldTriIdx < oldTriNum; oldTriIdx += getThreadNum() )
    {
        ////
        // Skip copying free triangle information
        ////

        const TriangleStatus status = oldStarData._triStatusArr[1][ oldTriIdx ]; 

        if ( Free == status )
            continue; 

        ////
        // Make map
        ////

        const int starIdx   = oldStarData._triStarArr[1][ oldTriIdx ];  // Star
        const int oldTriBeg = oldStarData._starTriMap[1][ starIdx ];    // Old location
        const int newTriBeg = newTriMap._arr[ starIdx ];                // New location
        const int newTriIdx = oldTriIdx - oldTriBeg + newTriBeg;        // New location

        oldNewMap._arr[ oldTriIdx ]     = newTriIdx;
        newTriStar._arr[ newTriIdx ]    = starIdx;
        newTriStatus._arr[ newTriIdx ]  = status;
    }

    return;
}

__global__ void kerGetActiveTriCount
(
KerStarData starData,
KerIntArray activeStarArr,
KerIntArray activeTriCountArr
)
{
    // Iterate stars
    for ( int idx = getCurThreadIdx(); idx < activeStarArr._num; idx += getThreadNum() )
    {
        const int star                  = activeStarArr._arr[ idx ];
        const StarInfo starInfo         = starData.getStarInfo( star );
        activeTriCountArr._arr[ idx ]   = starInfo._totalSize; 
    }
    
    return;
}

////////////////////////////////////////////////////////////////////////////////
