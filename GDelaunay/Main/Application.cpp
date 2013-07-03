/*
Author: Ashwin Nanjappa
Filename: Application.cpp

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
//                                 Application
////////////////////////////////////////////////////////////////////////////////

// Self
#include "Application.h"

// Project
#include "Config.h"
#include "DtRandom.h"
#include "GDelaunay.h"
#include "STLWrapper.h"
#include "PerfTimer.h"

void Application::doRun()
{
    Config& config = getConfig();

    assert( ( config._run >= 0 ) && ( config._run < config._runNum ) && "Invalid run numbers!" );

    ////
    // Run Delaunay iterations
    ////

    double pbaTimeSum   = 0.0;
    double starTimeSum  = 0.0;
    double splayTimeSum = 0.0;
    double outTimeSum   = 0.0;
    double gregTimeSum  = 0.0;

    for ( ; config._run < config._runNum; ++config._run )
    {
        _init();
        _doRun( pbaTimeSum, starTimeSum, splayTimeSum, outTimeSum, gregTimeSum );
        _deInit();
    }

    // Write time to file

    ofstream logFile( "greg-time.txt", ofstream::app );
    assert( logFile && "Could not open time file!" );

    logFile << "GridSize,"   << config._gridSize  << ",";
    logFile << "Runs,"       << config._runNum    << ",";
    logFile << "Points,"     << config._pointNum  << ",";
    logFile << "WeightMax,"  << config._weightMax << ",";
    logFile << "Input,"      << ( config._inFile ? config._inFilename : DistStr[ config._dist ] ) << ",";
    logFile << "Total Time," << gregTimeSum  / ( config._runNum * 1000.0 ) << ",";
    logFile << "PBA Time,"   << pbaTimeSum   / ( config._runNum * 1000.0 ) << ",";
    logFile << "Star Time,"  << starTimeSum  / ( config._runNum * 1000.0 ) << ",";
    logFile << "Splay Time," << splayTimeSum / ( config._runNum * 1000.0 ) << ",";
    logFile << "Out Time,"   << outTimeSum   / ( config._runNum * 1000.0 ) << endl;

    return;
}

void Application::_init()
{
    // Pick the best CUDA device
    const int deviceIdx = cutGetMaxGflopsDeviceId();
    CudaSafeCall( cudaSetDevice( deviceIdx ) );

    // CUDA configuration
    CudaSafeCall( cudaDeviceSetCacheConfig( cudaFuncCachePreferShared ) );

    return;
}

void Application::_doRun
(
double& pbaTimeSum,
double& starTimeSum,
double& splayTimeSum,
double& outTimeSum,
double& gregTimeSum
)
{
    Config& config = getConfig();

    cout << "Run: " << config._run << endl;

    // Get points and weights

    if ( config._inFile ) readPoints();
    else                  makePoints();

    makeWeights();

    // Initialize

    gdelInit( config, _pointVec, _weightVec );

    // Compute Delaunay

    double timePba, timeInitialStar, timeConsistent, timeOutput; 

    HostTimer timerAll;
    timerAll.start();
        gdelCompute( timePba, timeInitialStar, timeConsistent, timeOutput );
    timerAll.stop();

    const double timeTotal = timerAll.value();

    cout << "PBA:         " << timePba         << endl; 
    cout << "InitStar:    " << timeInitialStar << endl;
    cout << "Consistency: " << timeConsistent  << endl;
    cout << "StarOutput:  " << timeOutput      << endl;
    cout << "TOTAL Time:  " << timeTotal       << endl;

    pbaTimeSum   += timePba;
    starTimeSum  += timeInitialStar;
    splayTimeSum += timeConsistent;
    outTimeSum   += timeOutput;
    gregTimeSum  += timeTotal;

    // Check

    if ( config._doCheck )
    {
        TetraMesh tetMesh;
        tetMesh.setPoints( _pointVec, _weightVec );
        getTetraFromGpu( tetMesh );
        tetMesh.check();
    }

    // Destroy

    gdelDeInit();
    _pointVec.clear();
    _weightVec.clear();

    return;
}

void Application::_deInit()
{
    CudaSafeCall( cudaDeviceReset() );

    return;
}

RealType Application::scalePoint( RealType gridWidth, float minVal, float maxVal, RealType inVal )
{
    // Translate
    inVal = inVal - minVal; // MinVal can be -ve
    assert( inVal >= 0 );

    // Scale
    const float rangeVal = maxVal - minVal;
    inVal = ( gridWidth - 3.0f ) * inVal / rangeVal;
    assert( inVal <= ( gridWidth - 2 ) );

    inVal += 1.0f;

    return inVal;
}

void Application::readPoints()
{
    assert( _pointVec.empty() && "Input point vector not empty!" );

    Config& config = getConfig();

    std::ifstream inFile( config._inFilename.c_str() );

    if ( !inFile )
    {
        std::cout << "Error opening input file: " << config._inFilename << " !!!\n";
        exit( 1 );
    }

    ////
    // Read input points
    ////

    std::string strVal;
    Point3 point;
    Point3HVec inPointVec;
    Point3Set pointSet;
    int idx         = 0;
    int orgCount    = 0;
    float val       = 0.0f;
    float minVal    = 999.0f;
    float maxVal    = -999.0f;

    while ( inFile >> strVal )
    {
        std::istringstream iss( strVal );

        // Read a coordinate
        iss >> val;
        point._p[ idx ] = val;
        ++idx;

        // Compare bounds
        if ( val < minVal ) minVal = val;
        if ( val > maxVal ) maxVal = val;

        // Read a point
        if ( 3 == idx )
        {
            idx = 0;

            ++orgCount;

            // Check if point unique
            if ( pointSet.end() == pointSet.find( point ) )
            {
                pointSet.insert( point );
                inPointVec.push_back( point );
            }
        }
    }

    ////
    // Check for duplicate points
    ////

    const int dupCount = orgCount - ( int ) inPointVec.size();

    if ( dupCount > 0 )
    {
        std::cout << dupCount << " duplicate points in input file!" << std::endl;
    }

    ////
    // Scale points and store them
    ////

    pointSet.clear();

    // Iterate input points
    for ( int ip = 0; ip < ( int ) inPointVec.size(); ++ip )
    {
        Point3& inPt = inPointVec[ ip ];

        // Iterate coordinates
        for ( int vi = 0; vi < 3; ++vi )
        {
            const RealType inVal  = inPt._p[ vi ];
            const RealType outVal = scalePoint( ( RealType ) config._gridSize, minVal, maxVal, inVal );
            inPt._p[ vi ]         = outVal;
        }

        // Check if point unique
        if ( pointSet.end() == pointSet.find( inPt ) )
        {
            pointSet.insert( inPt );
            _pointVec.push_back( inPt );
        }
    }

    // Update
    config._pointNum = _pointVec.size();

    cout << "Unique points: " << _pointVec.size() << endl;

    return;
}

void Application::makePoints()
{
    Config& config = getConfig();

    assert( _pointVec.empty() && "Input point vector not empty!" );
    assert( _weightVec.empty() && "Input weight vector not empty!" );

    ////
    // Initialize seed
    ////

    // Points in range [1..width-2]
    const int minWidth = 1;
    const int maxWidth = config._gridSize - 2;

    DtRandom randGen;

    switch ( config._dist )
    {
    case UniformDistribution:
    case GaussianDistribution:
    case GridDistribution:
        randGen.init( config._run, minWidth, maxWidth );
        break;
    case BallDistribution:
    case SphereDistribution:
        randGen.init( config._run, 0, 1 );
        break;
    default:
        assert( false );
        break;
    }

    ////
    // Generate points
    ////
    
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    Point3Set pointSet;

    for ( int i = 0; i < config._pointNum; ++i )
    {
        bool uniquePoint = false;

        // Loop until we get unique point
        while ( !uniquePoint )
        {
            switch ( config._dist )
            {
            case UniformDistribution:
                {
                    x = randGen.getNext();
                    y = randGen.getNext();
                    z = randGen.getNext();
                }
                break;

            case BallDistribution:
                {
                    RealType d;

                    do
                    {
                        x = randGen.getNext() - 0.5f; 
                        y = randGen.getNext() - 0.5f; 
                        z = randGen.getNext() - 0.5f; 
                    
                        d = x * x + y * y + z * z;
                        
                    } while ( d > 0.45f * 0.45f );

                    x += 0.5f;
                    y += 0.5f;
                    z += 0.5f;
                    x *= maxWidth;
                    y *= maxWidth;
                    z *= maxWidth;
                }
                break;

            case SphereDistribution:
                {
                    RealType d;

                    do
                    {
                        x = randGen.getNext() - 0.5f; 
                        y = randGen.getNext() - 0.5f; 
                        z = randGen.getNext() - 0.5f; 
                    
                        d = x * x + y * y + z * z;
                        
                    } while ( d > ( 0.45f * 0.45f ) || d < ( 0.4f * 0.4f ) );

                    x += 0.5f;
                    y += 0.5f;
                    z += 0.5f;
                    x *= maxWidth;
                    y *= maxWidth;
                    z *= maxWidth;
                }
                break;

            case GaussianDistribution:
                {
                    randGen.nextGaussian( x, y, z);
                }
                break;

            case GridDistribution:
                {
                    float v[3];

                    for ( int i = 0; i < 3; ++i )
                    {
                        const float val     = randGen.getNext();
                        const float frac    = val - floor( val );
                        v[ i ]              = ( frac < 0.5f ) ? floor( val ) : ceil( val );
                    }

                    x = v[0];
                    y = v[1];
                    z = v[2];
                }
                break;
               
            default:
                {
                    assert( false );
                }
                break;
            }

            // Adjust to bounds
            if ( floor( x ) >= maxWidth )   x -= 1.0f;
            if ( floor( y ) >= maxWidth )   y -= 1.0f;
            if ( floor( z ) >= maxWidth )   z -= 1.0f;

            const Point3 point = { x, y, z };

            if ( pointSet.end() == pointSet.find( point ) )
            {
                pointSet.insert( point );
                _pointVec.push_back( point );

                uniquePoint = true;
            }
        }
    }

    return;
}

void Application::makeWeights()
{
    Config& config = getConfig();

    DtRandom weightGen;
    weightGen.init( config._run, 0, config._weightMax );

    for ( int pi = 0; pi < config._pointNum; ++pi )
    {
        // For testing Delaunay with zero weights
        if ( config._weightMax <= 0 )
        {
            _weightVec.push_back( 0 );
        }
        else
        {
            const Weight w = weightGen.getNext();
            _weightVec.push_back( w );
        }
    }

    return;
}

////////////////////////////////////////////////////////////////////////////////
