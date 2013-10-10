/*
Author: Ashwin Nanjappa
Filename: Main.cpp

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
//                                     Main
////////////////////////////////////////////////////////////////////////////////

#include "Application.h"
#include "Config.h"

int main( int argc, const char* argv[] )
{
    ////
    // Set default configuration
    ////

    Config& config      = getConfig();
    config._run         = 0;
    config._runNum      = 1;
    config._gridSize    = 512;
    config._pointNum    = 100000;
    config._dist        = UniformDistribution;
    config._facetMax    = -1;   // Set below
    config._weightMax   = 1;
    config._logVerbose  = false;
    config._logStats    = false;
    config._logTiming   = false;
    config._doCheck     = false;
    config._inFile      = false;

    ////
    // Parse input arguments
    ////

    cout << "Syntax: greg3d [-n PointNum] [-g GridSize] [-s Seed] [-r Max] [-d Distribution] [-f FacetMax] [-w WeightMax] [-verbose] [-stats] [-timing] [-check]" << endl; 
    cout << "Distribution: 0: Uniform, 1: Gaussian, 2: Ball, 3: Sphere, 4: Grid" << endl << endl;

    int idx = 1;

    while ( idx < argc )
    {
        if ( 0 == string( "-n" ).compare( argv[ idx ] ) )
        {
            config._pointNum = atoi( argv[ idx + 1 ] ); 
            ++idx; 
        }
        else if ( 0 == string( "-g" ).compare( argv[ idx ] ) )
        {
            config._gridSize = atoi( argv[ idx + 1 ] ); 
            ++idx; 
        }
        else if ( 0 == string( "-r" ).compare( argv[ idx ] ) )
        {
            config._runNum = atoi( argv[ idx + 1 ] ); 
            ++idx; 
        }
        else if ( 0 == string( "-s" ).compare( argv[ idx ] ) )
        {
            config._run = atoi( argv[ idx + 1 ] ); 
            ++idx; 
        }
        else if ( 0 == string( "-f" ).compare( argv[ idx ] ) )
        {
            config._facetMax = atoi( argv[ idx + 1 ] ); 
            ++idx; 
        }
        else if ( 0 == string( "-d" ).compare( argv[ idx ] ) )
        {
            const int distVal = atoi( argv[ idx + 1 ] );
            config._dist      = ( Distribution ) distVal;
            ++idx; 
        }
        else if ( 0 == string( "-verbose" ).compare( argv[ idx ] ) )
        {
            config._logVerbose = true;
        }
        else if ( 0 == string( "-stats" ).compare( argv[ idx ] ) )
        {
            config._logStats = true;
        }
        else if ( 0 == string( "-timing" ).compare( argv[ idx ] ) )
        {
            config._logTiming = true;
        }
        else if ( 0 == string( "-check" ).compare( argv[ idx ] ) )
        {
            config._doCheck = true;
        }
        else if ( 0 == string( "-w" ).compare( argv[ idx ] ) )
        {
            config._weightMax = atoi( argv[ idx + 1 ] ); 
            ++idx; 
        }
        else if ( 0 == std::string( "-inFile" ).compare( argv[ idx ] ) )
        {
            config._inFile     = true;
            config._inFilename = std::string( argv[ idx + 1 ] );

            ++idx;
        }
        else
        {
            cerr << "Cannot understand input argument: " << argv[ idx ] << endl;
            exit( 1 );
        }

        ++idx; 
    }

    // Adjust run
    if ( config._runNum <= config._run )
        config._runNum = config._run + 1;

    ////
    // Adjust facetMax
    ////

    const int FacetMaxUniform    = 12000000;
    const int FacetMaxNonUniform = 5000000;

    // Check if facet bound is not set
    if ( -1 == config._facetMax )
    {
        if ( UniformDistribution == config._dist )
        {
            if ( config._weightMax <= 1 )
                config._facetMax = FacetMaxUniform;
            else if ( config._weightMax <= 10 )
                config._facetMax = 10000000;
            else if ( config._weightMax <= 100 )
                config._facetMax = 5000000;
            else
                config._facetMax = 4000000;
        }
        else
        {
            config._facetMax = FacetMaxNonUniform;
        }
    }

    cout << "gReg3D running ...";
    cout << " Seed: "       << config._run;
    cout << " RunNum: "     << config._runNum;
    cout << " Grid: "       << config._gridSize;
    cout << " Points: "     << config._pointNum;
    cout << " Input: "      << ( config._inFile ? config._inFilename : DistStr[ config._dist ] );
    cout << " Facets: "     << config._facetMax;
    cout << " Max weight: " << config._weightMax;
    cout << endl;

    ////
    // Compute Delaunay
    ////

    Application app;
    app.doRun();

    return 0;
}

Config& getConfig()
{
    static Config _config;
    return _config;
}

////////////////////////////////////////////////////////////////////////////////
