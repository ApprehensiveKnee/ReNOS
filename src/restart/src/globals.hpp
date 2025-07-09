/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: globals.hpp
//  Description: header file for the declaration of some global variables
//                  
//               Great inspiration taken from the Booksim2 NoC simulator (https://github.com/booksim/booksim2)
//               Copyright (c) 2007-2015, Trustees of The Leland Stanford Junior University
//               All rights reserved.
//
//               Redistribution and use in source and binary forms, with or without
//               modification, are permitted provided that the following conditions are met:
//
//               Redistributions of source code must retain the above copyright notice, this 
//               list of conditions and the following disclaimer.
//               Redistributions in binary form must reproduce the above copyright notice, this
//               list of conditions and the following disclaimer in the documentation and/or
//               other materials provided with the distribution.
//
//               THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
//               ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
//               WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
//               DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
//               ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//               (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//               LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
//               ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//               (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//               SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//  Created by:  Edoardo Cabiati
//  Date:  09/10/2024
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////


#ifndef GLOBALS_HPP
#define GLOBALS_HPP

#include <iostream>
#include <string>
#include <vector>
#include <iostream>
#include "logger.hpp"

/* forward definition of traffic manager*/
class TrafficManager;
class tRoutingParameters;

struct NullBuf
    : std::streambuf {
    char buffer[100];
    int overflow(int c) override {
        setp(buffer, buffer + sizeof buffer);
        return c;
    }
    std::streamsize xsputn(const char*, std::streamsize n) override {
        return n;
    }
};

class NullStream : public std::ostream {
public:
    NullStream() : std::ostream(&m_sb) {}
private:
    NullBuf m_sb;
};


class SimulationContext {
  public:
    bool gPrintActivity;
    int gK; // radix
    int gN; // dimension
    int gC; // concentration
    int gNodes;
    bool gTrace;
    std::ostream *gWatchOut;
    // also defined a file to write the output
    std::ostream *gDumpFile;
    // a logger object to later plot the activity of the network
    EventLogger * logger;

    TrafficManager *trafficManager;

    NullStream nullStream;

    SimulationContext()
        : gPrintActivity(false), gK(0), gN(0), gC(0), gNodes(0), gTrace(false), gWatchOut(NULL), gDumpFile(NULL), logger(NULL), trafficManager(nullptr) {}

    void setTrafficManager(TrafficManager *tm) {
        trafficManager = tm;
    };
    
    void setLogger(EventLogger * log) {
        logger = log;
    };

    void PrintEvents( std::ostream & os = std::cout ) const {
        if (logger) {
            logger->print_events(os);
        }
        else {
        os << "------------------------------\n";
        os << "PrintEvents() is not available\n";
        os << "------------------------------";
        }
    };


};


/* to be declared in main.cpp */
int GetSimTime(const SimulationContext* context);

class Stats;
Stats * GetStats(const std::string & name, const SimulationContext* context);




#endif