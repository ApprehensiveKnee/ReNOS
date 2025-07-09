/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: routefunc.hpp
//  Description: header file for the declaration of the routing functions
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

#ifndef ROUTEFUNC_HPP
#define ROUTEFUNC_HPP

#include "packet.hpp"
#include "router.hpp"
#include "outset.hpp"
#include "config.hpp"
#include "globals.hpp"

typedef void (*tRoutingFunction)( const SimulationContext *, const Router *, const tRoutingParameters *, const Flit *, int in_channel, OutSet *, bool );

// define a new class to hold all of the routing parameters
class tRoutingParameters {
    public:
        std::map<std::string, tRoutingFunction> gRoutingFunctionMap;

        int gNumVCs;
        int gReadReqBeginVC, gReadReqEndVC;
        int gWriteReqBeginVC, gWriteReqEndVC;
        int gReadReplyBeginVC, gReadReplyEndVC;
        int gWriteReplyBeginVC, gWriteReplyEndVC;

        tRoutingParameters():
            gRoutingFunctionMap(),
            gNumVCs(0),
            gReadReqBeginVC(0), gReadReqEndVC(0),
            gWriteReqBeginVC(0), gWriteReqEndVC(0),
            gReadReplyBeginVC(0), gReadReplyEndVC(0),
            gWriteReplyBeginVC(0), gWriteReplyEndVC(0)
        {}

};

tRoutingParameters initializeRoutingMap( const Configuration & config);



#endif //ROUTEFUNC_HPP
