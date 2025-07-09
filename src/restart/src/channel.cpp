/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: channel.cpp
//  Description: Source file for the definiton of the channel class
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
//  Date:  01/09/2024
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "channel.hpp"
#include "router.hpp" // line to include the Router class definition

FlitChannel::FlitChannel(Module * parent, const SimulationContext& context, const std::string & name, const int & classes) : Channel<Flit>(parent, context, name), _src_router(nullptr), _snk_router(nullptr), _src_port(-1), _snk_port(-1), _idle(0){_active.resize(classes,0);};

void FlitChannel::setSrcRouter(Router const * router, int port) {
    _src_router = router;
    _src_port = port;
}

void FlitChannel::setSnkRouter(Router const * router, int port) {
    _snk_router = router;
    _snk_port = port;
}

void FlitChannel::send(Flit * flit) {
    if(flit){
        ++_active[flit->cl];
    }
    else{
        ++_idle;
    }

    if (_context->logger && flit &&flit->head) {
        if (_src_router == nullptr) {
            _context->logger->add_tevent_history(flit->rpid, flit->type, _snk_router->GetID(), _snk_router->GetID(), GetSimTime(_context));
        }
        else if (_snk_router == nullptr) {
            _context->logger->add_tevent_history(flit->rpid, flit->type, _src_router->GetID(), _src_router->GetID(), GetSimTime(_context));
        }
        else {
            _context->logger->add_tevent_history(flit->rpid, flit->type, _src_router->GetID(), _snk_router->GetID(), GetSimTime(_context));
        }
    }

    Channel<Flit>::send(flit);
}


Flit * FlitChannel::receive() {

    if (_context->logger && _output_end && _output_end->tail) {
        if (_src_router == nullptr) {
            _context->logger->modify_tevent_history(_output_end->rpid, _output_end->type, _snk_router->GetID(), _snk_router->GetID(), GetSimTime(_context));
        }
        else if (_snk_router == nullptr) {
            _context->logger->modify_tevent_history(_output_end->rpid, _output_end->type, _src_router->GetID(), _src_router->GetID(), GetSimTime(_context));
        }
        else {
            _context->logger->modify_tevent_history(_output_end->rpid, _output_end->type, _src_router->GetID(), _snk_router->GetID(), GetSimTime(_context));
        }
    }
    
    return Channel<Flit>::receive();
}


    

