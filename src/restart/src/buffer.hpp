/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: buffer.hpp
//  Description: Header for the declaration of the buffer module
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
//  Date:  01/10/2024
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////


#ifndef BUFFER_HPP
#define BUFFER_HPP


#include "vc.hpp"
#include "packet.hpp"
#include "outset.hpp"
#include "routefunc.hpp"
#include "config.hpp"

class Buffer : public Module {

    private:
        int _occupancy;
        int _size;

        std::vector<VC *> _vc;
    
    #ifdef TRACK_BUFFERS
        std::vector<int> _class_occupancy;
    #endif

    public:

        Buffer( const Configuration& config, const SimulationContext& context, const tRoutingParameters& par, int outputs, Module *parent, const std::string& name );
        ~Buffer();

        void addFlit( int vc, Flit *f );

        inline Flit *removeFlit( int vc )
        {
            --_occupancy;
    #ifdef TRACK_BUFFERS
            int cl = _vc[vc]->frontFlit()->cl;
            assert(_class_occupancy[cl] > 0);
            --_class_occupancy[cl];
    #endif
            return _vc[vc]->removeFlit( );
        }
        
        inline Flit *frontFlit( int vc ) const
        {
            return _vc[vc]->frontFlit( );
        }
        
        inline bool empty( int vc ) const
        {
            return _vc[vc]->empty( );
        }

        inline bool full( ) const
        {
            return _occupancy >= _size;
        }

        inline VC::eVCState getState( int vc ) const
        {
            return _vc[vc]->getState( );
        }

        inline void setState( int vc, VC::eVCState s )
        {
            _vc[vc]->setState(s);
        }

        inline const OutSet *getRouteSet( int vc ) const
        {
            return _vc[vc]->getRouteSet( );
        }

        inline void setRouteSet( int vc, OutSet * output_set )
        {
            _vc[vc]->setRouteSet(output_set);
        }

        inline void setOutput( int vc, int out_port, int out_vc )
        {
            _vc[vc]->setOutput(out_port, out_vc);
        }

        inline int getOutputPort( int vc ) const
        {
            return _vc[vc]->getOutputPort( );
        }

        inline int getOutputVC( int vc ) const
        {
            return _vc[vc]->getOutputVC( );
        }

        inline int getPriority( int vc ) const
        {
            return _vc[vc]->getPriority( );
        }

        inline void route( int vc, tRoutingFunction rf, const Router* router, const Flit* f, int in_channel )
        {
            _vc[vc]->route(rf, router, f, in_channel);
        }

        // ==== Debug functions ====

        inline void setWatch( int vc, bool watch = true )
        {
            _vc[vc]->setWatch(watch);
        }

        inline bool isWatched( int vc ) const
        {
            return _vc[vc]->isWatched( );
        }

        inline int getOccupancy( ) const
        {
            return _occupancy;
        }

        inline int getOccupancy( int vc ) const
        {
            return _vc[vc]->getOccupancy( );
        }

        #ifdef TRACK_BUFFERS
        inline int getOccupancyForClass(int c) const
        {
            return _class_occupancy[c];
        }
        #endif

        void display( std::ostream & os ) const;


};


#endif // BUFFER_HPP