/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: routefunc.hpp
//  Description: header file for the declaration of the VC class implementation
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
//  Date:  11/10/2024
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef VC_HPP
#define VC_HPP

#include <deque>
#include <climits>

#include "base.hpp"
#include "packet.hpp"
#include "outset.hpp"
#include "config.hpp"
#include "routefunc.hpp"

class VC : public Module {

    public:
        enum eVCState {
            state_min = 0,
            idle = state_min, 
            routing,
            vc_alloc,
            active, 
		    state_max = active 
        };
        struct state_info_t {
            int cycles;
        };
        static const char * const VCSTATE[];

    private:

        std::deque<Flit *> _buffer;  // buffer of flits for the VC

        eVCState _state;  // state of the virtual channel

        const SimulationContext * _context;
        const tRoutingParameters * _par;

        OutSet * _route_set;
        int _out_port, _out_vc;

        enum ePrioType { local_age_based, queue_length_based, hop_count_based, none, other };

        ePrioType _pri_type;

        int _pri;

        int _priority_donation;

        bool _watched;

        int _expected_pid;

        int _last_id;
        int _last_pid;

        bool _lookahead_routing;

        public:
        
        VC( const Configuration& config, const SimulationContext& context, const tRoutingParameters& par, int outputs,
            Module *parent, const std::string& name );
        ~VC();

        void addFlit( Flit *f );
        inline Flit *frontFlit( ) const
        {
            return _buffer.empty() ? NULL : _buffer.front();
        }
        
        Flit *removeFlit( );
        
        
        inline bool empty( ) const
        {
            return _buffer.empty( );
        }

        inline VC::eVCState getState( ) const
        {
            return _state;
        }


        void setState( eVCState s );

        const OutSet *getRouteSet( ) const;
        void setRouteSet( OutSet * output_set );

        void setOutput( int port, int vc );

        inline int getOutputPort( ) const
        {
            return _out_port;
        }


        inline int getOutputVC( ) const
        {
            return _out_vc;
        }

        void updatePriority();
        
        inline int getPriority( ) const
        {
            return _pri;
        }
        void route( tRoutingFunction rf, const Router* router, const Flit* f, int in_channel );

        inline int getOccupancy() const
        {
            return (int)_buffer.size();
        }

        // ==== Debug functions ====

        void setWatch( bool watch = true );
        bool isWatched( ) const;
        void display( std::ostream & os ) const;



};



#endif // VC_HPP