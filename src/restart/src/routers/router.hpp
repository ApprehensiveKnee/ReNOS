/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: router.hpp
//  Description: Header for the declaration of the Router class
//                  
//               Great inspiration taken from the Booksim2 NoC simulator (https://github.com/booksim/booksim2)
//               Copyright (c) 2007-2015, Trustees of The Leland Stanford Junior University
//               All rights reserved.
//               Great inspiration taken from the Booksim2 NoC simulator (https://github.com/booksim/booksim2)
//  Created by:  Edoardo Cabiati
//  Date:  03/10/2024
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////


#ifndef ROUTER_HPP
#define ROUTER_HPP

#include <vector>
#include <string>

#include "base.hpp"
#include "channel.hpp"
#include "packet.hpp"
#include "config.hpp"



class Router : public TimedModule {

    public:
        const SimulationContext * context;
        const tRoutingParameters * par;

    protected:
        static int const STALL_BUFFER_BUSY;
        static int const STALL_BUFFER_CONFLICT;
        static int const STALL_BUFFER_FULL;
        static int const STALL_BUFFER_RESERVED;
        static int const STALL_CROSSBAR_CONFLICT;

        int _id;

        int _inputs; // number of input ports
        int _outputs; // number of output ports

        int _classes;

        int _input_speedup;
        int _output_speedup;

        double _internal_speedup;
        double _partial_internal_cycles;

        int _crossbar_delay;
        int _credit_delay;

        // each router stores a set of input and output channels
        std::vector<FlitChannel *>   _input_channels;
        std::vector<CreditChannel *> _input_credits;
        std::vector<FlitChannel *>   _output_channels;
        std::vector<CreditChannel *> _output_credits;
        std::vector<bool>            _channel_faults;

        #ifdef TRACK_FLOWS
        vector<vector<int> > _received_flits;
        vector<vector<int> > _stored_flits;
        vector<vector<int> > _sent_flits;
        vector<vector<int> > _outstanding_credits;
        vector<vector<int> > _active_packets;
        #endif

        #ifdef TRACK_STALLS
        vector<int> _buffer_busy_stalls;
        vector<int> _buffer_conflict_stalls;
        vector<int> _buffer_full_stalls;
        vector<int> _buffer_reserved_stalls;
        vector<int> _crossbar_conflict_stalls;
        #endif

        virtual void _InternalStep() = 0;  // overloaded by derived router class

    public:
        Router( const Configuration& config,
                const SimulationContext& context,
                const tRoutingParameters & par,
                Module *parent, const std::string & name, int id,
                int inputs, int outputs);

        static Router *NewRouter( const Configuration& config,
                const SimulationContext& context,
                const tRoutingParameters & par,   
			    Module *parent, const std::string & name, int id,
			    int inputs, int outputs );

        virtual void AddInputChannel( FlitChannel *channel, CreditChannel *backchannel );
        virtual void AddOutputChannel( FlitChannel *channel, CreditChannel *backchannel );
        
        inline FlitChannel * GetInputChannel( int input ) const {
            assert((input >= 0) && (input < _inputs));
            return _input_channels[input];
        }
        inline FlitChannel * GetOutputChannel( int output ) const {
            assert((output >= 0) && (output < _outputs));
            return _output_channels[output];
        }

        virtual void readInputs( ) = 0;
        virtual void evaluate( );
        virtual void writeOutputs( ) = 0;

        void OutChannelFault( int c, bool fault = true );
        bool IsFaultyOutput( int c ) const;

        inline int GetID( ) const {return _id;}


        virtual int GetUsedCredit(int o) const = 0;
        virtual int GetBufferOccupancy(int i) const = 0;

        #ifdef TRACK_BUFFERS
        virtual int GetUsedCreditForClass(int output, int cl) const = 0;
        virtual int GetBufferOccupancyForClass(int input, int cl) const = 0;
        #endif

        #ifdef TRACK_FLOWS
        inline vector<int> const & GetReceivedFlits(int c) const {
            assert((c >= 0) && (c < _classes));
            return _received_flits[c];
        }
        inline vector<int> const & GetStoredFlits(int c) const {
            assert((c >= 0) && (c < _classes));
            return _stored_flits[c];
        }
        inline vector<int> const & GetSentFlits(int c) const {
            assert((c >= 0) && (c < _classes));
            return _sent_flits[c];
        }
        inline vector<int> const & GetOutstandingCredits(int c) const {
            assert((c >= 0) && (c < _classes));
            return _outstanding_credits[c];
        }

        inline vector<int> const & GetActivePackets(int c) const {
            assert((c >= 0) && (c < _classes));
            return _active_packets[c];
        }

        inline void ResetFlowStats(int c) {
            assert((c >= 0) && (c < _classes));
            _received_flits[c].assign(_received_flits[c].size(), 0);
            _sent_flits[c].assign(_sent_flits[c].size(), 0);
        }
        #endif

        virtual std::vector<int> UsedCredits() const = 0;
        virtual std::vector<int> FreeCredits() const = 0;
        virtual std::vector<int> MaxCredits() const = 0;

        #ifdef TRACK_STALLS
        inline int GetBufferBusyStalls(int c) const {
            assert((c >= 0) && (c < _classes));
            return _buffer_busy_stalls[c];
        }
        inline int GetBufferConflictStalls(int c) const {
            assert((c >= 0) && (c < _classes));
            return _buffer_conflict_stalls[c];
        }
        inline int GetBufferFullStalls(int c) const {
            assert((c >= 0) && (c < _classes));
            return _buffer_full_stalls[c];
        }
        inline int GetBufferReservedStalls(int c) const {
            assert((c >= 0) && (c < _classes));
            return _buffer_reserved_stalls[c];
        }
        inline int GetCrossbarConflictStalls(int c) const {
            assert((c >= 0) && (c < _classes));
            return _crossbar_conflict_stalls[c];
        }

        inline void ResetStallStats(int c) {
            assert((c >= 0) && (c < _classes));
            _buffer_busy_stalls[c] = 0;
            _buffer_conflict_stalls[c] = 0;
            _buffer_full_stalls[c] = 0;
            _buffer_reserved_stalls[c] = 0;
            _crossbar_conflict_stalls[c] = 0;
        }
        #endif

        inline int NumInputs() const {return _inputs;}
        inline int NumOutputs() const {return _outputs;}

    };

 #endif // ROUTER_HPP