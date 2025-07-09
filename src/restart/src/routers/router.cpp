/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: router.cpp
//  Description: Src file for the definition of the Router class
//                  
//               Great inspiration taken from the Booksim2 NoC simulator (https://github.com/booksim/booksim2)
//               Copyright (c) 2007-2015, Trustees of The Leland Stanford Junior University
//               All rights reserved.
//               Great inspiration taken from the Booksim2 NoC simulator (https://github.com/booksim/booksim2)
//  Created by:  Edoardo Cabiati
//  Date:  03/10/2024
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////


#include "router.hpp"

//////////////////Sub router types//////////////////////
#include "iq_router.hpp"
#include "event_router.hpp"
#include "chaos_router.hpp"
///////////////////////////////////////////////////////

#include <iostream>
#include <cassert>


int const Router::STALL_BUFFER_BUSY = -2;
int const Router::STALL_BUFFER_CONFLICT = -3;
int const Router::STALL_BUFFER_FULL = -4;
int const Router::STALL_BUFFER_RESERVED = -5;
int const Router::STALL_CROSSBAR_CONFLICT = -6;

Router::Router( const Configuration& config,
    const SimulationContext& context,
    const tRoutingParameters& par,
		Module *parent, const std::string & name, int id,
		int inputs, int outputs ) :
TimedModule( parent, name ), par(&par), context(&context), _id( id ), _inputs( inputs ), _outputs( outputs ),
   _partial_internal_cycles(0.0)
{
  _crossbar_delay   = ( config.getIntField( "st_prepare_delay" ) + 
			config.getIntField( "st_final_delay" ) );
  _credit_delay     = config.getIntField( "credit_delay" );
  _input_speedup    = config.getIntField( "input_speedup" );
  _output_speedup   = config.getIntField( "output_speedup" );
  _internal_speedup = config.getFloatField( "internal_speedup" );
  _classes          = config.getIntField( "classes" );

#ifdef TRACK_FLOWS
  _received_flits.resize(_classes, std::vector<int>(_inputs, 0));
  _stored_flits.resize(_classes);
  _sent_flits.resize(_classes, std::vector<int>(_outputs, 0));
  _active_packets.resize(_classes);
  _outstanding_credits.resize(_classes, std::vector<int>(_outputs, 0));
#endif

#ifdef TRACK_STALLS
  _buffer_busy_stalls.resize(_classes, 0);
  _buffer_conflict_stalls.resize(_classes, 0);
  _buffer_full_stalls.resize(_classes, 0);
  _buffer_reserved_stalls.resize(_classes, 0);
  _crossbar_conflict_stalls.resize(_classes, 0);
#endif

}

void Router::AddInputChannel( FlitChannel *channel, CreditChannel *backchannel )
{
  _input_channels.push_back( channel );
  _input_credits.push_back( backchannel );
  channel->setSnkRouter( this, _input_channels.size() - 1 ) ;
}

void Router::AddOutputChannel( FlitChannel *channel, CreditChannel *backchannel )
{
  _output_channels.push_back( channel );
  _output_credits.push_back( backchannel );
  _channel_faults.push_back( false );
  channel->setSrcRouter(this, _output_channels.size() - 1 ) ;
}

void Router::evaluate( )
{
  _partial_internal_cycles += _internal_speedup;
  while( _partial_internal_cycles >= 1.0 ) {
    _InternalStep();
    _partial_internal_cycles -= 1.0;
  }
}

void Router::OutChannelFault( int c, bool fault )
{
  assert( ( c >= 0 ) && ( (size_t)c < _channel_faults.size( ) ) );
  _channel_faults[c] = fault;
}

bool Router::IsFaultyOutput( int c ) const
{
  assert( ( c >= 0 ) && ( (size_t)c < _channel_faults.size( ) ) );
  return _channel_faults[c];
}

/*Router constructor*/
Router *Router::NewRouter( const Configuration& config,
         const SimulationContext& context,
         const tRoutingParameters& par,
			   Module *parent, const std::string & name, int id,
			   int inputs, int outputs )
{
  const std::string type = config.getStrField( "router" );
  Router *r = NULL;
  if ( type == "iq" ) {
    r = new IQRouter( config, context, par, parent, name, id, inputs, outputs );
  } else if ( type == "event" ) {
    r = new EventRouter( config, context, par, parent, name, id, inputs, outputs );
  } else if ( type == "chaos" ) {
    r = new ChaosRouter( config, context, par, parent, name, id, inputs, outputs );
  } else {
    std::cerr << "Unknown router type: " << type << std::endl;
  }
  /*For additional router, add another else if statement*/
  /*Original booksim specifies the router using "flow_control"
   *we now simply call these types. 
   */

  return r;
}


