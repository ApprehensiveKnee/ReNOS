/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: routefunc.hpp
//  Description: source file for the definition of the VC class implementation
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

#include <limits>
#include <sstream>

#include "globals.hpp"
#include "vc.hpp"

const char * const VC::VCSTATE[] = {"idle",
				    "routing",
				    "vc_alloc",
				    "active"};

VC::VC( const Configuration& config, const SimulationContext& context, const tRoutingParameters& par, int outputs, 
	Module *parent, const std::string& name )
  : Module( parent, name ),_par(&par),_context(&context), 
    _state(idle), _out_port(-1), _out_vc(-1), _pri(0), _watched(false), 
    _expected_pid(-1), _last_id(-1), _last_pid(-1)
{
  _lookahead_routing = !config.getIntField("routing_delay");
  _route_set = _lookahead_routing ? NULL : new OutSet( );

  std::string priority = config.getStrField( "priority" ); // set the priority policy
  if ( priority == "local_age" ) {
    _pri_type = local_age_based;
  } else if ( priority == "queue_length" ) {
    _pri_type = queue_length_based;
  } else if ( priority == "hop_count" ) {
    _pri_type = hop_count_based;
  } else if ( priority == "none" ) {
    _pri_type = none;
  } else {
    _pri_type = other;
  }

  _priority_donation = config.getIntField("vc_priority_donation");
}

VC::~VC()
{
  if(!_lookahead_routing) {
    delete _route_set;
  }
}

void VC::addFlit( Flit *f )
{
  assert(f); // first assert the flit pointer is not null

  if(_expected_pid >= 0) {
    if(f->pid != _expected_pid) {
      std::ostringstream err;
      err << "Received flit " << f->id << " with unexpected packet ID: " << f->pid 
	  << " (expected: " << _expected_pid << ")";
      error(err.str());
    } else if(f->tail) {
      _expected_pid = -1; // if the flit received is tail, reset the cached pid on channel
    }
  } else if(!f->tail) {
    _expected_pid = f->pid; //if the packet id associated to the flit receive is different than the cached one, refresh
  }
    
  // update flit priority before adding to VC buffer
  if(_pri_type == local_age_based) {
    f->priority = std::numeric_limits<int>::max() - GetSimTime(_context);
    assert(f->priority >= 0);
  } else if(_pri_type == hop_count_based) {
    f->priority = f->hops;
    assert(f->priority >= 0);
  }

  _buffer.push_back(f);
  updatePriority();
}

Flit *VC::removeFlit( )
{
  Flit *f = NULL;
  if ( !_buffer.empty( ) ) {
    f = _buffer.front( );
    _buffer.pop_front( );
    _last_id = f->id;
    _last_pid = f->pid;
    updatePriority();
  } else {
    error("Trying to remove flit from empty buffer.");
  }
  return f;
}



void VC::setState( eVCState s )
{
  Flit * f = frontFlit();
  
  if(f && f->watch)
    *(_context->gWatchOut) << GetSimTime(_context) << " | " << getFullName() << " | "
		<< "Changing state from " << VC::VCSTATE[_state]
		<< " to " << VC::VCSTATE[s] << "." << std::endl;
  
  _state = s;
}

const OutSet *VC::getRouteSet( ) const
{
  return _route_set;
}

void VC::setRouteSet( OutSet * output_set )
{
  _route_set = output_set;
  _out_port = -1;
  _out_vc = -1;
}

void VC::setOutput( int port, int vc )
{
  _out_port = port;
  _out_vc   = vc;
}

void VC::updatePriority()
{
  if(_buffer.empty()) return;
  if(_pri_type == queue_length_based) {
    _pri = _buffer.size();
  } else if(_pri_type != none) {
    Flit * f = _buffer.front();
    if((_pri_type != local_age_based) && _priority_donation) {
      Flit * df = f;
      for(size_t i = 1; i < _buffer.size(); ++i) {
        Flit * bf = _buffer[i];
        //seach for higher priority flits in the buffer
        if(bf->priority > df->priority) df = bf; 
      }
      if((df != f) && (df->watch || f->watch)) {
	*(_context->gWatchOut) << GetSimTime(_context) << " | " << getFullName() << " | "
		    << "Flit " << df->id
		    << " donates priority to flit " << f->id
		    << "." << std::endl;
      }
      f = df;
    }
    if(f->watch)
      *(_context->gWatchOut) << GetSimTime(_context) << " | " << getFullName() << " | "
		  << "Flit " << f->id
		  << " sets priority to " << f->priority
		  << "." << std::endl;
    _pri = f->priority;
  }
}


void VC::route( tRoutingFunction rf, const Router* router,  const Flit* f, int in_channel )
{
  rf(_context, router, _par, f, in_channel, _route_set, false );
  _out_port = -1;
  _out_vc = -1;
}

// ==== Debug functions ====

void VC::setWatch( bool watch )
{
  _watched = watch;
}

bool VC::isWatched( ) const
{
  return _watched;
}

void VC::display( std::ostream & os ) const
{
  if ( _state != VC::idle ) {
    os << getFullName() << ": "
       << " state: " << VCSTATE[_state];
    if(_state == VC::active) {
      os << " out_port: " << _out_port
	 << " out_vc: " << _out_vc;
    }
    os << " fill: " << _buffer.size();
    if(!_buffer.empty()) {
      os << " front: " << _buffer.front()->id;
    }
    os << " pri: " << _pri;
    os << std::endl;
  }
}



