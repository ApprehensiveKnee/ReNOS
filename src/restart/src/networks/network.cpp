// $Id$

/*
 Copyright (c) 2007-2015, Trustees of The Leland Stanford Junior University
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 Redistributions of source code must retain the above copyright notice, this 
 list of conditions and the following disclaimer.
 Redistributions in binary form must reproduce the above copyright notice, this
 list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*network.cpp
 *
 *This class is the basis of the entire network, it contains, all the routers
 *channels in the network, and is extended by all the network topologies
 *
 */

#include <cassert>
#include <sstream>

#include "network.hpp"
#include "routefunc.hpp"
#include "kncube.hpp"
#include "fly.hpp"
#include "cmesh.hpp"
#include "flatfly_onchip.hpp"
#include "qtree.hpp"
#include "tree4.hpp"
#include "fattree.hpp"
#include "anynet.hpp"
#include "dragonfly.hpp"


Network::Network( const Configuration &config, SimulationContext& context, tRoutingParameters& par, const string & name ) :
  TimedModule( 0, name ), par(&par), context(&context)
{
  _size     = -1; 
  _nodes    = -1; 
  _channels = -1;
  _classes  = config.getIntField("classes");
}

Network::~Network( )
{
  for ( int r = 0; r < _size; ++r ) {
    if ( _routers[r] ) delete _routers[r];
  }
  for ( int s = 0; s < _nodes; ++s ) {
    if ( _inject[s] ) delete _inject[s];
    if ( _inject_cred[s] ) delete _inject_cred[s];
  }
  for ( int d = 0; d < _nodes; ++d ) {
    if ( _eject[d] ) delete _eject[d];
    if ( _eject_cred[d] ) delete _eject_cred[d];
  }
  for ( int c = 0; c < _channels; ++c ) {
    if ( _chan[c] ) delete _chan[c];
    if ( _chan_cred[c] ) delete _chan_cred[c];
  }
}

Network * Network::New(const Configuration & config, SimulationContext& context, tRoutingParameters& par, const string & name)
{
  const string topo = config.getStrField( "topology" );
  Network * n = NULL;
  if ( topo == "torus" ) {
    KNCube::RegisterRoutingFunctions(par) ;
    n = new KNCube( config, context, par, name, false );
  } else if ( topo == "mesh" ) {
    KNCube::RegisterRoutingFunctions(par) ;
    n = new KNCube( config, context, par, name, true );
  } else if ( topo == "cmesh" ) {
    CMesh::RegisterRoutingFunctions(par) ;
    n = new CMesh( config, context, par, name );
  } else if ( topo == "fly" ) {
    KNFly::RegisterRoutingFunctions(par) ;
    n = new KNFly( config, context, par, name );
  } else if ( topo == "qtree" ) {
    QTree::RegisterRoutingFunctions(par) ;
    n = new QTree( config, context, par, name );
  } else if ( topo == "tree4" ) {
    Tree4::RegisterRoutingFunctions(par) ;
    n = new Tree4( config, context, par, name );
  } else if ( topo == "fattree" ) {
    FatTree::RegisterRoutingFunctions(par) ;
    n = new FatTree( config, context, par, name );
  } else if ( topo == "flatfly" ) {
    FlatFlyOnChip::RegisterRoutingFunctions(par) ;
    n = new FlatFlyOnChip( config, context, par, name );
  } else if ( topo == "anynet"){
    AnyNet::RegisterRoutingFunctions(par) ;
    n = new AnyNet(config, context, par, name);
  } else if ( topo == "dragonflynew"){
    DragonFlyNew::RegisterRoutingFunctions(par) ;
    n = new DragonFlyNew(config, context, par, name);
  } else {
    cerr << "Unknown topology: " << topo << endl;
  }
  
  /*legacy code that insert random faults in the networks
   *not sure how to use this
   */
  if ( n && ( config.getIntField( "link_failures" ) > 0 ) ) {
    n->InsertRandomFaults( config );
  }
  return n;
}

void Network::_Alloc( )
{
  assert( ( _size != -1 ) && 
	  ( _nodes != -1 ) && 
	  ( _channels != -1 ) );

  _routers.resize(_size);
  context->gNodes = _nodes;

  /*booksim used arrays of flits as the channels which makes have capacity of
   *one. To simulate channel latency, flitchannel class has been added
   *which are fifos with depth = channel latency and each cycle the channel
   *shifts by one
   *credit channels are the necessary counter part
   */
  _inject.resize(_nodes);
  _inject_cred.resize(_nodes);
  for ( int s = 0; s < _nodes; ++s ) {
    ostringstream name;
    name << getName() << "_fchan_ingress" << s;
    _inject[s] = new FlitChannel(this, *context, name.str(), _classes);
    _inject[s]->setSrcRouter(NULL, s);
    _timed_modules.push_back(_inject[s]);
    name.str("");
    name << getName() << "_cchan_ingress" << s;
    _inject_cred[s] = new CreditChannel(this, *context, name.str());
    _timed_modules.push_back(_inject_cred[s]);
  }
  _eject.resize(_nodes);
  _eject_cred.resize(_nodes);
  for ( int d = 0; d < _nodes; ++d ) {
    ostringstream name;
    name << getName() << "_fchan_egress" << d;
    _eject[d] = new FlitChannel(this, *context, name.str(), _classes);
    _eject[d]->setSnkRouter(NULL, d);
    _timed_modules.push_back(_eject[d]);
    name.str("");
    name << getName() << "_cchan_egress" << d;
    _eject_cred[d] = new CreditChannel(this, *context, name.str());
    _timed_modules.push_back(_eject_cred[d]);
  }
  _chan.resize(_channels);
  _chan_cred.resize(_channels);
  for ( int c = 0; c < _channels; ++c ) {
    ostringstream name;
    name << getName() << "_fchan_" << c;
    _chan[c] = new FlitChannel(this, *context, name.str(), _classes);
    _timed_modules.push_back(_chan[c]);
    name.str("");
    name << getName() << "_cchan_" << c;
    _chan_cred[c] = new CreditChannel(this, *context, name.str());
    _timed_modules.push_back(_chan_cred[c]);
  }
}

void Network::readInputs( )
{
  for(deque<TimedModule *>::const_iterator iter = _timed_modules.cbegin();
      iter != _timed_modules.cend();
      ++iter) {
    (*iter)->readInputs( );
  }
}

void Network::evaluate( )
{
  for(deque<TimedModule *>::const_iterator iter = _timed_modules.cbegin();
      iter != _timed_modules.cend();
      ++iter) {
    (*iter)->evaluate( );
  }
}

void Network::writeOutputs( )
{
  for(deque<TimedModule *>::const_iterator iter = _timed_modules.cbegin();
      iter != _timed_modules.cend();
      ++iter) {
    (*iter)->writeOutputs( );
  }
}

void Network::WriteFlit( Flit *f, int source )
{
  assert( ( source >= 0 ) && ( source < _nodes ) );
  _inject[source]->send(f);
}

Flit *Network::ReadFlit( int dest )
{
  assert( ( dest >= 0 ) && ( dest < _nodes ) );
  return _eject[dest]->receive();
}

void Network::WriteCredit( Credit *c, int dest )
{
  assert( ( dest >= 0 ) && ( dest < _nodes ) );
  _eject_cred[dest]->send(c);
}

Credit *Network::ReadCredit( int source )
{
  assert( ( source >= 0 ) && ( source < _nodes ) );
  return _inject_cred[source]->receive();
}

void Network::InsertRandomFaults( const Configuration &config )
{
  error( "InsertRandomFaults not implemented for this topology!" );
}

void Network::OutChannelFault( int r, int c, bool fault )
{
  assert( ( r >= 0 ) && ( r < _size ) );
  _routers[r]->OutChannelFault( c, fault );
}

double Network::Capacity( ) const
{
  return 1.0;
}

/* this function can be heavily modified to display any information
 * neceesary of the network, by default, call display on each router
 * and display the channel utilization rate
 */
void Network::Display( ostream & os ) const
{
  for ( int r = 0; r < _size; ++r ) {
    _routers[r]->display( os );
  }
}

void Network::DumpChannelMap( ostream & os, string const & prefix ) const
{
  os << prefix << "source_router,source_port,dest_router,dest_port" << endl;
  for(int c = 0; c < _nodes; ++c)
    os << prefix
       << "-1," 
       << _inject[c]->getSrcPort()<< ',' 
       << _inject[c]->getSnkRouter()->GetID() << ',' 
       << _inject[c]->getSnkPort() << endl;
  for(int c = 0; c < _channels; ++c)
    os << prefix
       << _chan[c]->getSrcRouter()->GetID() << ',' 
       << _chan[c]->getSrcPort() << ',' 
       << _chan[c]->getSnkRouter()->GetID() << ',' 
       << _chan[c]->getSnkPort() << endl;
  for(int c = 0; c < _nodes; ++c)
    os << prefix
       << _eject[c]->getSrcRouter()->GetID() << ',' 
       << _eject[c]->getSrcPort() << ',' 
       << "-1," 
       << _eject[c]->getSnkPort() << endl;
}

void Network::DumpNodeMap( ostream & os, string const & prefix ) const
{
  os << prefix << "source_router,dest_router" << endl;
  for(int s = 0; s < _nodes; ++s)
    os << prefix
       << _eject[s]->getSrcRouter()->GetID() << ','
       << _inject[s]->getSnkRouter()->GetID() << endl;
}
