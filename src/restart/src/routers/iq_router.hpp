/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: iq_router.hpp
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


#ifndef _IQ_ROUTER_HPP_
#define _IQ_ROUTER_HPP_

#include <string>
#include <deque>
#include <queue>
#include <set>
#include <map>

#include "router.hpp"
#include "routefunc.hpp"

using namespace std;

class VC;
class Flit;
class Credit;
class Buffer;
class BufferState;
class Allocator;
class SwitchMonitor;
class BufferMonitor;

class IQRouter : public Router {

  int _vcs;

  bool _vc_busy_when_full;
  bool _vc_prioritize_empty;
  bool _vc_shuffle_requests;

  bool _speculative;
  bool _spec_check_elig;
  bool _spec_check_cred;
  bool _spec_mask_by_reqs;
  
  bool _active;

  int _routing_delay;
  int _vc_alloc_delay;
  int _sw_alloc_delay;
  
  map<int, Flit *> _in_queue_flits;

  deque<pair<int, pair<Credit *, int> > > _proc_credits;

  deque<pair<int, pair<int, int> > > _route_vcs;
  deque<pair<int, pair<pair<int, int>, int> > > _vc_alloc_vcs;  
  deque<pair<int, pair<pair<int, int>, int> > > _sw_hold_vcs;
  deque<pair<int, pair<pair<int, int>, int> > > _sw_alloc_vcs;

  deque<pair<int, pair<Flit *, pair<int, int> > > > _crossbar_flits;

  map<int, Credit *> _out_queue_credits;

  vector<Buffer *> _buf;
  vector<BufferState *> _next_buf;

  Allocator *_vc_allocator;
  Allocator *_sw_allocator;
  Allocator *_spec_sw_allocator;
  
  vector<int> _vc_rr_offset;
  vector<int> _sw_rr_offset;

  tRoutingFunction   _rf;

  int _output_buffer_size;
  vector<queue<Flit *> > _output_buffer;

  vector<queue<Credit *> > _credit_buffer;

  bool _hold_switch_for_packet;
  vector<int> _switch_hold_in;
  vector<int> _switch_hold_out;
  vector<int> _switch_hold_vc;

  bool _noq;
  vector<vector<int> > _noq_next_output_port;
  vector<vector<int> > _noq_next_vc_start;
  vector<vector<int> > _noq_next_vc_end;

#ifdef TRACK_FLOWS
  vector<vector<queue<int> > > _outstanding_classes;
#endif

  bool _ReceiveFlits( );
  bool _ReceiveCredits( );

  virtual void _InternalStep( );

  bool _SWAllocAddReq(int input, int vc, int output);

  void _InputQueuing( );

  void _RouteEvaluate( );
  void _VCAllocEvaluate( );
  void _SWHoldEvaluate( );
  void _SWAllocEvaluate( );
  void _SwitchEvaluate( );

  void _RouteUpdate( );
  void _VCAllocUpdate( );
  void _SWHoldUpdate( );
  void _SWAllocUpdate( );
  void _SwitchUpdate( );

  void _OutputQueuing( );

  void _SendFlits( );
  void _SendCredits( );
  
  void _UpdateNOQ(int input, int vc, Flit const * f);

  // ----------------------------------------
  //
  //   Router Power Modellingyes
  //
  // ----------------------------------------

  SwitchMonitor * _switchMonitor ;
  BufferMonitor * _bufferMonitor ;
  
public:

  IQRouter( Configuration const & config,
      const SimulationContext & context,
      const tRoutingParameters & par,
	    Module *parent, string const & name, int id,
	    int inputs, int outputs );
  
  virtual ~IQRouter( );
  
  virtual void AddOutputChannel(FlitChannel * channel, CreditChannel * backchannel);

  virtual void readInputs( );
  virtual void writeOutputs( );
  
  void Display( ostream & os = cout ) const;

  virtual int GetUsedCredit(int o) const;
  virtual int GetBufferOccupancy(int i) const;

#ifdef TRACK_BUFFERS
  virtual int GetUsedCreditForClass(int output, int cl) const;
  virtual int GetBufferOccupancyForClass(int input, int cl) const;
#endif

  virtual vector<int> UsedCredits() const;
  virtual vector<int> FreeCredits() const;
  virtual vector<int> MaxCredits() const;

  SwitchMonitor const * const GetSwitchMonitor() const {return _switchMonitor;}
  BufferMonitor const * const GetBufferMonitor() const {return _bufferMonitor;}

};

#endif
