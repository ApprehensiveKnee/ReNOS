/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: buffer.hpp
//  Description: Source file for the definiton of buffer_state class methods
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
//  Date:  13/10/2024
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <sstream>
#include <cassert>
#include <cstdlib>
#include <limits>

#include "buffer_state.hpp"
#include "random_utils.hpp"
#include "globals.hpp"

BufferState::BufferPolicy::BufferPolicy(Configuration const & config, BufferState * parent, const std::string & name)
: Module(parent, name), _buffer_state(parent)
{
}

void BufferState::BufferPolicy::takeBuffer(int vc) {
}

void BufferState::BufferPolicy::sendingFlit(Flit const * const f) {
}

void BufferState::BufferPolicy::freeSlotFor(int vc) {
}

BufferState::BufferPolicy * BufferState::BufferPolicy::New(Configuration const & config, SimulationContext const & context , BufferState * parent, const std::string & name)
{
  BufferPolicy * sp = NULL;
  std::string buffer_policy = config.getStrField("buffer_policy");
  if(buffer_policy == "private") {
    sp = new PrivateBufferPolicy(config, parent, name);
  } else if(buffer_policy == "shared") {
    sp = new SharedBufferPolicy(config, parent, name);
  } else if(buffer_policy == "limited") {
    sp = new LimitedSharedBufferPolicy(config, parent, name);
  } else if(buffer_policy == "dynamic") {
    sp = new DynamicLimitedSharedBufferPolicy(config, parent, name);
  } else if(buffer_policy == "shifting") {
    sp = new ShiftingDynamicLimitedSharedBufferPolicy(config, parent, name);
  } else if(buffer_policy == "feedback") {
    sp = new FeedbackSharedBufferPolicy(config, &context,  parent, name);
  } else if(buffer_policy == "simplefeedback") {
    sp = new SimpleFeedbackSharedBufferPolicy(config, &context, parent, name);
  } else {
    *(context.gDumpFile) << "Unknown buffer policy: " << buffer_policy << std::endl;
  }
  return sp;
}

BufferState::PrivateBufferPolicy::PrivateBufferPolicy(Configuration const & config, BufferState * parent, const std::string & name)
  : BufferPolicy(config, parent, name)
{
  int const vcs = config.getIntField( "num_vcs" );
  int const buf_size = config.getIntField("buf_size");
  if(buf_size <= 0) {
    _vc_buf_size = config.getIntField("vc_buf_size");
  } else {
    _vc_buf_size = buf_size / vcs;
  }
  assert(_vc_buf_size > 0);
}

void BufferState::PrivateBufferPolicy::sendingFlit(Flit const * const f)
{
  int const vc = f->vc; // get the vc for the flit
  if(_buffer_state->occupancyFor(vc) > _vc_buf_size) {
    std::ostringstream err;
    err << "Buffer overflow for VC " << vc;
    error(err.str());
  }
}

bool BufferState::PrivateBufferPolicy::isFullFor(int vc) const
{
  return (_buffer_state->occupancyFor(vc) >= _vc_buf_size);
}

int BufferState::PrivateBufferPolicy::availableFor(int vc) const
{
  return _vc_buf_size - _buffer_state->occupancyFor(vc);
}

int BufferState::PrivateBufferPolicy::limitFor(int vc) const
{
  return _vc_buf_size;
}

BufferState::SharedBufferPolicy::SharedBufferPolicy(Configuration const & config, BufferState * parent, const std::string & name)
  : BufferPolicy(config, parent, name), _shared_buf_occupancy(0)
{
  int const vcs = config.getIntField( "num_vcs" );
  int num_private_bufs = config.getIntField("private_bufs");
  if(num_private_bufs < 0) {
    num_private_bufs = vcs;
  } else if(num_private_bufs == 0) {
    num_private_bufs = 1;
  }
  
  _private_buf_occupancy.resize(num_private_bufs, 0);

  _buf_size = config.getIntField("buf_size");
  if(_buf_size < 0) {
    _buf_size = vcs * config.getIntField("vc_buf_size");
  }

  _private_buf_size = config.getIntArray("private_buf_size");
  if(_private_buf_size.empty()) {
    int const bs = config.getIntField("private_buf_size");
    if(bs < 0) {
      _private_buf_size.push_back(_buf_size / num_private_bufs);
    } else {
      _private_buf_size.push_back(bs);
    }
  }
  _private_buf_size.resize(num_private_bufs, _private_buf_size.back());
  
  std::vector<int> start_vc = config.getIntArray("private_buf_start_vc");
  if(start_vc.empty()) {
    int const sv = config.getIntField("private_buf_start_vc");
    if(sv < 0) {
      start_vc.resize(num_private_bufs);
      for(int i = 0; i < num_private_bufs; ++i) {
	start_vc[i] = i * vcs / num_private_bufs;
      }
    } else {
      start_vc.push_back(sv);
    }
  }
  
  std::vector<int> end_vc = config.getIntArray("private_buf_end_vc");
  if(end_vc.empty()) {
    int const ev = config.getIntField("private_buf_end_vc");
    if(ev < 0) {
      end_vc.resize(num_private_bufs);
      for(int i = 0; i < num_private_bufs; ++i) {
	end_vc[i] = (i + 1) * vcs / num_private_bufs - 1;
      }
    } else {
      end_vc.push_back(ev);
    }
  }

  _private_buf_vc_map.resize(vcs, -1);
  _shared_buf_size = _buf_size;
  for(int i = 0; i < num_private_bufs; ++i) {
    _shared_buf_size -= _private_buf_size[i];
    assert(start_vc[i] <= end_vc[i]);
    for(int v = start_vc[i]; v <= end_vc[i]; ++v) {
      assert(_private_buf_vc_map[v] < 0);
      _private_buf_vc_map[v] = i;
    }
  }
  assert(_shared_buf_size >= 0);

  _reserved_slots.resize(vcs, 0);
}

void BufferState::SharedBufferPolicy::processFreeSlot(int vc)
{
  int i = _private_buf_vc_map[vc];
  --_private_buf_occupancy[i];
  if(_private_buf_occupancy[i] < 0) {
    std::ostringstream err;
    err << "Private buffer occupancy fell below zero for buffer " << i;
    error(err.str());
  } else if(_private_buf_occupancy[i] >= _private_buf_size[i]) {
    --_shared_buf_occupancy;
    if(_shared_buf_occupancy < 0) {
      error("Shared buffer occupancy fell below zero.");
    }
  }
}

void BufferState::SharedBufferPolicy::sendingFlit(Flit const * const f)
{
  int const vc = f->vc;
  if(_reserved_slots[vc] > 0) {
    --_reserved_slots[vc];
  } else {
    int i = _private_buf_vc_map[vc];
    ++_private_buf_occupancy[i];
    if(_private_buf_occupancy[i] > _private_buf_size[i]) {
      ++_shared_buf_occupancy;
      if(_shared_buf_occupancy > _shared_buf_size) {
	error("Shared buffer overflow.");
      }
    }
  }
  if(f->tail) {
    while(_reserved_slots[vc]) {
      --_reserved_slots[vc];
      processFreeSlot(vc);
    }
  }
}

void BufferState::SharedBufferPolicy::freeSlotFor(int vc)
{
  if(!_buffer_state->isAvailableFor(vc) && _buffer_state->isEmptyFor(vc)) {
    ++_reserved_slots[vc];
  } else {
    processFreeSlot(vc);
  }
}

bool BufferState::SharedBufferPolicy::isFullFor(int vc) const
{
  int i = _private_buf_vc_map[vc];
  return ((_reserved_slots[vc] == 0) &&
	  (_private_buf_occupancy[i] >= _private_buf_size[i]) &&
	  (_shared_buf_occupancy >= _shared_buf_size));
}

int BufferState::SharedBufferPolicy::availableFor(int vc) const
{
  int i = _private_buf_vc_map[vc];
  return (_reserved_slots[vc] + 
	  std::max(_private_buf_size[i] - _private_buf_occupancy[i], 0) +
	  (_shared_buf_size - _shared_buf_occupancy));
}

int BufferState::SharedBufferPolicy::limitFor(int vc) const
{
  int i = _private_buf_vc_map[vc];
  return (_private_buf_size[i] + _shared_buf_size);
}

BufferState::LimitedSharedBufferPolicy::LimitedSharedBufferPolicy(Configuration const & config, BufferState * parent, const std::string & name)
  : SharedBufferPolicy(config, parent, name), _active_vcs(0)
{
  _vcs = config.getIntField("num_vcs");
  _max_held_slots = config.getIntField("max_held_slots");
  if(_max_held_slots < 0) {
    _max_held_slots = _buf_size;
  }
}

void BufferState::LimitedSharedBufferPolicy::takeBuffer(int vc)
{
  ++_active_vcs;
  if(_active_vcs > _vcs) {
    error("Number of active VCs is too large.");
  }
}

void BufferState::LimitedSharedBufferPolicy::sendingFlit(Flit const * const f)
{
  SharedBufferPolicy::sendingFlit(f);
  if(f->tail) {
    --_active_vcs;
    if(_active_vcs < 0) {
      error("Number of active VCs fell below zero.");
    }
  }
}

bool BufferState::LimitedSharedBufferPolicy::isFullFor(int vc) const
{
  return (SharedBufferPolicy::isFullFor(vc) ||
	  (_buffer_state->occupancyFor(vc) >= _max_held_slots));
}

int BufferState::LimitedSharedBufferPolicy::availableFor(int vc) const
{
  return std::min(SharedBufferPolicy::availableFor(vc), 
	     _max_held_slots - _buffer_state->occupancyFor(vc));
}

int BufferState::LimitedSharedBufferPolicy::limitFor(int vc) const
{
  return std::min(SharedBufferPolicy::limitFor(vc), _max_held_slots);
}

BufferState::DynamicLimitedSharedBufferPolicy::DynamicLimitedSharedBufferPolicy(Configuration const & config, BufferState * parent, const std::string & name)
  : LimitedSharedBufferPolicy(config, parent, name)
{
  _max_held_slots = _buf_size;
}

void BufferState::DynamicLimitedSharedBufferPolicy::takeBuffer(int vc)
{
  LimitedSharedBufferPolicy::takeBuffer(vc);
  assert(_active_vcs > 0);
  _max_held_slots = _buf_size / _active_vcs;
  assert(_max_held_slots > 0);
}

void BufferState::DynamicLimitedSharedBufferPolicy::sendingFlit(Flit const * const f)
{
  LimitedSharedBufferPolicy::sendingFlit(f);
  if(f->tail && _active_vcs) {
    _max_held_slots = _buf_size / _active_vcs;
  }
  assert(_max_held_slots > 0);
}

BufferState::ShiftingDynamicLimitedSharedBufferPolicy::ShiftingDynamicLimitedSharedBufferPolicy(Configuration const & config, BufferState * parent, const std::string & name)
  : DynamicLimitedSharedBufferPolicy(config, parent, name)
{

}

void BufferState::ShiftingDynamicLimitedSharedBufferPolicy::takeBuffer(int vc)
{
  LimitedSharedBufferPolicy::takeBuffer(vc);
  assert(_active_vcs);
  int i = _active_vcs - 1;
  _max_held_slots = _buf_size;
  while(i) {
    _max_held_slots >>= 1;
    i >>= 1;
  }
  assert(_max_held_slots > 0);
}

void BufferState::ShiftingDynamicLimitedSharedBufferPolicy::sendingFlit(Flit const * const f)
{
  LimitedSharedBufferPolicy::sendingFlit(f);
  if(f->tail && _active_vcs) {
    int i = _active_vcs - 1;
    _max_held_slots = _buf_size;
    while(i) {
      _max_held_slots >>= 1;
      i >>= 1;
    }
  }
  assert(_max_held_slots > 0);
}

BufferState::FeedbackSharedBufferPolicy::FeedbackSharedBufferPolicy(Configuration const & config, const SimulationContext * context,  BufferState * parent, const std::string & name)
  : SharedBufferPolicy(config, parent, name)
{
  _aging_scale = config.getIntField("feedback_aging_scale");
  _offset = config.getIntField("feedback_offset");
  _context = context;
  _vcs = config.getIntField("num_vcs");

  _occupancy_limit.resize(_vcs, _buf_size);
  _round_trip_time.resize(_vcs, -1);
  _flit_sent_time.resize(_vcs);
  _total_mapped_size = _buf_size * _vcs;
  _min_latency = -1;
}

void BufferState::FeedbackSharedBufferPolicy::setMinLatency(int min_latency)
{
#ifdef DEBUG_FEEDBACK
  std::cerr << getFullName() << ": Setting minimum latency to "
       << min_latency << "." << std::endl;
#endif
  _min_latency = min_latency;
}

void BufferState::FeedbackSharedBufferPolicy::sendingFlit(Flit const * const f)
{
  SharedBufferPolicy::sendingFlit(f);
  _flit_sent_time[f->vc].push(GetSimTime(_context));
}

int BufferState::FeedbackSharedBufferPolicy::_computeRTT(int vc, int last_rtt) const
{
  // compute moving average of round-trip time
  int rtt = _round_trip_time[vc];
  if(rtt < 0) {
    return last_rtt;
  }
  return ((rtt << _aging_scale) + last_rtt - rtt) >> _aging_scale;
}

int BufferState::FeedbackSharedBufferPolicy::_computeLimit(int rtt) const
{
  // for every cycle that the measured average round trip time exceeded the 
  // observed minimum round trip time, reduce buffer occupancy limit by one
  assert(_min_latency >= 0);
  return std::max((_min_latency << 1) - rtt + _offset, 1);
}

int BufferState::FeedbackSharedBufferPolicy::_computeMaxSlots(int vc) const
{
  int max_slots = _occupancy_limit[vc];
  if(!_flit_sent_time[vc].empty()) {
    int min_rtt = GetSimTime(_context) - _flit_sent_time[vc].front();
    int rtt = _computeRTT(vc, min_rtt);
    int limit = _computeLimit(rtt);
    max_slots = std::min(max_slots, limit);
  }
  return max_slots;
}

void BufferState::FeedbackSharedBufferPolicy::freeSlotFor(int vc)
{
  SharedBufferPolicy::freeSlotFor(vc);
  assert(!_flit_sent_time[vc].empty());
  int const last_rtt = GetSimTime(_context) - _flit_sent_time[vc].front();
#ifdef DEBUG_FEEDBACK
  cerr << FullName() << ": Probe for VC "
       << vc << " came back after "
       << last_rtt << " cycles."
       << std::endl;
#endif
  _flit_sent_time[vc].pop();
  
  int rtt = _computeRTT(vc, last_rtt);
#ifdef DEBUG_FEEDBACK
  int old_rtt = _round_trip_time[vc];
  if(rtt != old_rtt) {
    cerr << FullName() << ": Updating RTT estimate for VC "
	 << vc << " from "
	 << old_rtt << " to "
	 << rtt << " cycles."
	 << std::endl;
  }
#endif
  _round_trip_time[vc] = rtt;

  int limit = _computeLimit(rtt);
#ifdef DEBUG_FEEDBACK
  int old_limit = _occupancy_limit[vc];
  int old_mapped_size = _total_mapped_size;
#endif
  _total_mapped_size += (limit - _occupancy_limit[vc]);
  _occupancy_limit[vc] = limit;
#ifdef DEBUG_FEEDBACK
  if(limit != old_limit) {
    std::cerr << getFullName() << ": Occupancy limit for VC "
	 << vc << " changed from "
	 << old_limit << " to "
	 << limit << " slots."
	 << std::endl;
    std::cerr << getFullName() << ": Total mapped buffer space changed from "
	 << old_mapped_size << " to "
	 << _total_mapped_size << " slots."
	 << std::endl;
  }
#endif
}

bool BufferState::FeedbackSharedBufferPolicy::isFullFor(int vc) const
{
  if(SharedBufferPolicy::isFullFor(vc)) {
    return true;
  }
  return (_buffer_state->occupancyFor(vc) >= _computeMaxSlots(vc));
}

int BufferState::FeedbackSharedBufferPolicy::availableFor(int vc) const
{
  return std::min(SharedBufferPolicy::availableFor(vc), 
	     _computeMaxSlots(vc) - _buffer_state->occupancyFor(vc));
}

int BufferState::FeedbackSharedBufferPolicy::limitFor(int vc) const
{
  return std::min(SharedBufferPolicy::limitFor(vc), _computeMaxSlots(vc));
}

BufferState::SimpleFeedbackSharedBufferPolicy::SimpleFeedbackSharedBufferPolicy(Configuration const & config, const SimulationContext * context ,BufferState * parent, const std::string & name)
  : FeedbackSharedBufferPolicy(config, context, parent, name)
{
  _pending_credits.resize(_vcs, 0);
}

void BufferState::SimpleFeedbackSharedBufferPolicy::sendingFlit(Flit const * const f)
{
  int const & vc = f->vc;
  if(_flit_sent_time[vc].empty()) {
    assert(_buffer_state->occupancyFor(vc) > 0);
    _pending_credits[vc] = _buffer_state->occupancyFor(vc) - 1;
#ifdef DEBUG_SIMPLEFEEDBACK
    cerr << FullName() << ": Sending probe flit for VC "
	 << vc << "; "
	 << _pending_credits[vc] << " non-probe flits in flight."
	 << std::endl;
#endif
    FeedbackSharedBufferPolicy::sendingFlit(f);
    return;
  }
  SharedBufferPolicy::sendingFlit(f);
}

void BufferState::SimpleFeedbackSharedBufferPolicy::freeSlotFor(int vc)
{
  if(!_flit_sent_time[vc].empty() && _pending_credits[vc] == 0) {
#ifdef DEBUG_SIMPLEFEEDBACK
    cerr << FullName() << ": Probe credit for VC "
	 << vc << " came back." << std::endl;
#endif
    FeedbackSharedBufferPolicy::freeSlotFor(vc);
    return;
  }
  if(_pending_credits[vc] > 0) {
    assert(!_flit_sent_time[vc].empty());
    --_pending_credits[vc];
#ifdef DEBUG_SIMPLEFEEDBACK
    cerr << FullName() << ": Ignoring non-probe credit for VC "
	 << vc << "; "
	 << _pending_credits[vc] << " remaining."
	 << std::endl;
#endif
  }
  SharedBufferPolicy::freeSlotFor(vc);
}

BufferState::BufferState( const Configuration& config, const SimulationContext& context, Module *parent, const std::string& name ) : 
  Module( parent, name ), _occupancy(0)
{
  _vcs = config.getIntField( "num_vcs" );
  _size = config.getIntField("buf_size");
  if(_size < 0) {
    _size = _vcs * config.getIntField("vc_buf_size");
  }

  _buffer_policy = BufferPolicy::New(config, context, this, "policy");

  _wait_for_tail_credit = config.getIntField( "wait_for_tail_credit" );

  _vc_occupancy.resize(_vcs, 0);

  _in_use_by.resize(_vcs, -1);
  _tail_sent.resize(_vcs, false);

  _last_id.resize(_vcs, -1);
  _last_pid.resize(_vcs, -1);

#ifdef TRACK_BUFFERS
  _classes = config.getIntField("classes");
  _outstanding_classes.resize(_vcs);
  _class_occupancy.resize(_classes, 0);
#endif
}

BufferState::~BufferState()
{
  delete _buffer_policy;
}

void BufferState::processCredit( Credit const * const c )
{
  assert( c );

  std::set<int>::iterator iter = c->vc.begin();
  while(iter != c->vc.end()) {

    int const vc = *iter;

    assert( ( vc >= 0 ) && ( vc < _vcs ) );

    if ( ( _wait_for_tail_credit ) && 
	 ( _in_use_by[vc] < 0 ) ) {
      std::ostringstream err;
      err << "Received credit for idle VC " << vc;
      error( err.str() );
    }
    --_occupancy;
    if(_occupancy < 0) {
      error("Buffer occupancy fell below zero.");
    }
    --_vc_occupancy[vc];
    if(_vc_occupancy[vc] < 0) {
      std::ostringstream err;
      err << "Buffer occupancy fell below zero for VC " << vc;
      error(err.str());
    }
    if(_wait_for_tail_credit && !_vc_occupancy[vc] && _tail_sent[vc]) {
      assert(_in_use_by[vc] >= 0);
      _in_use_by[vc] = -1;
    }

#ifdef TRACK_BUFFERS
    assert(!_outstanding_classes[vc].empty());
    int cl = _outstanding_classes[vc].front();
    _outstanding_classes[vc].pop();
    assert((cl >= 0) && (cl < _classes));
    assert(_class_occupancy[cl] > 0);
    --_class_occupancy[cl];
#endif

    _buffer_policy->freeSlotFor(vc);

    ++iter;
  }
}


void BufferState::sendingFlit( Flit const * const f )
{
  int const vc = f->vc;

  assert( f && ( vc >= 0 ) && ( vc < _vcs ) );

  ++_occupancy;
  if(_occupancy > _size) {
    error("Buffer overflow.");
  }

  ++_vc_occupancy[vc];
  
  _buffer_policy->sendingFlit(f);
  
#ifdef TRACK_BUFFERS
  _outstanding_classes[vc].push(f->cl);
  ++_class_occupancy[f->cl];
#endif

  if ( f->tail ) {
    _tail_sent[vc] = true;
    
    if ( !_wait_for_tail_credit ) {
      assert(_in_use_by[vc] >= 0);
      _in_use_by[vc] = -1;
    }
  }
  _last_id[vc] = f->id;
  _last_pid[vc] = f->pid;
}

void BufferState::takeBuffer( int vc, int tag )
{
  assert( ( vc >= 0 ) && ( vc < _vcs ) );

  if ( _in_use_by[vc] >= 0 ) {
    std::ostringstream err;
    err << "Buffer taken while in use for VC " << vc;
    error( err.str() );
  }
  _in_use_by[vc] = tag;
  _tail_sent[vc] = false;
  _buffer_policy->takeBuffer(vc);
}

void BufferState::display( std::ostream & os ) const
{
  os << getFullName() << " :" << std::endl;
  os << " occupied = " << _occupancy << std::endl;
  for ( int v = 0; v < _vcs; ++v ) {
    os << "  VC " << v << ": ";
    os << "in_use_by = " << _in_use_by[v] 
       << ", tail_sent = " << _tail_sent[v]
       << ", occupied = " << _vc_occupancy[v] << std::endl;
  }
}
