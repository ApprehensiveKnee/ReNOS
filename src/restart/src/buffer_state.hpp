/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: buffer.hpp
//  Description: Header for the declaration of the buffer_state classs
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

#ifndef BUFFER_STATE_HPP
#define BUFFER_STATE_HPP

#include <vector>
#include <queue>

#include "base.hpp"
#include "packet.hpp"
#include "config.hpp"
#include "routefunc.hpp"

class BufferState : public Module {
  
  class BufferPolicy : public Module {
    protected:
        BufferState const * const _buffer_state;
    public:
        BufferPolicy(Configuration const & config, BufferState * parent, 
            const std::string & name);
        virtual void setMinLatency(int min_latency) {}
        virtual void takeBuffer(int vc = 0);
        virtual void sendingFlit(Flit const * const f);
        virtual void freeSlotFor(int vc = 0);
        virtual bool isFullFor(int vc = 0) const = 0;
        virtual int availableFor(int vc = 0) const = 0;
        virtual int limitFor(int vc = 0) const = 0;

        static BufferPolicy * New(Configuration const & config, SimulationContext const & context,
                    BufferState * parent, const std::string & name);
  };
  
  class PrivateBufferPolicy : public BufferPolicy {
    protected:
        int _vc_buf_size;
    public:
        PrivateBufferPolicy(Configuration const & config, BufferState * parent, 
                const std::string & name);
        virtual void sendingFlit(Flit const * const f);
        virtual bool isFullFor(int vc = 0) const;
        virtual int availableFor(int vc = 0) const;
        virtual int limitFor(int vc = 0) const;
  };
  
  class SharedBufferPolicy : public BufferPolicy {
    protected:
        int _buf_size;
        std::vector<int> _private_buf_vc_map;
        std::vector<int> _private_buf_size;
        std::vector<int> _private_buf_occupancy;
        int _shared_buf_size;
        int _shared_buf_occupancy;
        std::vector<int> _reserved_slots;
        void processFreeSlot(int vc = 0);
    public:
        SharedBufferPolicy(Configuration const & config, BufferState * parent, 
                const std::string & name);
        virtual void sendingFlit(Flit const * const f);
        virtual void freeSlotFor(int vc = 0);
        virtual bool isFullFor(int vc = 0) const;
        virtual int availableFor(int vc = 0) const;
        virtual int limitFor(int vc = 0) const;
  };

  class LimitedSharedBufferPolicy : public SharedBufferPolicy {
    protected:
        int _vcs;
        int _active_vcs;
        int _max_held_slots;
    public:
        LimitedSharedBufferPolicy(Configuration const & config, 
                    BufferState * parent,
                    const std::string & name);
        virtual void takeBuffer(int vc = 0);
        virtual void sendingFlit(Flit const * const f);
        virtual bool isFullFor(int vc = 0) const;
        virtual int availableFor(int vc = 0) const;
        virtual int limitFor(int vc = 0) const;
  };
    
  class DynamicLimitedSharedBufferPolicy : public LimitedSharedBufferPolicy {
    public:
        DynamicLimitedSharedBufferPolicy(Configuration const & config, 
                        BufferState * parent,
                        const std::string & name);
        virtual void takeBuffer(int vc = 0);
        virtual void sendingFlit(Flit const * const f);
  };
  
  class ShiftingDynamicLimitedSharedBufferPolicy : public DynamicLimitedSharedBufferPolicy {
    public:
        ShiftingDynamicLimitedSharedBufferPolicy(Configuration const & config, 
                            BufferState * parent,
                            const std::string & name);
        virtual void takeBuffer(int vc = 0);
        virtual void sendingFlit(Flit const * const f);
  };
  
  class FeedbackSharedBufferPolicy : public SharedBufferPolicy {
    protected:
        int _computeRTT(int vc, int last_rtt) const;
        int _computeLimit(int rtt) const;
        int _computeMaxSlots(int vc) const;
        const SimulationContext * _context;
        int _vcs;
        std::vector<int> _occupancy_limit;
        std::vector<int> _round_trip_time;
        std::vector<std::queue<int> > _flit_sent_time;
        int _min_latency;
        int _total_mapped_size;
        int _aging_scale;
        int _offset;
    public:
        FeedbackSharedBufferPolicy(Configuration const & config, const SimulationContext * context,
                    BufferState * parent, const std::string & name);
        virtual void setMinLatency(int min_latency);
        virtual void sendingFlit(Flit const * const f);
        virtual void freeSlotFor(int vc = 0);
        virtual bool isFullFor(int vc = 0) const;
        virtual int availableFor(int vc = 0) const;
        virtual int limitFor(int vc = 0) const;
  };
  
  class SimpleFeedbackSharedBufferPolicy : public FeedbackSharedBufferPolicy {
    protected:
        std::vector<int> _pending_credits;
    public:
        SimpleFeedbackSharedBufferPolicy(Configuration const & config, const SimulationContext * context,
                        BufferState * parent, const std::string & name);
        virtual void sendingFlit(Flit const * const f);
        virtual void freeSlotFor(int vc = 0);
  };
  
  bool _wait_for_tail_credit;
  int  _size;
  int  _occupancy;
  std::vector<int> _vc_occupancy;
  int  _vcs;
  
  BufferPolicy * _buffer_policy;
  
  std::vector<int> _in_use_by;
  std::vector<bool> _tail_sent;
  std::vector<int> _last_id;
  std::vector<int> _last_pid;

#ifdef TRACK_BUFFERS
  int _classes;
  std::vector<queue<int> > _outstanding_classes;
  std::vector<int> _class_occupancy;
#endif

public:

  BufferState( const Configuration& config, const SimulationContext& context, 
	       Module *parent, const std::string& name );

  ~BufferState();

  inline void setMinLatency(int min_latency) {
    _buffer_policy->setMinLatency(min_latency);
  }

  void processCredit( Credit const * const c );
  void sendingFlit( Flit const * const f );

  void takeBuffer( int vc = 0, int tag = 0 );

  inline bool isFull() const {
    assert(_occupancy <= _size);
    return (_occupancy == _size);
  }
  inline bool isFullFor( int vc = 0 ) const {
    return _buffer_policy->isFullFor(vc);
  }
  inline int availableFor( int vc = 0 ) const {
    return _buffer_policy->availableFor(vc);
  }
  inline int limitFor( int vc = 0 ) const {
    return _buffer_policy->limitFor(vc);
  }
  inline bool isEmptyFor(int vc = 0) const {
    assert((vc >= 0) && (vc < _vcs));
    return (_vc_occupancy[vc] == 0);
  }
  inline bool isAvailableFor( int vc = 0 ) const {
    assert( ( vc >= 0 ) && ( vc < _vcs ) );
    return _in_use_by[vc] < 0;
  }
  inline int usedBy(int vc = 0) const {
    assert( ( vc >= 0 ) && ( vc < _vcs ) );
    return _in_use_by[vc];
  }
    
  inline int occupancy() const {
    return _occupancy;
  }

  inline int occupancyFor( int vc = 0 ) const {
    assert((vc >= 0) && (vc < _vcs));
    return _vc_occupancy[vc];
  }
  
#ifdef TRACK_BUFFERS
  inline int OccupancyForClass(int c) const {
    assert((c >= 0) && (c < _classes));
    return _class_occupancy[c];
  }
#endif

  void display( std::ostream & os ) const;
};

#endif // BUFFER_STATE_HPP