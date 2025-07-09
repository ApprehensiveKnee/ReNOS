
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

#ifndef _INJECTION_HPP_
#define _INJECTION_HPP_

#include "params.hpp"
#include "config.hpp"
#include "base.hpp"
#include "traffic.hpp"
#include "logger.hpp"
#include "memory.hpp"
#include "npu.hpp"

using namespace std;

commType intToCommType(int i);

/* ============ SPECIAL CLASS WRAPPER FOR DEPENTENT INJECTION PARAMETERS ============ */
class DependentInjectionProcessParameters {
  public:
    int nodes;
    int localMemSize;
    int localMemThreshold;
    double reconfigCycles;
    double compCycles;
    int reconfBatchSize;
    int flitSize;
    Clock* clock;
    TrafficPattern * trafficPattern;
    std::vector<std::set<std::tuple<int,int,int>>> * processedPackets;
    
    // constructor
    DependentInjectionProcessParameters(
        int nodes,
        int localMemSize,
        int localMemThreshold,
        double reconfigCycles,
        double compCycles,
        int reconfBatchSize,
        int flitSize,
        Clock * clock,
        TrafficPattern * trafficPattern,
        std::vector<std::set<std::tuple<int,int,int>>> * processedPackets
    ) : nodes(nodes),
        localMemSize(localMemSize),
        localMemThreshold(localMemThreshold),
        reconfigCycles(reconfigCycles),
        compCycles(compCycles),
        reconfBatchSize(reconfBatchSize),
        flitSize(flitSize),
        clock(clock),
        trafficPattern(trafficPattern),
        processedPackets(processedPackets)
    {}

    // static method to create a new instance of the class
    static DependentInjectionProcessParameters * New(
        int nodes,
        int localMemSize,
        int localMemThreshold,
        double reconfigCycles,
        double compCycles,
        int reconfBatchSize,
        int flitSize,
        Clock * clock,
        TrafficPattern * trafficPattern,
        std::vector<std::set<std::tuple<int,int,int>>> * processedPackets
    ){
      return new DependentInjectionProcessParameters(
        nodes,
        localMemSize,
        localMemThreshold,
        reconfigCycles,
        compCycles,
        reconfBatchSize,
        flitSize,
        clock,
        trafficPattern,
        processedPackets
      );
    }
};
/* ============ SPECIAL CLASS WRAPPER FOR DEPENTENT INJECTION PARAMETERS ============ */

class InjectionProcess {
protected:
  int _nodes;
  double _rate;
  InjectionProcess(int nodes, double rate);
public:
  virtual ~InjectionProcess() {}
  virtual bool test(int source) = 0;
  virtual bool isIdle(int source) { return false;};
  virtual void requestInterrupt(int source, int processing_time){};
  virtual void addToWaitingQueue(int source, Packet * p){};
  virtual int getAvailableMemory(int source){return 0;};
  virtual void stageReconfiguration(int source, bool bypass_empty_memory_check = false) {};
  virtual void finalizeCommunication(int source, int w_id, int packet_id){}
  virtual void finalizeWrite(int dest, int w_prev_id, int packet_id){};
  virtual void deallocateMemory(int source, const ComputingWorkload * w){};
  virtual void reset();
  bool reached_end;
  static InjectionProcess * New(string const & inject, int nodes, double load,
				Configuration const * const config = NULL);
  static InjectionProcess * NewUserDefined(string const & inject, const DependentInjectionProcessParameters * dep_par , const SimulationContext * context,
        Configuration const * const config = NULL);

};

class BernoulliInjectionProcess : public InjectionProcess {
public:
  BernoulliInjectionProcess(int nodes, double rate);
  virtual bool test(int source);
};

class OnOffInjectionProcess : public InjectionProcess {
private:
  double _alpha;
  double _beta;
  double _r1;
  vector<int> _initial;
  vector<int> _state;
public:
  OnOffInjectionProcess(int nodes, double rate, double alpha, double beta, 
			double r1, vector<int> initial);
  virtual void reset();
  virtual bool test(int source);
};

// ------ additions to support user defined injection processes ------

class DependentInjectionProcess : public InjectionProcess {

  private:
    int _resorting;

    // timer to keep track of computation and reconfiguration timers have been 
    // separated into NPU and memory object timers.

    // ==================  RECONFIGURATION ==================
    bool _enable_reconf;
    int _reconf_batch_size;
    MemorySet * _memory_set;
    MemoryRateRegister * _memory_rate_register; // a register for memory reconfiguration rates
    // ==================  RECONFIGURATION ==================

    NPUSet * _npu_set; // a set of NPUs, one for each node
    NPURateRegister * _npu_rate_register; // a register for comp. rates of the NPUs

    vector<set<tuple<int, int, int>>> * _processed; // register for processed packets
    vector<set<tuple<int, int, int>>> _executed; // a register for executed workloads
    vector<deque<const Packet *>> _waiting_packets; // a queue for the packets waiting to be injected
    // pending packets: NON-BLOCKING STRATEGY IS ADOPTED
    vector<deque<const ComputingWorkload * >> _waiting_workloads; // a queue for the workloads waiting to be processed
    vector<const ComputingWorkload *> _pending_workloads; // used as buffer for workloads being processed

    const SimulationContext * _context;
    EventLogger * _logger;
    Clock * _clock;
    //a pointer to the traffic object, holding the packets to be injected
    TrafficPattern * _traffic;

    // a method to check if the dependencies have been satistisfied for the packet
    int _dependenciesPSatisfied (const Packet * p, int source, bool extensive_search = false) const;

    // a method to check if the dependencies have been satisfied for the workload
    int _dependenciesWSatisfied (const ComputingWorkload * u, int source, bool extensive_search= false) const;

    // a new method to resort the waiting queues of packets
    bool _resortWaitingPQueues( vector<deque<const Packet *>> & waiting_packets, int source);


    //================ RECONFIGURATION =================
    // a method to manage the reconfiguration of the PEs
    void _reconfigure(int source);
    // a method to manage end of computation and integration with reconfiguration
    bool _manageReconfPacketInjection(const Packet * p, int p_dep_time, int source);
    //================ RECONFIGURATION =================

    bool _managePacketInjection(const Packet * p);
    // a method to build the waiting queues
    void _buildStaticWaitingQueues();

  public:
    // constructor
    DependentInjectionProcess(const DependentInjectionProcessParameters& params, const SimulationContext * context , int resort = 0);
    // descructor
    virtual ~DependentInjectionProcess();


    virtual void reset();
    // a method used to append additional packets to the waiting queues
    virtual void addToWaitingQueue(int source, Packet * p);
    // a method to remove a packet id from _requiring_ouput_deallocation
    void finalizeCommunication(int source, int w_id, int packet_id){

      _memory_set->getMemoryUnit(source).markReplyReceived(w_id, packet_id);
      if (_memory_set->getMemoryUnit(source).checkReplyReceived(w_id)){
        // remove the output placeholder from memory at the source node
        assert(_memory_set->getMemoryUnit(source).removeOutputPlaceholder(w_id));
        // and deallocate the corresponding space
        const ComputingWorkload * associated_workload = _traffic->workloadByID(w_id);
        assert(associated_workload);
        _memory_set->deallocateOutput(source, associated_workload);

        (*_context->gDumpFile) << " DEALLOCATING OUTPUT FOR WORKLOAD " << associated_workload->id << " ON NODE " << source << std::endl;
        
        // stage reconfiguration
        (_reconf_batch_size>0)? stageBatchReconfiguration(source): stageReconfiguration(source);
      }
    };

    // a method to temporarly allocate space for the receiving results
    void finalizeWrite(int dest, int w_prev_id, int packet_id){

      // find the receving workload
      auto w_next = std::find_if(_waiting_workloads[dest].begin(), _waiting_workloads[dest].end(), [packet_id](const ComputingWorkload * w){
        for (auto & d : w->dep){
          if (d == packet_id){
            return true;
          }
        }
        return false;
      });

      // if it is not found, we can skip the allocation
      if (w_next == _waiting_workloads[dest].end()){
        return;
      }
      // if the workload is already allocated, we can skip allocation
      if (_memory_set->getMemoryUnit(dest).isWorkloadAllocated(*w_next)){
        return;
      }
      
      // if not, we must allocate the output space for the workload
      const ComputingWorkload * associated_workload = _traffic->workloadByID(w_prev_id);
      assert(associated_workload);
      
      // 1. check if the entry for w_next already exists, if not create it
      _memory_set->getMemoryUnit(dest).createTempStoredResultsEntry((*w_next)->id);
      // 2. if no other entry for the same workload is present, insert the entry in the set
      _memory_set->getMemoryUnit(dest).addTempStoredResults((*w_next)->id, associated_workload);
      // 3. allocate space for the output on the node
      _memory_set->allocateOutput(dest, associated_workload);
      (*_context->gDumpFile) << " ALLOCATING OUTPUT FOR WORKLOAD " << associated_workload->id << " ON NODE " << dest << std::endl;
      *(_context->gDumpFile) << "Output of workload " << associated_workload->id << " to be deallocated on reconfiguration of workload " << (*w_next)->id << " at node " << dest << std::endl;
    }

    
    // a method to test if the landed packets can be processed.
    // the node is idle is its not computing or reconfiguring
    virtual bool isIdle(int source){
      return (_memory_set->getMemoryUnit(source).isIdle() && _npu_set->getNPU(source).isIdle());
    };


    // a method to model the time required for the "simulated" interrupts
    virtual void requestInterrupt(int source, int processing_time){
      // first check if the node is either computing or reconfiguring:
      if (!_memory_set->getMemoryUnit(source).isIdle()){
        // append the processing time to the timer for reconfiguration
        int time_needed = _memory_set->getMemoryUnit(source).required_time + processing_time;
        _memory_set->getMemoryUnit(source).required_time = time_needed;
        *(_context->gDumpFile) << "Appending INTERRRUPT PROCESSING TIME " << processing_time << " to node " << source << " for reconfiguration" << std::endl;
      }
      else if (!_npu_set->getNPU(source).isIdle()){
        // append the processing time to the timer for computation
        int time_needed = _npu_set->getNPU(source).required_time + processing_time;
        _npu_set->getNPU(source).required_time = time_needed;
        *(_context->gDumpFile) << "Appending INTERRRUPT PROCESSING TIME " << processing_time << " to node " << source << " for computation" << std::endl;
      }
      
    };
    // a method to get the total available memory on a node
    virtual int getAvailableMemory(int source){return _memory_set->getAvailable(source);};
    // a method to deallocate memory on a node
    virtual void deallocateMemory(int source, const ComputingWorkload * w){_memory_set->deallocateOutput(source, w);};
    // a method to stage a reconfiguration
    virtual void stageReconfiguration(int source, bool bypass_empty_memory_check = false);
    // a method to stage a batch reconfiguration
    virtual void stageBatchReconfiguration(int source, bool bypass_output_check = false);
    // finally, a method to test if the packet can be injected
    virtual bool test(int source);
    
};



#endif 
