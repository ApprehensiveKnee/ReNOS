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

#include <iostream>
#include <vector>
#include <cassert>
#include <limits>
#include "random_utils.hpp"
#include "injection.hpp"

bool operator==(const std::pair<int, const ComputingWorkload *> &lhs, const std::pair<int, const ComputingWorkload *> &rhs) {
    return lhs.first == rhs.first && lhs.second == rhs.second;
}

using namespace std;


commType intToCommType(int i)
{
  switch(i) {
    case 1: return READ_REQ;
    case 2: return WRITE_REQ;
    case 3: return READ_ACK; 
    case 4: return WRITE_ACK; 
    case 5: return READ;
    case 6: return WRITE;
    case 0: return ANY; 
    default: throw std::invalid_argument("Invalid integer value for commType");
  }
};

InjectionProcess::InjectionProcess(int nodes, double rate)
  : _nodes(nodes), _rate(rate), reached_end(false)
{
  if(nodes <= 0) {
    cout << "Error: Number of nodes must be greater than zero." << endl;
    exit(-1);
  }
  if((rate < 0.0) || (rate > 1.0)) {
    cout << "Error: Injection process must have load between 0.0 and 1.0."
	 << endl;
    exit(-1);
  }
}

void InjectionProcess::reset()
{

}

InjectionProcess * InjectionProcess::New(string const & inject, int nodes, 
					 double load,
					 Configuration const * const config)
{
  string process_name;
  string param_str;
  size_t left = inject.find_first_of('(');
  if(left == string::npos) {
    process_name = inject;
  } else {
    process_name = inject.substr(0, left);
    size_t right = inject.find_last_of(')');
    if(right == string::npos) {
      param_str = inject.substr(left+1);
    } else {
      param_str = inject.substr(left+1, right-left-1);
    }
  }
  vector<string> params = tokenize_str(param_str);

  InjectionProcess * result = NULL;
  if(process_name == "bernoulli") {
    result = new BernoulliInjectionProcess(nodes, load);
  } else if(process_name == "on_off") {
    bool missing_params = false;
    double alpha = numeric_limits<double>::quiet_NaN();
    if(params.size() < 1) {
      if(config) {
	alpha = config->getFloatField("burst_alpha");
      } else {
	missing_params = true;
      }
    } else {
      alpha = atof(params[0].c_str());
    }
    double beta = numeric_limits<double>::quiet_NaN();
    if(params.size() < 2) {
      if(config) {
	beta = config->getFloatField("burst_beta");
      } else {
	missing_params = true;
      }
    } else {
      beta = atof(params[1].c_str());
    }
    double r1 = numeric_limits<double>::quiet_NaN();
    if(params.size() < 3) {
      r1 = config ? config->getFloatField("burst_r1") : -1.0;
    } else {
      r1 = atof(params[2].c_str());
    }
    if(missing_params) {
      cout << "Missing parameters for injection process: " << inject << endl;
      exit(-1);
    }
    if((alpha < 0.0 && beta < 0.0) || 
       (alpha < 0.0 && r1 < 0.0) || 
       (beta < 0.0 && r1 < 0.0) || 
       (alpha >= 0.0 && beta >= 0.0 && r1 >= 0.0)) {
      cout << "Invalid parameters for injection process: " << inject << endl;
      exit(-1);
    }
    vector<int> initial(nodes);
    if(params.size() > 3) {
      initial = tokenize_int(params[2]);
      initial.resize(nodes, initial.back());
    } else {
      for(int n = 0; n < nodes; ++n) {
	initial[n] = randomInt(1);
      }
    }
    result = new OnOffInjectionProcess(nodes, load, alpha, beta, r1, initial);
  } else {
    cout << "Invalid injection process: " << inject << endl;
    exit(-1);
  }
  return result;
}

InjectionProcess * InjectionProcess::NewUserDefined(string const & inject, const DependentInjectionProcessParameters * dep_par, const SimulationContext * context,
        Configuration const * const config)
{
  string process_name;
  string param_str;
  size_t left = inject.find_first_of('(');
  if(left == string::npos) {
    process_name = inject;
  } else {
    process_name = inject.substr(0, left);
    size_t right = inject.find_last_of(')');
    if(right == string::npos) {
      param_str = inject.substr(left+1);
    } else {
      param_str = inject.substr(left+1, right-left-1);
    }
  }
  vector<string> params = tokenize_str(param_str);

  InjectionProcess * result = NULL;
  if(process_name == "dependent") {
    int resort = config ? config->getIntField("resort_waiting_queues") : 0;
    assert (dep_par);
    result = new DependentInjectionProcess(*dep_par, context, resort);
  } else {
    cout << "Invalid injection process: " << inject << endl;
    exit(-1);
  }
  return result;
}

//=============================================================

BernoulliInjectionProcess::BernoulliInjectionProcess(int nodes, double rate)
  : InjectionProcess(nodes, rate)
{

}

bool BernoulliInjectionProcess::test(int source)
{
  assert((source >= 0) && (source < _nodes));
  return (randomFloat() < _rate);
}

//=============================================================

OnOffInjectionProcess::OnOffInjectionProcess(int nodes, double rate, 
					     double alpha, double beta, 
					     double r1, vector<int> initial)
  : InjectionProcess(nodes, rate), 
    _alpha(alpha), _beta(beta), _r1(r1), _initial(initial)
{
  assert(alpha <= 1.0);
  assert(beta <= 1.0);
  assert(r1 <= 1.0);
  if(alpha < 0.0) {
    assert(beta >= 0.0);
    assert(r1 >= 0.0);
    _alpha = beta * rate / (r1 - rate);
  } else if(beta < 0.0) {
    assert(alpha >= 0.0);
    assert(r1 >= 0.0);
    _beta = alpha * (r1 - rate) / rate;
  } else {
    assert(r1 < 0.0);
    _r1 = rate * (alpha + beta) / alpha;
  }
  reset();
}

void OnOffInjectionProcess::reset()
{
  _state = _initial;
}

bool OnOffInjectionProcess::test(int source)
{
  assert((source >= 0) && (source < _nodes));

  // advance state
  _state[source] = 
    _state[source] ? (randomFloat() >= _beta) : (randomFloat() < _alpha);

  // generate packet
  return _state[source] && (randomFloat() < _r1);
}



// ============================================================================================================

DependentInjectionProcess::DependentInjectionProcess(const DependentInjectionProcessParameters& params, const SimulationContext * context,int resort) 
  : InjectionProcess(params.nodes, 0.0), _traffic(params.trafficPattern), _processed(params.processedPackets), _clock(params.clock), _logger(context->logger), _context(context)
{

  assert(_traffic);
  assert(_processed);
  assert(_clock);
  assert(context);
  _enable_reconf = params.localMemSize > 0 ? true : false;
  _reconf_batch_size =  params.reconfBatchSize;

  int nodes = params.nodes;
  _executed.resize(nodes);
  _waiting_packets.resize(nodes);
  _waiting_workloads.resize(nodes);

  // initialize npus
  
  _npu_rate_register = new NPURateRegister();
  _npu_rate_register->registerWorkloadCycles(WorkloadType::ANY, params.compCycles);
  _npu_set = new NPUSet(nodes, _npu_rate_register, _logger);

  int size = (_enable_reconf) ? params.localMemSize : 1;
  int threshold = (_enable_reconf) ? params.localMemThreshold : 0;
  _memory_rate_register = new MemoryRateRegister(params.reconfigCycles);
  _memory_set = new MemorySet(nodes, size, threshold, _memory_rate_register, _logger);

  for (int i = 0; i < nodes; ++i){
    _waiting_packets[i].resize(0);
    _waiting_workloads[i].resize(0);
  }
  _pending_workloads.resize(nodes);

  _resorting = resort;
  _buildStaticWaitingQueues();

  // redefine flit size macro
  #undef FLIT_SIZE
  #define FLIT_SIZE flit_size
    
}

DependentInjectionProcess::~DependentInjectionProcess()
{
  delete _memory_set;
  delete _npu_set;
  delete _memory_rate_register;
  delete _npu_rate_register;
}

int DependentInjectionProcess::_dependenciesPSatisfied(const Packet * p, int source, bool extensive_search ) const{
  assert(p);
  assert((p->dep.size() > 0));
  set<int> dep(p->dep.begin(), p->dep.end());
  std::vector<int> dep_time;

  if (dep.size() == 1){
    // check wether the only element is -1
    if (find(dep.begin(), dep.end(), -1) != dep.end()){
      return 0;
    }
  }

  if(!extensive_search) // consider just the specified source
  {
    // check if the dependecies have been satisfied for the packet
    for (auto it = dep.begin(); it != dep.end(); ++it){
      // check in landed packets if the (last) dependency for the packet has been satisfied
      auto match = std::find_if(_executed[source].begin(), _executed[source].end(), [it](const tuple<int,int,int> & p){
        return std::get<0>(p) == *it;
      });
      if (match != _executed[source].end()){
        dep_time.push_back(std::get<2>(*match));
      }
    }
  }
  else{
    // check for the executed workload
    for (auto it = dep.begin(); it != dep.end(); ++it){
      // check in landed packets if the (last) dependency for the packet has been satisfied
      for (int i = 0; i < _nodes; ++i){
        auto match = std::find_if(_executed[i].begin(), _executed[i].end(), [it](const tuple<int,int,int> & p){
          return std::get<0>(p) == *it;
        });
        if (match != _executed[i].end()){
          dep_time.push_back(std::get<2>(*match));
        }
      }
    }
  }

  // if the set is empty, return the maximum time of the dependencies
  if (dep.size() == dep_time.size()){
    return *max_element(dep_time.begin(), dep_time.end());
  }
  else{
    return -1;
  }

};

int DependentInjectionProcess::_dependenciesWSatisfied (const ComputingWorkload * u, int source, bool extensive_search) const{
  // get the vector of dependencies and convert it to a set
  assert(u);
  assert((u->dep.size() > 0));
  set<int> dep(u->dep.begin(), u->dep.end());
  std::vector<int> dep_time;

  if (dep.size() == 1){
    // check wether the only element is -1
    if (find(dep.begin(), dep.end(), -1) != dep.end()){
      return 0;
    }
  }
  if (!extensive_search) // consider just the specified source
  {
     // check if the dependencies have been satisfied for the workload
     for (auto it = dep.begin(); it != dep.end(); ++it){
      // check in the processed packets for the coinsidered source if there are requests whose
      // id matches the dependency: THE PACKET ID MUST MATCH WITH THE DEPENDENCY, AS WELL AS BE A WRITE PACKET
      auto match = std::find_if(_processed->at(source).begin(), _processed->at(source).end(), [it](const tuple<int,int,int> & p){
        return std::get<0>(p) == *it && std::get<1>(p) == commType::WRITE;
      });
      if (match != _processed->at(source).end()){
        dep_time.push_back(std::get<2>(*match));
      }
    }
    
  }
  else{ // consider also all the other sources
    // check if the dependencies have been satisfied for the packet
    for (auto it = dep.begin(); it != dep.end(); ++it){
      // check in the processed packets for the coinsidered source if there are requests whose
      // id matches the dependency: THE PACKET ID MUST MATCH WITH THE DEPENDENCY, AS WELL AS BE A WRITE PACKET
      for (int i = 0; i < _nodes; ++i){
        auto match = std::find_if(_processed->at(i).begin(), _processed->at(i).end(), [it](const tuple<int,int,int> & p){
          return std::get<0>(p) == *it && std::get<1>(p) == commType::WRITE;
        });
        if (match != _processed->at(i).end()){
          dep_time.push_back(std::get<2>(*match));
        }
      }
    }
    
  }
  
  // if the set is empty, return the maximum time of the dependencies
  if (dep.size() == dep_time.size()){
    return *max_element(dep_time.begin(), dep_time.end());
  }
  else{
    return -1;
  }
}

bool DependentInjectionProcess::_resortWaitingPQueues( vector<deque<const Packet *>> & waiting_packets, int source){
  bool resorted = false;
  
  // loop over the waiting packets and keep going until you find a packet whose dependencies are
  // not satisfied yet. Mark this packet and move on with the search: if you find a packet whose
  // dependencies are satisfied, move it in front of the marked packet. Continue this way until you
  // reach the end of the queue. 

  std::deque<const Packet *>::iterator marked = waiting_packets[source].begin();
  int marked_position = 0;
  std::deque<const Packet *> to_move;
  bool assigned = false;
  const Packet * p = nullptr;
  
  if (waiting_packets[source].size() < 2){
    return false;
  }
  for (auto it = waiting_packets[source].begin(); it != waiting_packets[source].end(); ++it){
    if (!assigned){
      if (_dependenciesPSatisfied(*it, source) < 0){
        marked = it;
        assigned = true;
      }
      else{
        marked_position++;
      }
    }
    else{
      if (_dependenciesPSatisfied(*it, source) >= 0){
        // move the packet in front of the marked packet
        to_move.push_back(*it);
        resorted = true;
      }
    }
  }

  assert ((marked == _waiting_packets[source].begin())? (marked_position == 0 || marked_position == waiting_packets[source].size() ): (waiting_packets[source][marked_position])->id == (*marked)->id );
  // move the packets
  for (auto m = to_move.begin(); m != to_move.end(); ++m){
    p = *m;
    // insert the packet in front of the marked packet
    waiting_packets[source].insert(marked, p);
    // increase the marked position
    marked_position++;
    // update the marked iterator
    marked = waiting_packets[source].begin() + marked_position;
    
    
    // seach for the second occurence of the packet in the queue and remove it 
    bool found_first = false;
    for (auto it = waiting_packets[source].begin(); it != waiting_packets[source].end(); ++it){
      if ((*it)->id == p->id){
        if (found_first){
          waiting_packets[source].erase(it);
          break;
        }
        else{
          found_first = true;
        }
      }
    }
    assert(found_first);
  }

  return resorted;
}

void DependentInjectionProcess::_reconfigure(int source){
  // reconfigure the memory of the source node
  _memory_set->reconfigure(source, _waiting_workloads[source], (*_context->gDumpFile));
}

void DependentInjectionProcess::stageReconfiguration(int source, bool bypass_output_check){
  // stage the reconfiguration at source node
  _memory_set->stageReconfiguration(source, _clock->time(), bypass_output_check, _waiting_workloads[source]);

}

void DependentInjectionProcess::stageBatchReconfiguration(int source, bool bypass_output_check){
  
  if (_memory_set->getMemoryUnit(source).checkReconfNeed(bypass_output_check, _waiting_workloads[source])){
    _memory_set->getMemoryUnit(source).reconf_staged = true;
  }

  //check that the reconfiguration is needed by other_reconf_batch_size - 1 nodes
  int count = 0;
  for (int i = 0; i < this->_nodes; ++i){
    if (_memory_set->getMemoryUnit(i).reconf_staged){
      count++;
    }
  }
  if (count >= _reconf_batch_size){
    // in this case we can start the reconfigurations for the first reconf_batch_size nodes
    for (int i = 0; i < this->_nodes; ++i){
      if (_memory_set->getMemoryUnit(i).reconf_staged && count > 0){
        stageReconfiguration(i, bypass_output_check);
        _memory_set->getMemoryUnit(i).reconf_staged = false;
        count--;
      }
    }
  }
  else{
    // if the reconfiguration is not needed for the first reconf_batch_size nodes, we check
    // if some conditions are cleared on the remaining nodes, to avoid blocking behaviour
    bool all_ready = true;
    for (int i = 0; i < this->_nodes; ++i){
      if (_memory_set->getMemoryUnit(i).reconf_staged){
        continue;  // if the node is ready, we skip it
      }
      else{
        // otherwise:
        // 1. check if the node has no more workloads to reconfigure
        // 2. check if the nodes have allocated workloads wich are dependent
        // on the output of workloads that are not yet allocated (to avoid deadlocks)

        //  for 1: if the node has still workloads to reconfigure but the dependencie are satisfied, we set all_ready to false
        //  otherwise, if no more workloads are to be reconfigured, we can leave the all_ready to true

        //  for 2: this is true if all the other workloads on other nodes have been deallocated: meaning that they have already
        // relayed their results to the dependent tasks, but the allocated workloads on the source nodes still lack some replies

        bool no_more_to_reconfigure = _memory_set->getMemoryUnit(i).no_more_to_reconfigure;
        bool dependencies_satisfied = true;
        for (auto & w : _memory_set->getMemoryUnit(i).getCurAllocatedWorkloads()){
          if (_dependenciesWSatisfied(w,i)<0){
            dependencies_satisfied = false;
            break;
          }
        }
        if (!no_more_to_reconfigure && dependencies_satisfied && (_memory_set->getMemoryUnit(i).reconf_staged != true)) // if the node has still workloads to reconfigure, the dependencies are satisfied and the reconfiguration have not yet been scheduled
        {
          all_ready = false;
          break;
        }
      }
    }
    if (all_ready){
      for (int i = 0; i < _nodes; ++i){
        if (_memory_set->getMemoryUnit(i).reconf_staged){
          stageReconfiguration(i, bypass_output_check);
        }
      }
    }
  }

}


bool DependentInjectionProcess::_manageReconfPacketInjection(const Packet * p, int p_dep_time, int source){
  // this method gets called each time a packet can be sent by the node in case of active reconfiguration.

  bool valid = false;
  const Packet * elem = nullptr;

  if (p && p_dep_time >=0){
  
    int src = p->src;
    int dst = p->dst;

    if (p->dep[0] == -1){
      return _managePacketInjection(p);
    }

    if (src == dst){
      // in this case, there is no actual need to inject the packet
      // on the NoC, but we have to mark it as processed
      int type = p->type;
      if (type == 1 || type == 2){
        // if the packet is a write request or a read request, we change the type to the corresponding bulk message
        type = type == 1 ? 5 : 6;
      }        
      _processed->at(src).insert(make_tuple(p->id, type, _clock->time()));
      _waiting_packets[src].pop_front(); // remove the packet from the waiting queue
      *(_context->gDumpFile) << " --( No communication )-- Packet with ID:" << p->id <<" and type " << p->type << " at node " << p->src << " has been processed at time " << _clock->time() << std::endl;
      if (type == 6){
        // finalize the communication and write process for reconfiguration
        finalizeWrite(src, p->dep[0], p->id);
        finalizeCommunication(src, p->dep[0], p->id);

      }
    }
    else{
      _traffic->cur_packet = p;
      _waiting_packets[src].pop_front(); // remove the packet from the waiting queue
      //no need to set p to pending packet, as it will be injected direcly
      valid = true;
      *(_context->gDumpFile) << "Packet with ID:" << _traffic->cur_packet->id <<" and type " << _traffic->cur_packet->type << " at node " << p->src << " has been injected at time " << _clock->time() << std::endl;
    }
  }
  return valid;

}

bool DependentInjectionProcess::_managePacketInjection(const Packet * p){
  bool valid = false;
  if (p->src == p->dst){
    // in this case, there is no actual need to inject the packet
    // on the NoC, but we have to mark it as processed

    int type = p->type;
    if (type == 1 || type == 2){
      // if the packet is a write request or a read request, we change the type to the corresponding bulk message
      type = type == 1 ? 5 : 6;
    }        
    _processed->at(p->src).insert(make_tuple(p->id, type, _clock->time()));
    _waiting_packets[p->src].pop_front(); // remove the packet from the waiting queue
    *(_context->gDumpFile) << " --( No communication )--  Packet with ID:" << p->id <<" and type " << p->type << " at node " << p->src << " has been processed at time " << _clock->time() << std::endl;

  }
  else{
    _traffic->cur_packet = p;
    _waiting_packets[p->src].pop_front(); // remove the packet from the waiting queue
    //no need to set p to pending packet, as it will be injected direcly
    valid = true;
    *(_context->gDumpFile) << "Packet with ID:" << _traffic->cur_packet->id <<" and type " << _traffic->cur_packet->type << " at node " << p->src << " has been injected at time " << _clock->time() << std::endl;
  }
  return valid;

}


// the method is to be called within the constructor
void DependentInjectionProcess::_buildStaticWaitingQueues(){

  // starting from the first packet in the list, loop over the packets
  // and build the waiting queues for each source node
  _traffic->reset();

  // ==================  RECONFIGURATION ==================
  // buffers to compute the scheduled reconfigurations
  std::vector<int> running_filled_iospace(_nodes, 0); // for input/output
  std::vector<int> running_filled_wspace(_nodes, 0); // for weights
  bool use_freed_up_space = false; // a flag to enable the use of the freed up space
  // ==================  RECONFIGURATION ==================
  

  if (_logger) {
    _logger->initialize_event_info();
  }

  // STUPID COMPILING OPTION, MAY DECIDE TO UPGRADE LATER
  if (_traffic->getNextPacket() == nullptr){
    _traffic->reached_end_packets = true;
  }
  if (_traffic->getNextWorkload() == nullptr){
    _traffic->reached_end_workloads = true;

  }

  while(_traffic->reached_end_packets == false){
    const Packet * p = _traffic->getNextPacket();
    assert(p);
    assert((p->size > 0));
    assert((p->src >= 0) && (p->src < _nodes));
    int source = p->src;
    // make space for the packet
    _waiting_packets[source].emplace_back(p);
    _traffic->updateNextPacket();
    (*_context->gDumpFile) << "Packet " << p->id << " with size " << p->size << " inserted in queue at node " << source << std::endl;

    // create a EventInfo object for each packet
    if (_logger) {
      commType ctype = intToCommType(p->type);
      TrafficEventInfo * tei = new TrafficEventInfo(p->id, ctype, p->src, p->dst, p->size);
      _logger->add_tevent_info(tei);
    }
      
  }

  while(_traffic->reached_end_workloads == false){
    const ComputingWorkload * w = _traffic->getNextWorkload();
    assert(w);
    assert((w->ct_required > 0));
    assert((w->node >= 0) && (w->node < _nodes));
    int source = w->node;
    (*_context->gDumpFile) << "Workload " << w->id << " with size " << w->size << " inserted in queue at node " << source << std::endl;
    _waiting_workloads[source].emplace_back(w);
    _traffic->updateNextWorkload();

    // create a EventInfo object for each workload
    if (_logger) {
      ComputationEventInfo * cei = new ComputationEventInfo(w->id, w->node, w->ct_required);
      _logger->add_tevent_info(cei);
    }
  }

  _traffic->reset();
  _memory_set->init(_waiting_workloads);

  if (_enable_reconf){

    // for each node
    for (int i = 0; i < _nodes; ++i){
      // for each workload and packet, we initialize the waitngForReply registers
      _memory_set->getMemoryUnit(i).initWaitingForReply(_waiting_workloads[i], _waiting_packets[i]);
    }
    
    // load the first workloads in memory
    for (int i = 0; i < _nodes; ++i){
      _reconfigure(i); // for the first recofiguration, we don't care about the 
      // time the reconfiguration takes: we are assuming the workload are preloaded
    }
  }
}

void DependentInjectionProcess::reset()
{
  _memory_set->allReset();
  _npu_set->allReset();
}


void DependentInjectionProcess::addToWaitingQueue(int source, Packet * p)
{
  // starting from the first packet in the list, iterate over the packets
  // and append the new packet before the first one with 0 priority
  p->priority = 1;

  auto it = _waiting_packets[source].begin();
  while (it != _waiting_packets[source].end()){
    // first check if the packet dependecy as been cleared
    if (_dependenciesPSatisfied(*it, source) == -1 && _dependenciesPSatisfied(p, source) >= 0){
      // if the dependency for the packet in the waiting queue has not been satisfied,
      // we can insert the packet before it
      break; 
    }
    if ((*it)->priority == 0){
      break;
    }
    it = std::next(it);
  }

  _waiting_packets[source].insert(it, p); // INSERT AFTER

  if (_logger) {
    commType ctype = intToCommType(p->type);
    TrafficEventInfo * tei = new TrafficEventInfo(p->id, ctype, p->src, p->dst, p->size);
    _logger->add_tevent_info(tei);
  }

}

bool DependentInjectionProcess::test(int source)
{

  // check that there are still waiting packets to inject
  // if there are none, _reached_end is set to true
  int count = 0;
  for(int i = 0; i < _nodes; i++){
    if(_waiting_packets[i].empty() && _waiting_workloads[i].empty() && _pending_workloads[i] == nullptr){
      count++;
    }
    if(count == _nodes  && _memory_set->allIdle() && _npu_set->allIdle()){
      reached_end = true;
    }   
  }

  bool valid = false;
  const ComputingWorkload * w = nullptr;
  const Packet * p = nullptr;

  // stall if there are workloads being processed or reconfigured
  if (_npu_set->getNPU(source).required_time>0){
    assert(_npu_set->getNPU(source).busy);
    assert(!(_pending_workloads[source] == nullptr));
    // check if the computation has been completed
    // if true, we reset the NPU timer and finalize computation, else we stall
    if (_npu_set->checkComputation(source, _clock->time())){
      _npu_set->finalizeComputation(source, _pending_workloads[source], _clock->time());
      *(_context->gDumpFile) << "Workload with ID:" << _pending_workloads[source]->id << " at node " << source << " has been processed at time " << _clock->time() << std::endl;
      _executed[source].insert(make_tuple(_pending_workloads[source]->id, _pending_workloads[source]->type, _clock->time()));

      // enable resorting of the waiting queues for the packets
      if (_resorting || _enable_reconf){
        if (_resortWaitingPQueues(_waiting_packets, source)){
          *(_context->gDumpFile) << "Resorted the waiting packets at node " << source << std::endl;
        }
      }

      if (_enable_reconf){
        // delete from memory the workload that has been processed
        if (_pending_workloads[source] != nullptr){
          int prev_mem = _memory_set->getAvailable(source);
          _memory_set->deallocate(source, _pending_workloads[source], true);
          int new_mem = _memory_set->getAvailable(source);
          *(_context->gDumpFile) << "DEALLOCATED WORKLOAD - id : " <<  _pending_workloads[source]->id <<",  size : " << new_mem - prev_mem << " from node: " << source << " at time: " << _clock->time() << "(prev mem : " << prev_mem << ", new mem: " << new_mem << ")" << std::endl;
        }

        // if no packet is dependent on the workload, check directly for reconfiguration, bypassing the output memory check
        if(_memory_set->getMemoryUnit(source).checkReplyReceived(_pending_workloads[source]->id)){
          stageReconfiguration(source, true); // border case (should only happen for example runs, so in this case we don't include the batch extension)
        }
      }
      _pending_workloads[source] = nullptr;

    }
    return valid;
  }
  else if (_memory_set->getMemoryUnit(source).required_time>0){
    assert(_memory_set->getMemoryUnit(source).reconf_active);
    // check if the reconfiguration has been completed
    // if true, we reset the memory timer and perform reconfiguration, else we stall
    if(_memory_set->checkReconfiguration(source, _clock->time())){
      _memory_set->reconfigure(source, _waiting_workloads[source], *(_context->gDumpFile));
    }
  }

  // if no workload or reconfigurations must be processed.
  // get the next packet and workloads in the queue
  if(!(_waiting_packets[source].empty())){
    p = _waiting_packets[source].front();
    assert(p);
    assert((p->size > 0));
    assert((p->src == source));
  }
  if (!(_waiting_workloads[source].empty())){
    w =_waiting_workloads[source].front();
    assert(w);
    assert((w->ct_required > 0));
    assert(w->node == source);
  }
  
  if (p == nullptr && w == nullptr && _pending_workloads[source] == nullptr ){
    return valid;
  }

  assert((source >= 0) && (source < _nodes));

  int dep_time_w = -1;
  int dep_time_p = -1;
  if (!(w == nullptr)){
    dep_time_w = _dependenciesWSatisfied(&(*w), source);
  }
  if (!(p == nullptr)){
    dep_time_p = _dependenciesPSatisfied(&(*p), source); 
  }


  // the new workload/packet can be executed only if its dependecies (packets and workloads) have been satisfied
  if (dep_time_p>=0 && source == p->src && !(p == nullptr)){
    // the node is idle and can process the packet request
    assert(_npu_set->getNPU(source).busy == false);
    // the packet has already been cleared for the dependencies, we can inject
    valid = _enable_reconf ? _manageReconfPacketInjection(p, dep_time_p, source) : _managePacketInjection(p);
    if (valid == true){
      return valid;
    }
    
  }
  if(dep_time_w>=0 && source == w->node && !(w==nullptr) && ((_enable_reconf)?_memory_set->isWorkloadAllocated(source, w):1)){
    assert(_npu_set->getNPU(source).busy == false && _memory_set->getMemoryUnit(source).reconf_active == false);
    // the node is idle and can process the workload
    assert(_pending_workloads[source] == nullptr);

    _pending_workloads[source] = w;
    _waiting_workloads[source].pop_front(); // remove the workload from the waiting queue
    _npu_set->startComputation(source, w, _clock->time());
    *(_context->gDumpFile) << "Workload with ID:" << w->id << " at node " << source << " has started processing at time " << _clock->time() << std::endl;
  
  }

  return valid;
}

// ============================================================================================================