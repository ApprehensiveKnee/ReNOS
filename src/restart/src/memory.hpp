/*
* ==================================================
* File: memory.hpp
* Project: src
* File Created: Wednesday, 19th March 2025
* Author: Edoardo Cabiati (edoardo.cabiati@mail.polimi.it)
* Under the supervision of: Politecnico di Milano
* ==================================================
*/


/*
 The following header file hosts the definition of a very high level implementation of the local memory unit
 available for each NPU used in the simulation for the UserDefinedInjection class.
 */


 #ifndef MEMORY_HPP
 #define MEMORY_HPP
 
 
 #include <deque>
 #include <algorithm>
 #include <set>
 #include <iostream>
 #include <cassert>
 #include <cmath>
 #include "config.hpp"
 #include "logger.hpp"
 
 /*
 A class to store the reconfiguration rates to compute the times needed for reconfigurations.
 This provides a very high level description.
 */
 class MemoryRateRegister{
     private:
         double _reconf_cycles; // number of cycles needed for reconfiguration of a single byte
 
     public:
         // constructor
         MemoryRateRegister( double reconf_cycles /* [cycle/byte] */) : _reconf_cycles(reconf_cycles) {}
 
         int getReconfCycles(int size /* [ byte size ] */) const {
             return std::ceil(double(size) *  _reconf_cycles);
         }
 };
 
 /*
 The following class will be used to model the memory unit of the PE (SRAM).
 For initialization, the class will be given an integer, representing the size (in bytes) of the local 
 memory unit available to the single NPU, and the number of bytes we would like to reserve for reconfiguration 
 purposes. The rest of space will be used to store partial results from other tasks
 */
 class MemoryUnit {
 
     private:
         int _size; // total size of the memory
         int _available; // total memory available
         int _threshold_bytes; // number of bytes to reserve for reconfiguration purposes
         
         std::set<const ComputingWorkload *> _allocated_workloads; // workloads currently allocated
         std::set<const ComputingWorkload *> _allocated_outputs; // output of the workloads currently allocated
         std::map<int, std::set<const ComputingWorkload *>> _temp_stored_results; // a dictionary to store information on the partial results that need to be dealloacted after reconfiguration of corresponding workloads
         std::map<int,std::set<int>> _waiting_for_reply; // a dictionary: its used to determine when its safe to deallocate the workload output on the source node.
 
         MemoryRateRegister * _reg;
     
     public:
 
         // to manage the times for reconfigurations, as for NPUs, we add a few more fields:
         int required_time;
         int start_time;
         bool reconf_active;
 
         bool no_more_to_reconfigure; // flag to signal that no more reconfigurations are needed
         bool reconf_staged;
 
         // constructors
         // ========================================
         MemoryUnit(int size, int threshold_bytes, MemoryRateRegister* reg);
         MemoryUnit(int size, float threshold, MemoryRateRegister* reg);
 
         // destructor
         // ========================================
         ~MemoryUnit() {};
 
         // getters
         // ========================================
         int getSize() const { return _size; }
         int getAvailable() const { return _available; }
         int getThresholdBytes() const { return _threshold_bytes; }
         int getAvailableForReconf() const { int available = std::min(_available, _threshold_bytes); 
                                             assert(available >= 0); return available; }
 
         int getNumCurAllocatedWorkloads() const { return _allocated_workloads.size(); }
         int getNumCurAllocatedOutputs() const {return _allocated_outputs.size(); }
         std::set<const ComputingWorkload *> & getCurAllocatedWorkloads() { return _allocated_workloads; }
         std::set<const ComputingWorkload *> & getCurAllocatedOutputs() { return _allocated_outputs; }
         std::map<int, std::set<const ComputingWorkload *>> & getTempStoredResults() { return _temp_stored_results; }
         std::map<int,std::set<int>> & getWaitingForReply() { return _waiting_for_reply; }
         
         // printers
         // ========================================
         void printCurAllocatedWorkloads(std::ostream & os) const;
         void printCurAllocatedOutputs(std::ostream & os) const;
         void printTempStoredResults(std::ostream & os) const;
         void printWaitingForReply(std::ostream & os) const;
         
         // memory management methods
         // ========================================
         
         // a method to check if the memory is empty
         bool isEmpty(){return _available == _size;}
 
         // a method to allocate a certain amount of memory
         void allocate_(int size);
 
         // a method to deallocate a certain amount of memory
         void deallocate_(int size);
 
         // a method to allocate the space required for the output of an associated workload
         void allocateOutput(const ComputingWorkload * w);
 
         // a method to deallocate the space for the output of the workload
         void deallocateOutput(const ComputingWorkload * w);
 
         // A method to allocate the space for a workload:
         // If the memory is available, the workload will be added to the list of allocated workloads
         // and the available memory will be decreased by the size of the workload.
         // Additionally, if the optimize flag is set to true, the method will try to optimize memory allocation
         // and this will be taken into account in the available memory calculation: specifically, if optimize is set to 
         // true, the method will snoop at the input_range of the workload and compare it with the input_range 
         // of the already allocated workloads. If some input overlapping is detected
         // the method will subtract the overlapping size from the size of memory that needs to be allocated
         // for the workload and then proceed with the allocation.
         // Debugging is still needed for this last part
         void allocate(const ComputingWorkload * w, bool optimize = false);
 
         // A corresponding method to deallocate the memory for the workload.
         // If the keep_output flag is set to true, the method will not deallocate the output space of the workload.
         void deallocate(const ComputingWorkload * w, bool keep_output = true, bool optimize = false);
 
         // a method to reset the memory unit
         void reset();
             
         // a method to remove the output placeholder from the memory
         bool removeOutputPlaceholder(int w_id);
 
         // a method to check for the memory availability
         bool checkMemoryAvail(int size);
 
         // a method to check if the workload is currently allocated
         bool isWorkloadAllocated(const ComputingWorkload * w);
 
         // a method to check if the output associated to a workload is currently allocated on the memory
         bool isOutputAllocated(const ComputingWorkload * w);
 
 
         // register management functions
         // ========================================
 
         // -*- WAITING FOR REPLY REGISTER -*-
 
         // a method to create an entry for the workload in the _waiting_for_reply map
         void createWaitingForReplyEntry(int workload_id);
 
         // a method to add a packet to the register for messages that are expected to be received
         void addReplyExpected(int workload_id, int packet_id);
 
         // a method to delete a packet from the register for messages that are expected to be received
         void markReplyReceived(int workload_id, int packet_id);
 
         //a method to delete an entry from the register for messages that are expected to be received
         void removeWatingForReplyEntry(int workload_id);
 
         // a method to check if there are still replies expected for a workload
         bool checkReplyReceived(int workload_id);
 
         // -*- TEMPORARY STORED RESULTS REGISTER -*-
 
         // a method to create an entry for the workload in the _temp_stored_results map
         void createTempStoredResultsEntry(int workload_id);
 
         // a method to add a workload to the register for partial results that need to be deallocated
         void addTempStoredResults(int workload_id, const ComputingWorkload * w);
 
         // a method to mark an output of a workload as deleted after reconfiguration
         void markResultDeleted(int workload_id, const ComputingWorkload * w);
 
         // a method to delete a workload from the register for partial results that need to be deallocated
         void removeTempStoredResults(int workload_id);
 
         // a method to check if a workload has partial results that need to be deallocated
         bool checkTempStoredResults(int workload_id);
 
 
         // reconfiguration functions
         // ========================================
 
         // a method to reset the timer for reconfiguration
         void resetTimer();
 
         // a method to check the status of the reconfiguration
         bool checkReconfiguration(int current_time){ return start_time + required_time < current_time; }
         
         // a method to determine if the node is busy reconfiguring
         bool isIdle(){ assert((required_time > 0) == reconf_active); return !reconf_active; }
 
         // a method initialize the _waiting_for_reply map
         void initWaitingForReply(const std::deque<const ComputingWorkload *> & waiting_workloads, const std::deque<const Packet *> & waiting_packets);
 
         // a method to compute the next batch size for the reconfiguration:
         // the method returns the size of the next batch of workloads, while
         // the head (not modified) is the first workload in the queue that needs to be reconfigured
         // and tail (modified) is the last workload in the queue that can be reconfigured
         int computeNextBatchSize(const ComputingWorkload * head, const std::deque<const ComputingWorkload *> & waiting_workloads);
 
         // a method used to check if the workload is in the next reconfiguration batch:
         // to determine this, we need to check for the availability of the memory.
         bool inNextReconfigBatch(const ComputingWorkload * w, const std::deque<const ComputingWorkload *> & waiting_workloads);
 
         // a method to check if the conditions to start reconfiguration have been cleared
         bool checkReconfNeed(bool bypass_output_check, const std::deque<const ComputingWorkload *> & waiting_workloads);
 
         // a method to stage the reconfiguration: this method implicitly sets the timer if the conditions are met
         void stageReconfiguration(int current_time, bool bypass_output_check, const std::deque<const ComputingWorkload *> & waiting_workloads);
 
         // a method to pefrom the reconfiguration: this method resets the timer un updates the current workloads and outputs
         void reconfigure( const ComputingWorkload *& head, const std::deque<const ComputingWorkload *> & waiting_workloads, std::ostream & os);
 
         // handy functions
         // ========================================
         
         // a method to check if two workloads have overlapping in the input space
         static int in_overlap_in(const ComputingWorkload * w1,const ComputingWorkload * w2);
 
         // a method to check if two workloads have overlapping in the output space
         static int out_overlap_in(const ComputingWorkload * w1,const ComputingWorkload * w2);
 };
 
 
 /*
 The class will be used to manage the set of local memories for the PEs, acting as a central manager for 
 the memory units.
 */
 class MemorySet {
 
     private:
         int _nodes;
         int _size;
         std::vector<const ComputingWorkload *> _pointed_workload_in_queue; // a pointer to the next workload in the queue to be included in the next reconfiguration
         std::vector<MemoryUnit> _memory_units;
 
         EventLogger * _logger;
 
     public: 
 
         // constructors
         // ========================================
         MemorySet(int nodes, int size, int threshold, MemoryRateRegister * reg, EventLogger * logger);
         MemorySet(int nodes, int size, float threshold, MemoryRateRegister * reg, EventLogger * logger);
 
         // getters
         // ========================================
         MemoryUnit & getMemoryUnit(int node){return _memory_units[node];}
         int getSize() const { return _size; }
         const ComputingWorkload * getHeadWorkload(int node){return _pointed_workload_in_queue[node];}
 
         // setters
         // ========================================
         void setHeadWorkload(int node, const ComputingWorkload * w){_pointed_workload_in_queue[node] = w;}
 
 
         // functional methods
         // ========================================
 
         // a method to initialize the memory set head pointers based on the waiting workloads
         void init(std::vector<std::deque<const ComputingWorkload * >> & waiting_workloads);
 
         // a method to check if all the memory units are empty
         bool allEmpty(){
         for (int i = 0; i < _nodes; ++i){
             if (!_memory_units[i].isEmpty()){
             return false;
             }
         }
         return true;
         }
 
         // a method to reset all the memory units
         void allReset(){
             for (int i = 0; i < _nodes; ++i){
                 _memory_units[i].resetTimer();
             }
         }
 
         bool allIdle(){
             for (int i = 0; i < _nodes; ++i){
                 if (!_memory_units[i].isIdle()){
                     return false;
                 }
             }
             return true;
         }
 
         // hooks
         // ========================================
 
 
         // a method to check if the memory is hosting a workload
         bool isWorkloadAllocated(int node, const ComputingWorkload * w){return _memory_units[node].isWorkloadAllocated(w);}
 
         // a method to check if the memory is hosting the output of a workload
         bool isOutputAllocated(int node, const ComputingWorkload * w){return _memory_units[node].isOutputAllocated(w);}
 
         // a method to check if a specific memory unit is empty
         bool isEmpty(int node){
             return _memory_units[node].isEmpty();
         }
     
         void reset(int node){
             _memory_units[node].reset();
         }
 
         bool checkReconfiguration(int node, int current_time){
             return _memory_units[node].checkReconfiguration(current_time);
         }
 
         bool checkMemoryAvail(int node, int size){
             return _memory_units[node].checkMemoryAvail(size);
         }
 
         int getAvailable(int node) const { return _memory_units[node].getAvailable(); }
 
         void allocate_(int node, int size){
             _memory_units[node].allocate_(size);
         }
 
         void deallocate_(int node, int size){
             _memory_units[node].deallocate_(size);
         }
 
         void allocateOutput(int node, const ComputingWorkload * w){
             _memory_units[node].allocateOutput(w);
         }
 
         void deallocateOutput(int node, const ComputingWorkload * w){
             _memory_units[node].deallocateOutput(w);
         }
 
         void allocate(int node, const ComputingWorkload * w, bool optimize = false){
             _memory_units[node].allocate(w, optimize);
         }
 
         
         void deallocate(int node, const ComputingWorkload * w, bool keep_output , bool optimize = false){
             _memory_units[node].deallocate(w, keep_output, optimize);
         }
         
         // reconfiguration methods
         // ========================================
 
         // stage reconfiguration hook
         void stageReconfiguration(int node, int current_time, bool bypass_output_check, const std::deque<const ComputingWorkload *> & waiting_workloads){
             bool init_state = _memory_units[node].reconf_active;
             _memory_units[node].stageReconfiguration(current_time, bypass_output_check, waiting_workloads);
             if (_logger && !init_state && _memory_units[node].reconf_active){
                 _logger->register_event(EventType::START_RECONFIGURATION, current_time, node);
             }
         }
 
         // reconfiguration hook
         void reconfigure(int node, const std::deque<const ComputingWorkload *> & waiting_workloads, std::ostream & os){
             os << "=============================" << std::endl;
             os << "RECONFIGURING MEMORY UNIT " << node << std::endl;
             os << "=============================" << std::endl;
             
             const ComputingWorkload * head = getHeadWorkload(node);
             _memory_units[node].reconfigure(head, waiting_workloads, os);
             if (_logger){
                 _logger->register_event(EventType::END_RECONFIGURATION, _memory_units[node].start_time + _memory_units[node].required_time, node);
             }
             // unset the reconfiguration flag and reset the timer
             _memory_units[node].resetTimer();
             // reset the head pointer (head is modified within the reconfigure method)
             setHeadWorkload(node, head);
 
         }
 
 
 };
 
 
 #endif