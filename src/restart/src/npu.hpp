/*
* ==================================================
* File: npu.hpp
* Project: src
* File Created: Wednesday, 19th March 2025
* Author: Edoardo Cabiati (edoardo.cabiati@mail.polimi.it)
* Under the supervision of: Politecnico di Milano
* ==================================================
*/

/*
The following header file hosts the definition of a very high level implementation of the NPU
(processing elements) used in the simulation for the UserDefinedInjection class.
*/

#ifndef NPU_HPP
#define NPU_HPP
#include <map>
#include <cassert>
#include "logger.hpp"
#include "config.hpp"

// workload types
enum class WorkloadType{
    ANY, // not specified
    CONV, // convolutional wokload
    FC, // fully connected workload
};


// NPU Register class referenced by the NPU unit class to store the computation cycles for each type of workload
class NPURateRegister{

    public:
        // default constructor
        NPURateRegister() {};
        // default destructor
        ~NPURateRegister() {};

        // a method to register a new entry in the map
        void registerWorkloadCycles(WorkloadType type, float cycles);

        // a method to get the total number of cycles given the size (in FLOPs) and type of workload
        int getWorkloadCycles(int size_FLOPs, WorkloadType type = WorkloadType::ANY) const;
        
    private:
        // a map to store the computation cycles for each type of workload
        std::map<WorkloadType, double> _workload_cycles;

};

class NPU{
    public:

        int required_time;
        int start_time;
        bool busy;

        // constructor
        NPU(NPURateRegister * reg): _reg(reg), required_time(0), start_time(0), busy(false) {
            assert(_reg);
        };
        // destructor
        ~NPU() {};
        
        void setTimer(int size, int start_time,  WorkloadType type = WorkloadType::ANY);

        void resetTimer();
        
        bool isIdle() const { assert((required_time > 0) == busy); return !busy; }
        // ==================================================

        void startComputation(const ComputingWorkload * w, int current_time){
            setTimer(w->size, current_time);
        }

        bool checkComputation(int current_time){
            return start_time + required_time < current_time;
        }

    private:
        // a pointer to the register class
        NPURateRegister * _reg;

        int _cyclesWorkload(int size, WorkloadType type = WorkloadType::ANY) const {
            return _reg->getWorkloadCycles(size, type);
        }
};

// a class to group the NPUs (one for each node) in a set
class NPUSet{

    private:
        int _nodes;
        std::vector<NPU> _npus;
        EventLogger * _logger;

    public:
        // constructor
        NPUSet(int nodes, NPURateRegister * reg, EventLogger * logger): _nodes(nodes), _logger(logger){
            for (int i = 0; i < nodes; ++i){
                _npus.push_back(NPU(reg));
            }
        }
        // destructor
        ~NPUSet() {};

        // a method to get the NPU at a specific node
        NPU & getNPU(int node){
            return _npus[node];
        }

        // functional methods
        // ========================================
        void allReset(){
            for (int i = 0; i < _nodes; ++i){
                _npus[i].resetTimer();
            }
        }

        bool allIdle(){
            for (int i = 0; i < _nodes; ++i){
                if (!_npus[i].isIdle()){
                    return false;
                }
            }
            return true;
        }

        // hooks
        // ========================================
        void reset(int node){
            _npus[node].resetTimer();
        }

        void startComputation(int node, const ComputingWorkload * w, int current_time){
            _npus[node].startComputation(w, current_time);
            if (_logger) {
                _logger->register_event(EventType::START_COMPUTATION, current_time, w->id);
            }
        }

        bool checkComputation(int node, int current_time){
            return _npus[node].checkComputation(current_time);
        }

        void finalizeComputation(int node, const ComputingWorkload * w, int current_time){
            
            if (_logger) {
                _logger->register_event(EventType::END_COMPUTATION, current_time, w->id);
            }
            _npus[node].resetTimer();
        }

};


#endif