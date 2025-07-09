/*
* ==================================================
* File: memory.cpp
* Project: src
* File Created: Wednesday, 19th March 2025
* Author: Edoardo Cabiati (edoardo.cabiati@mail.polimi.it)
* Under the supervision of: Politecnico di Milano
* ==================================================
*/



#include "memory.hpp"

// -*- MEMORY UNITS METHODS -*- //

// constructors
// ========================================
MemoryUnit::MemoryUnit(int size, int threshold_bytes,MemoryRateRegister* reg): 
_reg(reg), 
_size(size),
_available(size), 
_threshold_bytes(threshold_bytes),
no_more_to_reconfigure(false){
    assert(_size > 0);
    assert(_threshold_bytes >= 0);
    assert(_threshold_bytes <= _size);
    assert(_reg);
    resetTimer();
}

MemoryUnit::MemoryUnit(int size, float threshold, MemoryRateRegister* reg): 
_reg(reg), 
_size(size), 
_available(size), 
_threshold_bytes(threshold*size), 
no_more_to_reconfigure(false){
    assert(_size > 0);
    assert(_threshold_bytes >= 0);
    assert(_threshold_bytes <= _size);
    assert(_reg);
    resetTimer();
}

// printers
// ========================================
void MemoryUnit::printCurAllocatedWorkloads(std::ostream & os) const{
    os << "Current allocated workloads: " << std::endl;
    for (auto & w : _allocated_workloads){
        os << "Workload ID: " << w->id << std::endl;
    }
    os << std::endl;
}

void MemoryUnit::printCurAllocatedOutputs(std::ostream & os) const{
    os << "Current allocated outputs: " << std::endl;
    for (auto & w : _allocated_outputs){
        os << "Workload ID: " << w->id << std::endl;
    }
    os << std::endl;
}

void MemoryUnit::printTempStoredResults(std::ostream & os) const{
    os << "Temporaty stored results: " << std::endl;
    // loop over the map keys
    for (auto & [k, v] : _temp_stored_results){
        os << "Workload to load ID: " << k << std::endl;
        os << "Dependent workloads whose results have been stored: ";
        for (auto & p : v){
            os << p << ", ";
        }
        os << std::endl;
    }
}

void MemoryUnit::printWaitingForReply(std::ostream & os) const{
    os << "Waiting for reply: " << std::endl;
    // loop over the map keys (the ids of the workloads)
    for (auto & [k, v] : _waiting_for_reply){
        // for each workload, print the ids of the packets whose reply is expected
        os << "Output ID: " << k << std::endl;
        os << "Packets whose reply is expected: ";
        for (auto & p : v){
            os << p << ", ";
        }
        os << std::endl;
    }
}
// functional methods
// ========================================

void MemoryUnit::allocate_(int size){
    // assert that the size is less than the available memory and total size
    assert(("Not enough memory to allocate", size <= _size && _available - size > 0));
    _available -= size;
}

void MemoryUnit::deallocate_(int size){
    assert(("Memory out of bounds", size <= _size && _available>=0));
    _available += size;
}

void MemoryUnit::allocateOutput(const ComputingWorkload * w){
    int output_size = 1;
    for (int i = 0; i < w->output_range.lower_sp.size(); ++i){
        output_size *= (w->output_range.upper_sp[i] - w->output_range.lower_sp[i]);
    }
    if (w->output_range.channels.size() > 0)
        output_size *= (w->output_range.channels[1] - w->output_range.channels[0]);

    allocate_(output_size);
}

void MemoryUnit::deallocateOutput(const ComputingWorkload * w){
    int output_size = 1;
    for (int i = 0; i < w->output_range.lower_sp.size(); ++i){
        output_size *= (w->output_range.upper_sp[i] - w->output_range.lower_sp[i]);
    }
    if (w->output_range.channels.size() > 0)
        output_size *= (w->output_range.channels[1] - w->output_range.channels[0]);

    deallocate_(output_size);
}

void MemoryUnit::allocate(const ComputingWorkload * w, bool optimize){
    int size_to_allocate = w->size;

    if (optimize){
        // check for overlapping in the input space
        int in_overlap = 0;
        for (auto & ow : _allocated_workloads){
            if ( w->layer == ow->layer){
            int in_overlap_with_workload = 0;
            in_overlap_with_workload = in_overlap_in(w, ow);
            assert(in_overlap_with_workload > 0);
            in_overlap += in_overlap_with_workload;
            }
        }
        // subtract the overlapping space from the size of the workload
        size_to_allocate -= in_overlap;
    }

    assert(("Not enough memory to allocate", size_to_allocate <= _available));
    _allocated_workloads.insert(w);
    _allocated_outputs.insert(w);
    allocate_(size_to_allocate);
    
}

void MemoryUnit::deallocate(const ComputingWorkload * w, bool keep_output, bool optimize){
    int output_size = 1;
    for (int i = 0; i < w->output_range.lower_sp.size(); ++i){
        output_size *= (w->output_range.upper_sp[i] - w->output_range.lower_sp[i]);
    }
    if (w->output_range.channels.size() > 0)
        output_size *= (w->output_range.channels[1] - w->output_range.channels[0]);

    int size_to_deallocate = w->size - output_size;
    assert(size_to_deallocate >= 0);

    auto it = _allocated_workloads.find(w);
    assert(("Workload not found in the allocated workloads", it != _allocated_workloads.end()));
    
    if (optimize){
        // check for overlapping in the input space
        int in_overlap = 0;
        for (auto & ow : _allocated_workloads){
            if ( w->layer == ow->layer){
                int in_overlap_with_workload = 0;
                in_overlap_with_workload = in_overlap_in(w, ow);
                assert(in_overlap_with_workload > 0);
                in_overlap += in_overlap_with_workload;
            }
        }
        size_to_deallocate -= in_overlap;
    }
    _allocated_workloads.erase(it);
    if (!keep_output){
        deallocate_(output_size);
    }
    deallocate_(size_to_deallocate);
}
    

void MemoryUnit::reset(){
    _available = _size;
    _allocated_workloads.clear();
    _allocated_outputs.clear();
    _temp_stored_results.clear();
    _waiting_for_reply.clear();
    no_more_to_reconfigure = false;
    resetTimer();
    
}

bool MemoryUnit::removeOutputPlaceholder(int w_id){
    // search for the output in the set of allocated outputs:
    // remove it is found, else return false
    auto it = std::find_if(_allocated_outputs.begin(), _allocated_outputs.end(), [w_id](const ComputingWorkload * w){
        return w->id == w_id;
    });
    if (it != _allocated_outputs.end()){
        _allocated_outputs.erase(it);
        return true;
    }
    return false;
}

bool MemoryUnit::checkMemoryAvail(int size){
    if (size <= _available){
    return true;
    }
    return false;
    
}

bool MemoryUnit::isWorkloadAllocated(const ComputingWorkload * w){
    auto it = _allocated_workloads.find(w);
    return it != _allocated_workloads.end();
}

bool MemoryUnit::isOutputAllocated(const ComputingWorkload * w){
    auto it = _allocated_outputs.find(w);
    return it != _allocated_outputs.end();
}


// register management functions
// ========================================

void MemoryUnit::createWaitingForReplyEntry(int workload_id){
    if (_waiting_for_reply.find(workload_id) == _waiting_for_reply.end())
        _waiting_for_reply[workload_id] = std::set<int>();
}

void MemoryUnit::addReplyExpected(int workload_id, int packet_id){
    assert(("The packet is already in the waiting-for-reply register", _waiting_for_reply.at(workload_id).find(packet_id) == _waiting_for_reply.at(workload_id).end()));
    _waiting_for_reply.at(workload_id).insert(packet_id);
}


void MemoryUnit::markReplyReceived(int workload_id, int packet_id){
    auto it = _waiting_for_reply.find(workload_id);
    assert(("Workload not found in the waiting-for-reply register", it != _waiting_for_reply.end()));
    auto it2 = it->second.find(packet_id);
    assert(("Packet not found in the waiting-for-reply register", it2 != it->second.end()));
    it->second.erase(it2);
}

void MemoryUnit::removeWatingForReplyEntry(int workload_id){
    auto it = _waiting_for_reply.find(workload_id);
    assert(("Workload not found in the waiting-for-reply register", it != _waiting_for_reply.end()));
    _waiting_for_reply.erase(workload_id);
}

bool MemoryUnit::checkReplyReceived(int workload_id){
    return _waiting_for_reply.at(workload_id).empty();
}

void MemoryUnit::createTempStoredResultsEntry(int workload_id){
    if (_temp_stored_results.find(workload_id) == _temp_stored_results.end())
        _temp_stored_results[workload_id] = std::set<const ComputingWorkload *>();
}

void MemoryUnit::addTempStoredResults(int workload_id, const ComputingWorkload * w){
    assert(("The workload output is already in the temporary stored results register", _temp_stored_results.at(workload_id).find(w) == _temp_stored_results.at(workload_id).end()));
    _temp_stored_results.at(workload_id).insert(w);
}

void MemoryUnit::markResultDeleted(int workload_id, const ComputingWorkload * w){
    auto it = _temp_stored_results.find(workload_id);
    assert(("Workload not found in the temporary stored results register", it != _temp_stored_results.end()));
    auto it2 = it->second.find(w);
    assert(("Workload output not found in the temporary stored results register", it2 != it->second.end()));
    it->second.erase(it2);
}

void MemoryUnit::removeTempStoredResults(int workload_id){
    auto it = _temp_stored_results.find(workload_id);
    assert(("Workload not found in the temporary stored results register", it != _temp_stored_results.end()));
    _temp_stored_results.erase(workload_id);
}

bool MemoryUnit::checkTempStoredResults(int workload_id){
    // first check if the entry for the workload exists
    auto it = _temp_stored_results.find(workload_id);
    return it != _temp_stored_results.end();
}


// reconfiguration methods
// ========================================

void MemoryUnit::resetTimer(){
    required_time = 0;
    start_time = 0;
    reconf_active = false;
    reconf_staged = false;
}

void MemoryUnit::initWaitingForReply(const std::deque<const ComputingWorkload *> & waiting_workloads, const std::deque<const Packet *> & waiting_packets){
    for (auto & w : waiting_workloads){
        createWaitingForReplyEntry(w->id);
    }
    for (auto & p : waiting_packets){
        for (auto & d : p->dep){
            if (d != -1){
                addReplyExpected(d, p->id);
            }
        }
    }
}

int MemoryUnit::computeNextBatchSize(const ComputingWorkload * head, const  std::deque<const ComputingWorkload *> & waiting_workloads){
    // starting from the first valid workload, we allocate the memory of the next ones in the queue
    // until the memory is full

    assert(head == waiting_workloads.front());
    const ComputingWorkload * tail = head;
    int next_batch_size = 0;
    int avail_mem_for_reconf = this->getAvailableForReconf();
    int tot_avail = this->getThresholdBytes();

    auto it = waiting_workloads.begin();
    while (it != waiting_workloads.end()){
        const ComputingWorkload * n = *it;
        assert(("Workload size exceeds the total available memory for reconfiguration", n->size <= tot_avail));
        if (next_batch_size + n->size <= avail_mem_for_reconf){
            next_batch_size += n->size;
            tail = n;
            it = std::next(it);
        }
        else{
            break;
        }
    }
    return next_batch_size;
}

bool MemoryUnit::inNextReconfigBatch(const ComputingWorkload * w, const std::deque<const ComputingWorkload *> & waiting_workloads){
    
    int next_batch_size = 0;
    int avail_mem_for_reconf = this->getAvailableForReconf();
    int tot_avail = this->getThresholdBytes();

    auto it = waiting_workloads.begin();
    while(it != waiting_workloads.end()){
        const ComputingWorkload * n = *it;
        assert(("Workload size exceeds the total available memory for reconfiguration", n->size <= tot_avail));
        if (next_batch_size + n->size <= avail_mem_for_reconf){
            next_batch_size += n->size;
            if (n->id == w->id){
                return true;
            }
            it = std::next(it);
        }
        else{
            break;
        }
    }
    return false;
}

bool MemoryUnit::checkReconfNeed(bool bypass_output_check, const std::deque<const ComputingWorkload *> & waiting_workloads){
    // a reconfiguration should take place when:
    // 1. there are still workloads in the queue that need to be loaded in memory
    // 2. there are no more workloads allocated
    // 3. all reply messages created from sending the results of the last workload have been received
    //    (that means that all the ouput placeholders have been removed)
    int allocated_workloads = getNumCurAllocatedWorkloads();
    int allocated_outputs = getNumCurAllocatedOutputs();
    // when all of the above conditions are met, we can toggle the reconfiguration flag
    if ((bypass_output_check ? true : (allocated_outputs < 1))  &&
        allocated_workloads < 1 &&
        waiting_workloads.size() > 0){    
            return true;
        };
    return false;
}

void MemoryUnit::stageReconfiguration(int current_time, bool bypass_output_check, const std::deque<const ComputingWorkload *> & waiting_workloads){
    // the method is to be called in a few instances, e.g. right after the output deallocation
    if (checkReconfNeed(bypass_output_check, waiting_workloads) && !reconf_active){
        assert(("Checking for reconfiguration need when performing reconfiguration", !reconf_active));
        auto head = waiting_workloads.begin();
        int next_batch_size = computeNextBatchSize(*head, waiting_workloads);

        assert(("The next batch size is 0", next_batch_size > 0));

        // set the timer for the reconfiguration
        int time_needed = _reg->getReconfCycles(next_batch_size);
        reconf_active = true;
        assert(required_time == 0);
        required_time += time_needed;
        start_time = current_time;
    }
}

void MemoryUnit::reconfigure(const ComputingWorkload *& head, const std::deque<const ComputingWorkload *> & waiting_workloads, std::ostream & os){
    // starting from the first valid workload, we allocate the memory of the next ones in the queue
    // until the memory is full
    const ComputingWorkload * w = head;
    if (w == nullptr){
        assert(waiting_workloads.size() == 0);
    }
    else{
        assert(w->id == waiting_workloads.front()->id);
    }
    
    int next_batch_size = 0;
    int avail_mem_for_reconf = getAvailableForReconf();
    int tot_avail = getThresholdBytes();

    auto it = waiting_workloads.begin();
    while(it != waiting_workloads.end()){
        w = *it;
        assert(("Workload size exceeds the total available memory for reconfiguration", w->size <= tot_avail));
        if (next_batch_size + w->size <= avail_mem_for_reconf){
             
            // when a workload is allocated, we must delete the results of the dependencies that have been stored
            // temporarily in the memory

            //1.  if the workload has a key in the _temp_stored_results, we deallocate the outputs
            if (checkTempStoredResults(w->id)){
                for (auto & ow : _temp_stored_results[w->id]){
                    deallocateOutput(ow); // free space
                    os << " DEALLOCATING OUTPUT FOR WORKLOAD " << ow->id << " to host WORKLOAD " <<  w->id << std::endl;
                }
                removeTempStoredResults(w->id); // remove the entry from the register
            }

            //2. we must allocate the space for the workload that is getting loaded on memory
            allocate(w); // allocate space
            os << " ALLOCATING WORKLOAD " << w->id << std::endl;

            //3. add the size of the workload to the next batch size
            next_batch_size += w->size;
            it = std::next(it);
        }
        else{
            assert(("The next batch size is 0", next_batch_size > 0));
            break;
        }
    }

    // if we have reached the end of the queue, we set no_more_to_reconfigure to true
    if (it == waiting_workloads.end()){
        no_more_to_reconfigure = true;
    }

    // update the head of the queue
    head = w;

}

// handy functions
// ========================================

int MemoryUnit::in_overlap_in(const ComputingWorkload * w1,const ComputingWorkload * w2){
    int overlap = 0;
    // check for overlapping in the input space
    for (int i = 0; i < w1->input_range.lower_sp.size(); ++i){
        if (w2->input_range.lower_sp[i] <= w1->input_range.upper_sp[i] && w2->input_range.upper_sp[i] >= w1->input_range.lower_sp[i]){
            if (overlap == 0){
            int contribution = std::min(w2->input_range.upper_sp[i], w1->input_range.upper_sp[i]) - std::max(w2->input_range.lower_sp[i], w1->input_range.lower_sp[i]);
            overlap += contribution;
            } else {
            int contribution = std::min(w2->input_range.upper_sp[i], w1->input_range.upper_sp[i]) - std::max(w2->input_range.lower_sp[i], w1->input_range.lower_sp[i]);
            overlap *= contribution;
            }
        }
    }
    // multiply by the number of channels
    if (w1->input_range.channels.size() > 0 && w2->input_range.channels.size() > 0){
    overlap *= (w1->input_range.channels[1]- w1->input_range.channels[0]);
    }
    return overlap;
};

int MemoryUnit::out_overlap_in(const ComputingWorkload * w1,const ComputingWorkload * w2){
    {
        int overlap = 0;
        for (int i = 0; i < w1->output_range.lower_sp.size(); ++i){
            if (w2->input_range.lower_sp[i] <= w1->output_range.upper_sp[i] && w2->input_range.upper_sp[i] >= w1->output_range.lower_sp[i]){
                if (overlap == 0){
                int contribution = std::min(w2->input_range.upper_sp[i], w1->output_range.upper_sp[i]) - std::max(w2->input_range.lower_sp[i], w1->output_range.lower_sp[i]);
                overlap += contribution;
                } else {
                int contribution = std::min(w2->input_range.upper_sp[i], w1->output_range.upper_sp[i]) - std::max(w2->input_range.lower_sp[i], w1->output_range.lower_sp[i]);
                overlap *= contribution;
                }
            }
        }
        // multiply by the number of channels
        if (w1->output_range.channels.size() > 0 && w2->input_range.channels.size() > 0){
        overlap *= std::min(w1->output_range.channels[1], w2->input_range.channels[1]) - std::max(w1->output_range.channels[0], w2->input_range.channels[0]);
        }
        return overlap;
        
    };
};


// -*- MEMORY SET METHODS -*- //

MemorySet::MemorySet(int nodes, int size, int threshold,  MemoryRateRegister* reg, EventLogger* logger) : _nodes(nodes), _size(size), _logger(logger){
    _pointed_workload_in_queue.resize(_nodes, nullptr);
    for (int i = 0; i < _nodes; ++i){
        _memory_units.push_back(MemoryUnit(size, threshold,reg));
    }
}

MemorySet::MemorySet(int nodes, int size, float threshold,  MemoryRateRegister* reg, EventLogger* logger) : _nodes(nodes), _size(size), _logger(logger){
    _pointed_workload_in_queue.resize(_nodes, nullptr);
    for (int i = 0; i < _nodes; ++i){
        _memory_units.push_back(MemoryUnit(size, threshold, reg));
    }
}

void MemorySet::init(std::vector<std::deque<const ComputingWorkload * >> & waiting_workloads){
    for (int i = 0; i < _nodes; ++i){
        _pointed_workload_in_queue[i] = waiting_workloads[i].size() > 0 ? waiting_workloads[i][0] : nullptr;
    }
}
