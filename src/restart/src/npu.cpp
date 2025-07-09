/*
* ==================================================
* File: npu_cpp
* Project: src
* File Created: Wednesday, 19th March 2025
* Author: Edoardo Cabiati (edoardo.cabiati@mail.polimi.it)
* Under the supervision of: Politecnico di Milano
* ==================================================
*/


#include "npu.hpp"

void NPURateRegister::registerWorkloadCycles(WorkloadType type, float cycles){
    _workload_cycles[type] = cycles;// [cycles/FLOP]
}

int NPURateRegister::getWorkloadCycles(int size_FLOPs, WorkloadType type) const{
    return size_FLOPs * _workload_cycles.at(type);
}

void NPU::setTimer(int size, int start_time, WorkloadType type){
    required_time = _reg->getWorkloadCycles(size, type);
    required_time = required_time > 1 ? required_time : 1;
    this->start_time = start_time;
    busy = true;
}

void NPU::resetTimer(){
    required_time = 0;
    start_time = 0;
    busy = false;
}

