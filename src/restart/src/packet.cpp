
/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: packet.cpp
//  Description: Definitions for the packet class
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
//  Date:  20/09/2024
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "packet.hpp"
#include <cassert>
#define assertm(exp, msg) assert(((void)msg, exp))

///////////////////// FLIT ///////////////////////

std::ostream& operator<<(std::ostream& os, const Flit& flit) {
    os << "Flit ID: " << flit.id << " (" << &flit << ")"
       << "Size: " << flit.size << " "
       << "Packet: " << flit.pid << " "
       << "Type: " << flit.type << " "
       << "Head: " << flit.head << " Tail: " << flit.tail << std::endl;
    os << "Source" << flit.src << " Destination: " << flit.dst << std::endl;
    os << "Creation time: " << flit.ctime << ", Injection time: " << flit.itime << ", Arrival time: " << flit.atime << std::endl;
    os << "VC: " << flit.vc << " Priority: " << flit.priority << std::endl;

    return os;
};

Flit::Flit() {
    reset();
}

Flit::Flit(const int & id, 
            const commType & type,
            const int & size,
            const int & vc,
            const int & class_,
            const int & pid,
            const int & src,
            const int & dst,
            const int & ctime,
            const int & itime,
            const int & atime,
            const int & hops,
            const bool & isTail,
            const bool & isHead,
            const short int & priority)
            : id(id), type(type), size(size), vc(vc), cl(class_), pid(pid), src(src), dst(dst), ctime(ctime), itime(itime), atime(atime), hops(hops), tail(isTail), head(isHead), priority(priority) {
            };

Flit::~Flit() {};

void Flit::reset() 
{  
    //    id, size, vc, class_, packet, ctime, itime, atime, hops, isTail, isHead, priority
    id = -1;
    size = 0;
    vc = -1;
    cl = -1;
    pid = -1;
    src = -1;
    dst = -1;
    type = ANY;
    ctime = -1;
    itime = -1;
    atime = -1;
    hops = 0;
    tail = false;
    head = false;
    priority = 0;
    watch     = false ;
    record    = false ;
    intm = 0;
    intm =-1;
    ph = -1;
    data = 0;
}  

Flit * Flit::newFlit(FlitPool & pool) {
    Flit * f;
    if (pool.free.empty()) {
        f = new Flit();
        pool.all.push(f);
    } else {
        f = pool.free.top();
        f->reset();
        pool.free.pop();
    }
    return f;
}

void Flit::freeFlit(FlitPool & pool) {
  pool.free.push(this);
}

void FlitPool::freeAllFlits() {
  while(!this->all.empty()) {
    delete this->all.top();
    this->all.pop();
  }
}

///////////////////// CREDIT ///////////////////////

void Credit::reset(){
    vc.clear();
    id = -1;
    head = false;
    tail = false;
}

Credit * Credit::newCredit(CreditPool & pool) {
    Credit * c;
    if (pool.free.empty()) {
        c = new Credit();
        pool.all.push(c);
    } else {
        c = pool.free.top();
        c->reset();
        pool.free.pop();
    }
    return c;
}

void Credit::freeCredit(CreditPool & pool) {
    pool.free.push(this);
}

void CreditPool::freeAllCredits() {
    while(!this->all.empty()) {
        delete this->all.top();
        this->all.pop();
    }
}

int CreditPool::OutStanding(){
  return this->all.size()-this->free.size();
}



