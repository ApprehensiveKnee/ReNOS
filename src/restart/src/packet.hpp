
/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: packet.hpp
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

#ifndef PACKET_HPP
#define PACKET_HPP

#include <iostream>
#include <stack>
#include <set>
#include "base.hpp"
#include "params.hpp"
#include "outset.hpp"


// Gerarchies for packet automatic generation in Custom Workload Mode:
// 1. WRITE_REQ, WRITE packets generate automatically WRITE_ACKs as instant replies
// 2. READ_REQ, READ packets generate automatically READ_ACKs as instant replies
// 3. WRITE_REQ will generate READ_REQ from the dst node to the src node as soon as the dst node is free ( no dependencies)
// 4. READ_REQ will generate WRITE packets from the dst node to the src node as soon as the dst node is free

enum commType {
    READ_REQ = 1,
    READ_ACK = 3, // read reply type is substituted with read ack
    READ = 5,
    WRITE_REQ = 2,
    WRITE_ACK = 4, // write reply type is substituted with write ack
    WRITE  = 6,
    ANY = 0
};

const int NUM_FLIT_TYPES = 7;

class Flit;

class FlitPool {
    public:
        std::stack<Flit *> all;
        std::stack<Flit *> free;
        FlitPool() {}
        void freeAllFlits();
};

class Flit {
    private:
        Flit(); // private constructor
        
    public:
        
        // public construcotr for testing purpuses
        Flit(const int & id, 
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
            const short int & priority);
        ~Flit();

        int id;
        int size;
        int vc; // virtual channel
        int cl; // class of the flit
        

        //  -- packet common fields --
        int pid; // packet id
        int rpid; // request packet id
        int data_ptime_expected;
        int data_size;
        int src;
        int dst;
        commType type;
        // -- packet common fields --


        int ctime; // creation time
        int itime; // injection time
        int atime; // arrival time

        int hops; // number of hops since being sent
        bool tail; 
        bool head;
        short int priority;


        bool record;
        bool watch;
        int  subnetwork;

        // intermediate destination (if any)
        mutable int intm;

        // phase in multi-phase algorithms
        mutable int ph;

        // Fields for arbitrary data
        void* data ;
        int data_dep;

        // Lookahead route info
        OutSet la_route_set;

        void reset();

        static Flit * newFlit(FlitPool & pool);
        void freeFlit(FlitPool & pool);
        
        
};

std::ostream & operator<<(std::ostream & os, const Flit& flit);

class Credit;

class CreditPool {
    public:
        std::stack<Credit *> all;
        std::stack<Credit *> free;

        CreditPool() {}
        void freeAllCredits();
        int OutStanding();
};


class Credit {
        
    public:

        Credit() { this->reset(); }
        ~Credit() {};

        std::set<int> vc;
        int id;
        bool head,
             tail;
        
        void reset();
        static Credit * newCredit(CreditPool & pool);
        void freeCredit(CreditPool & pool);
};



#endif // PACKET_HPP