/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: channel.hpp
//  Description: Header for the declaration of the channel class
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
//  Date:  30/09/2024
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////


#ifndef CHANNEL_HPP
#define CHANNEL_HPP

#include <queue>
#include <cassert>
#include "params.hpp"
#include "base.hpp"
#include "packet.hpp"
#include "globals.hpp"



/////////////////////////////////////////////////////////////////////////////////////////////////////////

#define CHANNEL_CODE 3

template <class T> // May be packet or flit
class Channel : public TimedModule {
    protected:
        int _delay;
        const SimulationContext * _context;
        T * _input_end;
        T * _output_end;
        std::queue<std::pair<int, T *>>_wait_queue;

    public:
    Channel(Module * parent, const SimulationContext& context, const std::string & name) : TimedModule(parent, name), _delay(COMM_DELAY), _input_end(nullptr), _output_end(nullptr), _context(&context) {};
    ~Channel() {};

    void setLatency (int cycles);
    int getLatency() const;

    virtual void send(T * data);
    virtual T * receive();

    virtual void readInputs();
    inline virtual void evaluate() {};
    virtual void writeOutputs();

};

template <class T>
void Channel<T>::setLatency(int cycles) {
    if (cycles < 0) {
        error("Latency cannot be negative", CHANNEL_CODE);
    }
    _delay = cycles;
};


template <class T>
int Channel<T>::getLatency() const {
    return _delay;
};


template <class T>
void Channel<T>::send(T * data) {
    _input_end = data;
};

template <class T>
T * Channel<T>::receive() {
    return _output_end;
};

template <class T>
void Channel<T>::readInputs() {
    if (_input_end != nullptr) {
        _wait_queue.push(std::make_pair(GetSimTime(_context) + _delay - 1, _input_end));
        _input_end = nullptr;
    }
};

template <class T>
void Channel<T>::writeOutputs() {
    // set output_end to nullptr to prepare for next writing to output_end
    _output_end = nullptr;
    // check if the queue is empty
    if (_wait_queue.empty()) {
        return;
    }
    std::pair<int, T *> & data = _wait_queue.front();
    int const & time = data.first;
    if (GetSimTime(_context)< time) {
        return;
    }
    
    assert(GetSimTime(_context) == time);
    // flit is written after the delay to the receiving end
    _output_end = data.second;
    // assert that the packet is not null
    assert(_output_end != nullptr);
    _wait_queue.pop();

};



// Forward declaration for the Router class
class Router;


// Declaration for the FlitChannel class
class FlitChannel : public Channel<Flit> {
    
    private:
        Router const * _src_router; // pointer to the source router
        Router const * _snk_router; // pointer to the sink router
        int _src_port;
        int _snk_port;

        // Stats-purpoused members
        std::vector<int> _active;
        int _idle;
        
    public:
        FlitChannel(Module * parent, const SimulationContext& context, const std::string & name, const int & classes);

        void setSrcRouter(Router const * router, int port);
        inline Router const * getSrcRouter() const { return _src_router; }
        inline int const & getSrcPort() const { return _src_port; }

        void setSnkRouter(Router const * router, int port);
        inline Router const * getSnkRouter() const { return _snk_router; }
        inline int const & getSnkPort() const { return _snk_port; }
        inline std::vector<int> const & getActivity() const { return _active; }

        void send(Flit * flit) override;
        Flit * receive() override;


};


// Declaration for the CreditChannel class
typedef Channel<Credit> CreditChannel;



#endif // CHANNEL_HPP