/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: pipeline.hpp
//  Description: Header for the declaration of the PipelineFIFO class
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
//  Date:  01/10/2024
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////


#ifndef PIPEFIFO_HPP
#define PIPEFIFO_HPP


#include <vector>
#include "base.hpp"


template<class T> class PipelineFIFO : public Module {

    private:
        int _lanes;
        int _depth;

        int _pipe_len;
        int _pipe_ptr;

        std::vector<std::vector<T*>> _data;

    public:
        PipelineFIFO(Module * parent, const std::string & name, const int & lanes, const int & depth);
        ~PipelineFIFO(){};

        void write(T * data, const int & lane = 0);
        void writeAll ( T * data);

        T* read(int lane = 0);

        void advance();
};

template<class T> 
PipelineFIFO<T>::PipelineFIFO(Module * parent, const std::string & name, const int & lanes, const int & depth) : Module(parent, name), _lanes(lanes), _depth(depth), _pipe_len(depth + 1), _pipe_ptr(0) {
    _data.resize(_lanes);
    for (int i = 0; i < _lanes; i++) {
        _data[i].resize(_pipe_len,0);
    }
}

template<class T> 
void PipelineFIFO<T>::write(T * data, const int & lane) {
    _data[lane][_pipe_ptr] = data;
}

template<class T> 
void PipelineFIFO<T>::writeAll(T * data) {
    for (int i = 0; i < _lanes; i++) {
        _data[i][_pipe_ptr] = data;
    }
}

template<class T> 
T* PipelineFIFO<T>::read(int lane) {
    return _data[lane][_pipe_ptr];
}

template<class T> 
void PipelineFIFO<T>::advance() {
    _pipe_ptr = (_pipe_ptr + 1) % _pipe_len;
}



#endif // PIPEFIFO_HPP