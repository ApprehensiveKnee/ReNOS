/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: outset.cpp
//  Description: Source file for the definition of the output set class, used for the VC implementation:
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

#include "outset.hpp"
#include <cassert>

void OutSet::clear() {
    _outputs.clear(); // clear the set
}

void OutSet::add(int output_port, int vc, int pri) {
    addRange(output_port, vc, vc, pri);
} // add single VC mapped to output port

void OutSet::addRange(int output_port, int vc_start, int vc_end, int pri) {
    sSetElement se;

    se.vc_start = vc_start;
    se.vc_end = vc_end;
    se.pri = pri;
    se.output_port = output_port;
    auto res = _outputs.insert(se);
} // add range of VCs mapped to output port

bool OutSet::empty() const {
    return _outputs.empty();
} // check if the set is empty

bool OutSet::outputEmpty(int output_port) const {
    for ( const auto & se : _outputs) {
        if (se.output_port == output_port) {
            return false;
        }
    }
    return true;
} // check if the output port is empty

int OutSet::numOutputVCs(int output_port) const {
    int count = 0;
    for (const auto & se : _outputs) {
        if (se.output_port == output_port) {
            count += se.vc_end - se.vc_start + 1;
        }
    }
    return count;
} // return the number of VCs mapped to the output port

const std::set<OutSet::sSetElement> & OutSet::getOutSet() const {
    return _outputs;
} // return the set

int OutSet::getVC(int output_port, int vc_index, int *pri ) const {
    int set_range = 0;
    int remainer = vc_index;
    int vc = -1;

    if (pri) {
        *pri = -1;
    }

    for (const auto & se : _outputs) {
        if (se.output_port == output_port) {
            set_range = se.vc_end - se.vc_start + 1;
            if (remainer < set_range) {
                vc = se.vc_start + remainer;
                if (pri) {
                    *pri = se.pri;
                }
                break;
            } else {
                remainer -= set_range;
            }
        }
    }

    return vc;

}

bool OutSet::getPortVC(int *output_port, int *out_vc) const {
    bool sigle_output = false;
    int used_output = 0;

    std::set<sSetElement>::iterator it = _outputs.begin();
    if (it != _outputs.end()) {
        used_output = it->output_port;
        // output port of the first element
    }
    while(it!=_outputs.end()){
        if (it->vc_start == it->vc_end) {
            sigle_output = true;
            *output_port = it->output_port;
            *out_vc = it->vc_start;
        }
        else {
            //there is more than one VC for output port
            break;
        }
        if (it->output_port != used_output) {
            // there is more than one output port
            sigle_output = false;
            break;
        }
        it++;
    }
    return sigle_output;
}


