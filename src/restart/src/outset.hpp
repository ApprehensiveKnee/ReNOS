/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: outset.hpp
//  Description: Header for the declaration of the output set class, used for the VC implementation:
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


#ifndef OUTSET_HPP
#define OUTSET_HPP

#include <iostream>
#include <set>



class OutSet {

    public:

        struct sSetElement {
            int vc_start;
            int vc_end; // vc range
            int pri; // output priority
            int output_port;
        };

        void clear();
        void add(int output_port, int vc, int pri = 0);
        void addRange(int output_port, int vc_start, int vc_end, int pri = 0);

        bool empty() const;
        bool outputEmpty(int output_port) const;
        int numOutputVCs(int output_port) const;

        const std::set<sSetElement> & getOutSet() const;

        // mutiple sets associated to a single output port
        // the function returns the VC number used in the set and the priority of the set
        int getVC(int output_port, int vc_index, int *pri = 0) const;

        //return true it there is a single output with a single VC
        bool getPortVC(int *output_port, int *out_vc) const;

    private:
        std::set<sSetElement> _outputs;
};


// overloading operator< for the sSetElement struct base on priority
inline bool operator<(const OutSet::sSetElement & se1, const OutSet::sSetElement & se2) {
  return se1.pri > se2.pri;
}



#endif // OUTSET_HPP