/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: base.cpp
//  Description: Definitions for the base class methods
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


#include "base.hpp"


Module::Module(Module * parent, const std::string & name) : _name(name) {
    if (parent != nullptr) {
        _fullname = parent->getFullName() + "." + name;
        parent->addSubmodule(this);
    } else {
        _fullname = name;
    }
}

void Module::addSubmodule(Module * submodule) {
    _submodules.push_back(submodule);
}

void Module::printHierarchy(int lev, std::ostream & os) const {
    for (int i = 0; i < lev; i++) {
        os << "  ";
    }
    os << _name << std::endl;
    for (auto & submodule : _submodules) {
        submodule->printHierarchy(lev + 1, os);
    }
}

void Module::error(const std::string & msg, const short int & code, std::ostream & os) const {
    os << "ERROR: " << _fullname << ": " << msg << std::endl;
    exit(code);
}

void Module::printDebug(const std::string & msg, std::ostream & os) const {
    os << "DEBUG: " << _fullname << ": " << msg << std::endl;
}

void Module::display( std::ostream & os ) const 
{
  os << "Display method not implemented for " << getFullName() << std::endl;
}



