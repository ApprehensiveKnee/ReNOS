/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: base.hpp
//  Description: Header for the declaration of the base class
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

#ifndef BASE_HPP
#define BASE_HPP

#include <iostream>
#include <string>
#include <vector>

class TrafficManager;

class Clock {
    public:
        Clock() {reset();};
        ~Clock() {};

        inline void tick() { _time++; }
        inline int  time() const { return _time; }
        inline void reset() { _time = 0; }

    private: //accessed by the friend class TrafficManager
        int _time;
};


class Module {

    private:
        std::string _name;
        std::string _fullname;

        std::vector<Module *> _submodules;

    protected:
        void addSubmodule(Module * submodule);

    public:
        Module( Module * parent, const std::string & name);
        virtual ~Module(){};

        inline const std::string & getName() const { return _name; }
        inline const std::string & getFullName() const { return _fullname; }

        void printHierarchy(int lev, std::ostream & os) const;

        void error(const std::string & msg, const short int & code = 1, std::ostream & os = std::cerr) const;
        void printDebug(const std::string & msg, std::ostream & os) const;
        //virtual float power() const = 0;

        virtual void display( std::ostream & os) const;
    
};

class TimedModule : public Module {
    public:
        TimedModule(Module * parent, const std::string & name) : Module(parent, name) {};
        virtual ~TimedModule() {};

        virtual void readInputs() = 0;
        virtual void evaluate() = 0;
        virtual void writeOutputs() = 0;
};

#endif // BASE_HPP