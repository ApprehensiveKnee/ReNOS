/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: buffer.hpp
//  Description: Source file for the definition of the buffer module functions
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
//  Date:  13/10/2024
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <sstream>

#include "globals.hpp"
#include "buffer.hpp"

Buffer::Buffer( const Configuration& config,
    const SimulationContext& context,
    const tRoutingParameters& par, int outputs, 
		Module *parent, const std::string& name ) :
Module( parent, name ), _occupancy(0)
{
  int num_vcs = config.getIntField( "num_vcs" );

  _size = config.getIntField("buf_size");
  if(_size < 0) {
    _size = num_vcs * config.getIntField( "vc_buf_size" );
  };

  _vc.resize(num_vcs);

  for(int i = 0; i < num_vcs; ++i) {
    std::ostringstream vc_name;
    vc_name << "vc_" << i;
    _vc[i] = new VC(config, context, par, outputs, this, vc_name.str( ) );
  }

#ifdef TRACK_BUFFERS
  int classes = config.getIntField("classes");
  _class_occupancy.resize(classes, 0);
#endif
}

Buffer::~Buffer()
{
  for(std::vector<VC*>::iterator i = _vc.begin(); i != _vc.end(); ++i) {
    delete *i;
  }
}

void Buffer::addFlit( int vc, Flit *f )
{
  if(_occupancy >= _size) {
    error("Flit buffer overflow.");
  }
  ++_occupancy;
  _vc[vc]->addFlit(f);
#ifdef TRACK_BUFFERS
  ++_class_occupancy[f->cl];
#endif
}

void Buffer::display( std::ostream & os ) const
{
  for(std::vector<VC*>::const_iterator i = _vc.begin(); i != _vc.end(); ++i) {
    (*i)->display(os);
  }
}

