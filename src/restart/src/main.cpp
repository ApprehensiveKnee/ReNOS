// $Id$

/*
 Copyright (c) 2007-2015, Trustees of The Leland Stanford Junior University
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 Redistributions of source code must retain the above copyright notice, this 
 list of conditions and the following disclaimer.
 Redistributions in binary form must reproduce the above copyright notice, this
 list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*main.cpp
 *
 *The starting point of the network simulator
 *-Include all network header files
 *-initilize the network
 *-initialize the traffic manager and set it to run
 *
 *
 */
# include "simulation_fun.hpp"

/////////////////////////////////////////////////////////////////////////////

int main( int argc, char **argv )
{

  SimulationContext context;

  context.gDumpFile = &cout;
  //context.gDumpFile = &context.nullStream;

  BookSimConfig config;

  if ( !ParseArgs( &config, context, argc, argv ) ) {
    cerr << "Usage: " << argv[0] << " configfile... [param=value...]" << endl;
    return 0;
 } 
  config.PreprocessPackets(context.gDumpFile);

  
  /*initialize routing, traffic, injection functions
   */
  tRoutingParameters p=initializeRoutingMap(config);

  context.gPrintActivity = (config.getIntField("print_activity") > 0);
  context.gTrace = (config.getIntField("viewer_trace") > 0);
  
  string watch_out_file = config.getStrField( "watch_out" );
  if(watch_out_file == "") {
    context.gWatchOut = NULL;
  } else if(watch_out_file == "-") {
    context.gWatchOut = &cout;
  } else {
    context.gWatchOut = new ofstream(watch_out_file.c_str());
  }

  /*configure and run the simulator
   */
  int result = Simulate(config, context, p);
  return result >0 ? 0 : -1;
}
