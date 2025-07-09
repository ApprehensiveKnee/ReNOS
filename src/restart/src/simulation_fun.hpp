///////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: simulation_fun.hpp
//  Description: Define the higher level functions to run the simulation
//  Created by:  Edoardo Cabiati
//  Date:  24/01/2025
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <sys/time.h>

#include <string>
#include <cstdlib>
#include <iostream>
#include <fstream>

// bindings for pybind11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include "globals.hpp"
#include "routefunc.hpp"
#include "traffic.hpp"
#include "booksim_config.hpp"
#include "trafficmanager.hpp"
#include "random_utils.hpp"
#include "network.hpp"
#include "injection.hpp"
#include "power_module.hpp"


///////////////////////////////////////////////////////////////////////////////
// Definitions for the functions defined in globals.hpp
//////////////////////


int GetSimTime(const SimulationContext* context) {
  return context->trafficManager->getTime();
}

class Stats;
Stats * GetStats(const std::string & name, const SimulationContext* context) {
  Stats* test =  context->trafficManager->getStats(name);
  if(test == 0){
    cout<<"warning statistics "<<name<<" not found"<<endl;
  }
  return test;
}


 
/////////////////////////////////////////////////////////////////////////////

int Simulate( BookSimConfig const & config, SimulationContext & context, tRoutingParameters & par ) {
  
  vector<Network *> net;

  int subnets = config.getIntField("subnets");
  /*To include a new network, must register the network here
   *add an else if statement with the name of the network
   */
  net.resize(subnets);
  for (int i = 0; i < subnets; ++i) {
    ostringstream name;
    name << "network_" << i;
    net[i] = Network::New( config, context, par, name.str());
  }

  /*tcc and characterize are legacy
   *not sure how to use them 
   */

  bool logger = config.getIntField("logger");
    if(logger){
        EventLogger * event_logger = new EventLogger();
        context.setLogger(event_logger);
    }

  TrafficManager * trafficManager = TrafficManager::New( config, net, context, par );

  // Assign the traffic manager to the routing parameters class for timing purposes
  context.setTrafficManager(trafficManager);

  /*Start the simulation run
   */ 

  double total_time; /* Amount of time we've run */
  struct timeval start_time, end_time; /* Time before/after user code */
  total_time = 0.0;
  gettimeofday(&start_time, NULL);

  int result;
  if(config.getIntField("user_defined_traffic") > 0){
    result = trafficManager->RunUserDefined();

  }else{
    result = trafficManager->Run();
  }


  gettimeofday(&end_time, NULL);
  total_time = ((double)(end_time.tv_sec) + (double)(end_time.tv_usec)/1000000.0)
            - ((double)(start_time.tv_sec) + (double)(start_time.tv_usec)/1000000.0);

  *(context.gDumpFile) << "Total simulation time: " << total_time << " seconds" << endl;

//   context.PrintEvents();

  for (int i=0; i<subnets; ++i) {

    ///Power analysis
    if(config.getIntField("sim_power") > 0){
      Power_Module pnet(net[i], config);
      pnet.run();
    }

    delete net[i];
  }

  delete trafficManager;
  trafficManager = NULL;

  return result;
}

/////////////////////////////////////////////////////////////////////////////
// Create a wrapper for the simulator, in order to expose it to Python

namespace py = pybind11;

std::tuple<int, EventLogger*> SimulateWrapper(const std::string &config_file, const std::string &output_file) {

    // Generate a simulation context
    SimulationContext context;

    // define the dump file (name passed as argument)
    if (output_file == "") {
        context.gDumpFile = &context.nullStream;;
    } else if (output_file == "-") {
        context.gDumpFile = &std::cout;
    } else {
        context.gDumpFile = new std::ofstream(output_file.c_str());
    }

    // release GIL
    py::gil_scoped_release release;

    BookSimConfig config;
    char *argv[] = {const_cast<char *>("nocsim"), const_cast<char *>(config_file.c_str())};
    int argc = 2; // name of the program + name of the config file

    if (!ParseArgs(&config, context, argc, argv)) {
        throw std::runtime_error("Failed to parse configuration file");
    }
    config.PreprocessPackets(context.gDumpFile);

    tRoutingParameters p=initializeRoutingMap(config);

    context.gPrintActivity = (config.getIntField("print_activity") > 0);
    context.gTrace = (config.getIntField("viewer_trace") > 0);

    std::string watch_out_file = config.getStrField("watch_out");
    if (watch_out_file == "") {
        context.gWatchOut = &context.nullStream;
    } else if (watch_out_file == "-") {
        context.gWatchOut = &std::cout;
    } else {
        context.gWatchOut = new std::ofstream(watch_out_file.c_str());
    }


    // For now, just return the latency of the simulation
    int result = Simulate(config, context, p);

    if (result <= 0) {
        throw std::runtime_error("Simulation failed");
    }

    // Re-acquire GIL
    py::gil_scoped_acquire acquire;

    EventLogger *event_logger = context.logger;
    return std::make_tuple(result, event_logger);
    
}
