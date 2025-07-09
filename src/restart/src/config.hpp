/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: config.hpp
//  Description: Header for the declaration of the Config class, to specify the details of the simulation
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
//  Date:  03/10/2024
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////


#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include "globals.hpp"


#include <nlohmann/json.hpp>
#include "packet.hpp"
using json = nlohmann::json;
// the configutation file parsed needs to be in JSON format
// This approach is essentially different from the one used in booksim

// two main field will be required:
// 1. the network definition, with all of its architectural details
// 2. the list of packets to be sent over the network during the simulation



// ======================================================================
//                         BASIC PACKET DEFINITION
// ======================================================================
struct Packet {
    int id;
    int src;
    int dst;
    int size;
    std::vector<int> dep; // list of dependencies (intended as list of dependencies' ids)
    int cl;
    int type; // type of the packet (0: any, 1: read request, 2: write request, 3: read ack, 4: write ack, 5: read, 6: write)
    int pt_required; // estimation for the required time to process the data, assuming all the operations/processing phases have already
    // been divided between PEs
    int priority; // priority of the packet
    int data_dep = -1; // data dependency (USED TO PRESERVE DATA DEPENDENCIES while passing packets)
};
// ======================================================================


// Computing Workload is used to simulate the time required for a PE to
// perform operations of some sort. During this time, the node of the corresponding 
// PE will be able to forward packets to next nodes, but not to pass them direcly to the
// PE, as it will be busy in processing the data. Once the processing is completed, the PE
// will update a register of performed operations for the node, and a WRITE_REQ will be passed to the node of
// the next PE that will need to further process the data
struct ComputingWorkload {
    
    struct Bounds {
        std::vector<int> lower_sp; // lower bounds for the spatial dimensions
        std::vector<int> upper_sp; // upper bounds for the spatial dimensions
        std::vector<int> channels; // number of channels
    };


    int id; // computing workload id
    int node; // node id
    int layer; // layer id
    int size; // the total size of parameters needed for the computation (input/output/weights)
    int wsize; // the size of the weights (needed to compute the reconfigurations)
    Bounds input_range; // the range of the input space
    Bounds output_range; // the range of the output space
    std::vector<int> dep; // list of dependencies
    int ct_required; // computing time required
    int type; // type to identify the workload (always converted to -1)
};


class Configuration {

    private:
        std::FILE * _config_file;
        std::string _config_string;

    protected:
        std::map<std::string, std::string> _arch_str;
        std::map<std::string, int> _arch_int;
        std::map<std::string, double> _arch_float;
        std::map<std::string, std::vector<int>> _arch_int_array;
        std::map<std::string, std::vector<double>> _arch_float_array;
        std::map<std::string, std::vector<std::string>> _arch_str_array;
        std::vector<Packet> _packets;
        std::vector<ComputingWorkload> _workloads;
        // packets will be a list of elements, each one characterized by:
        // 1. the source node
        // 2. the destination node
        // 3. the packet size
        // 4. id
        // 5. id of the closest previus data dependency

    public:
        Configuration();

        void addStrField(std::string const & field, std::string const & value);
        void addIntField(std::string const & field, int value);
        void addFloatField(std::string const & field, double value);
        void addIntArray(std::string const & field, std::vector<int> const & value);
        void addFloatArray(std::string const & field, std::vector<double> const & value);
        void addStrArray(std::string const & field, std::vector<std::string> const & value);
        void addPacket(int src, int dst,int size, int id, const std::vector<int> & dep, const std::string & type, int cl, int pt_required);
        void addComputingWorkload(int node, int id, int layer, int size, int wsize, const ComputingWorkload::Bounds input_range, const ComputingWorkload::Bounds output_range, const std::vector<int> & dep, int ct_required, const std::string & type);

        void assignArch(std::string const &field, std::string const &value);
        void assignArch(std::string const &field, int value); 
        void assignArch(std::string const &field, double value);
        
        std::string getStrField(std::string const &field) const;
        int getIntField(std::string const &field) const ;
        double getFloatField(std::string const &field) const;
        std::vector<int> getIntArray(std::string const &field) const;
        std::vector<double> getFloatArray(std::string const &field) const;
        std::vector<std::string> getStrArray(std::string const &field) const;
        Packet getPacket(int index) const;
        ComputingWorkload getComputingWorkload(int index) const;


        void WriteFile(std::string const & filename);
        void WriteMatlabFile(std::ostream * o) const;

        /* =====  A function to preprocess the packets  based on the flit_size variables*/
        void PreprocessPackets(std::ostream * o);


        // set of functions to return tokenized vector of values out of the string valued field
        // std::vector<std::string> getStrArray(std::string const &field) const;
        // std::vector<int> getIntArray(std::string const &field) const;
        // std::vector<double> getFloatArray(std::string const &field) const;

        void parseJSONFile(const std::string  &filename);
        void parseString(std::string const & str);
        void parseError(const std::string &msg, unsigned int line = 0) const;

        inline const std::map<std::string, std::string> & getStrMap() const { return _arch_str; }
        inline const std::map<std::string, int> & getIntMap() const { return _arch_int; }
        inline const std::map<std::string, double> & getFloatMap() const { return _arch_float; }
        inline const std::map<std::string, std::vector<int>> & getIntArrayMap() const { return _arch_int_array; }
        inline const std::map<std::string, std::vector<double>> & getFloatArrayMap() const { return _arch_float_array; }
        inline const std::map<std::string, std::vector<std::string>> & getStrArrayMap() const { return _arch_str_array; }
        inline const std::vector<Packet> & getPackets() const { return _packets; }
        inline const std::vector<ComputingWorkload> & getComputingWorkloads() const { return _workloads; }
};

bool ParseArgs(Configuration * cf, const SimulationContext& context, int argc, char **argv);

std::vector<std::string> tokenize_str(std::string const & data);
std::vector<int> tokenize_int(std::string const & data);
std::vector<double> tokenize_float(std::string const & data);
using pStringInt = std::pair<std::string, int>;
using pStringFloat = std::pair<std::string, double>;
using pStringString = std::pair<std::string, std::string>;
using pStringIntArray = std::pair<std::string, std::vector<int>>;
using pStringFloatArray = std::pair<std::string, std::vector<double>>;
using pStringStringArray = std::pair<std::string, std::vector<std::string>>;

void recursiveScan(const json & j,const std::string & key, std::vector<pStringInt> & intFields, std::vector<pStringFloat> & floatFields, std::vector<pStringString> & strFields, std::vector<pStringIntArray> & intArrayFields, std::vector<pStringFloatArray> & floatArrayFields, std::vector<pStringStringArray> & strArrayFields);

template<typename T>
bool isVector(const T& var) {
    return std::is_same<decltype(var), std::vector<typename T::value_type>>::value;
}

template<class T>
std::ostream & operator<<(std::ostream & os, const std::vector<T> & v) {
  for(size_t i = 0; i < v.size() - 1; ++i) {
    os << v[i] << ",";
  }
  os << v[v.size()-1];
  return os;
}


#endif // CONFIG_HPP