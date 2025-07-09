/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: config.cpp
//  Description: Source for the definition of the Config class, to specify the details of the simulation
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

#include <cassert>
#include "config.hpp"



Configuration::Configuration(){
    _config_file = nullptr;
}

void Configuration::addStrField(std::string const &field, std::string const &value){
    _arch_str[field] = value;
}

void Configuration::addIntField(std::string const &field, int value){
    _arch_int[field] = value;
}

void Configuration::addFloatField(std::string const &field, double value){
    _arch_float[field] = value;
}

void Configuration::addIntArray(std::string const &field, std::vector<int> const &value){
    _arch_int_array[field] = value;
}

void Configuration::addFloatArray(std::string const &field, std::vector<double> const &value){
    _arch_float_array[field] = value;
}

void Configuration::addStrArray(std::string const &field, std::vector<std::string> const &value){
    _arch_str_array[field] = value;
}

void Configuration::assignArch(std::string const &field, std::string const & value){
    std::map<std::string, std::string>::iterator match = _arch_str.find(field);
    if(match != _arch_str.end()){
        match->second = value;
    }
    else{
        parseError("Field not found in the architecture string map (assign). Field: " + field);
    }
}

void Configuration::assignArch(std::string const &field, int value){
    std::map<std::string, int>::iterator match = _arch_int.find(field);
    if(match != _arch_int.end()){
        match->second = value;
    }
    else{
        parseError("Field not found in the architecture int map(assign). Field: " + field);
    }
}

void Configuration::assignArch(std::string const &field, double value){
    std::map<std::string, double>::iterator match = _arch_float.find(field);
    if(match != _arch_float.end()){
        match->second = value;
    }
    else{
        parseError("Field not found in the architecture double map(assign). Field: " + field);
    }
}

void Configuration::addPacket(int src, int dst, int size, int id, const std::vector<int> & dep, const std::string & type, int cl, int pt_required){
    // generate a new packet
    int tp = 0;
    if (type == "READ_REQ") {
        tp = 1;
    } else if (type == "READ_ACK") {
        tp = 3;
    } else if (type == "READ") {
        tp = 5;
    } else if (type == "WRITE_REQ") {
        tp = 2;
    } else if (type == "WRITE_ACK") {
        tp = 4;
    } else if (type == "WRITE") {
        tp = 6;
    } else {
        parseError("Invalid packet type");
        exit(-1);
    }
    const Packet newPacket = Packet{ id, src, dst, size, dep, cl, tp, pt_required, 0, dep.empty() ? -1 : dep[0]};
    _packets.push_back(newPacket);
}

void Configuration::addComputingWorkload(int node, int id, int layer, int size, int wsize, const ComputingWorkload::Bounds input_range, const ComputingWorkload::Bounds output_range, const std::vector<int> & dep, int ct_required, const std::string & type){
     
    int tp = -1;
    const ComputingWorkload newWorkload = ComputingWorkload{id, node, layer, size, wsize, input_range, output_range ,dep, ct_required, tp};
    _workloads.push_back(newWorkload);
}

std::string Configuration::getStrField(std::string const &field) const {
    std::map<std::string, std::string>::const_iterator match = _arch_str.find(field);
    if(match != _arch_str.end()){
        return match->second;
    }
    else{
        parseError("Field not found in the architecture string map(get). Field: " + field);
        exit(-1);
    }
}

int Configuration::getIntField(std::string const &field) const {
    std::map<std::string, int>::const_iterator match = _arch_int.find(field);
    if(match != _arch_int.end()){
        return match->second;
    }
    else{
        parseError("Field not found in the architecture int map(get). Field: " + field);
        exit(-1);
    }
}

double Configuration::getFloatField(std::string const &field) const {
    std::map<std::string, double>::const_iterator match = _arch_float.find(field);
    if(match != _arch_float.end()){
        return match->second;
    }
    else{
        parseError("Field not found in the architecture double map(get). Field: " + field);
        exit(-1);
    }
}

std::vector<int> Configuration::getIntArray(std::string const &field) const {
    std::map<std::string, std::vector<int>>::const_iterator match = _arch_int_array.find(field);
    if(match != _arch_int_array.end()){
        return match->second; 
    }
    else{
        return std::vector<int>();
    }
}

std::vector<double> Configuration::getFloatArray(std::string const &field) const {
    std::map<std::string, std::vector<double>>::const_iterator match = _arch_float_array.find(field);
    if(match != _arch_float_array.end()){
        return match->second;
    }
    else{
        return std::vector<double>();
    }
}

std::vector<std::string> Configuration::getStrArray(std::string const &field) const {
    std::map<std::string, std::vector<std::string>>::const_iterator match = _arch_str_array.find(field);
    if(match != _arch_str_array.end()){
        return match->second;
    }
    else{
        std::map<std::string, std::string>::const_iterator match_str = _arch_str.find(field);
        if(match_str != _arch_str.end()){
            return tokenize_str(match_str->second);
        }
        else{
            return std::vector<std::string>();
        } 
    }
}

Packet Configuration::getPacket(int index) const {
    if(index < _packets.size()){
        return _packets[index];
    }
    else{
        parseError("Packet index out of range");
        exit(-1);
    }
}

ComputingWorkload Configuration::getComputingWorkload(int index) const {
    if(index < _workloads.size()){
        return _workloads[index];
    }
    else{
        parseError("Computing workload index out of range");
        exit(-1);
    }
}



/*

std::vector<std::string> Configuration::getStrArray(std::string const & field) const {

    std::string const value_str = getStrField(field);
    return tokenize_str(value_str);

}

std::vector<int> Configuration::getIntArray(std::string const & field) const {
    std::string const value_str = getStrField(field);
    return tokenize_int(value_str);
}

std::vector<double> Configuration::getFloatArray(std::string const & field) const {
    std::string const value_str = getStrField(field);
    return tokenize_float(value_str);
}

*/

void Configuration::WriteFile(std::string const & filename) {
  
  std::ostream *config_out= new std::ofstream(filename.c_str());
  
  
  for(std::map<std::string,std::string>::const_iterator i = _arch_str.begin(); 
      i!=_arch_str.end();
      i++){
    //the parser won't read empty strings
    if(i->second[0]!='\0'){
      *config_out<<i->first<<" = "<<i->second<<";"<<std::endl;
    }
  }

  for(std::map<std::string, std::vector<std::string>>::const_iterator i= _arch_str_array.begin();
      i != _arch_str_array.end();
      i++){
    if(!i->second.empty()){
        *config_out<<i->first<<" = "<<i->second<<";"<<std::endl;
    }
}
  
  for(std::map<std::string, int>::const_iterator i = _arch_int.begin(); 
      i!=_arch_int.end();
      i++){
    *config_out<<i->first<<" = "<<i->second<<";"<<std::endl;

  }

  for(std::map<std::string, std::vector<int>>::const_iterator i = _arch_int_array.begin();
      i!= _arch_int_array.end();
      i++){
    if(!i->second.empty()){
        // print to stream using the overloaded << operator
        *config_out << "%"<<i->first<<" = \'"<<i->second<<"\';"<<std::endl;
    }
}

  for(std::map<std::string, double>::const_iterator i = _arch_float.begin(); 
      i!=_arch_float.end();
      i++){
    *config_out<<i->first<<" = "<<i->second<<";"<<std::endl;

  }

  for(std::map<std::string, std::vector<double>>::const_iterator i = _arch_float_array.begin();
      i!= _arch_float_array.end();
      i++){
    if(!i->second.empty()){
        // print to stream using the overloaded << operator
        *config_out << "%"<<i->first<<" = \'"<<i->second<<"\';"<<std::endl;
    }
}
  config_out->flush();
  delete config_out;
 
}



void Configuration::WriteMatlabFile(std::ostream * config_out) const {

  
  
  for(std::map<std::string,std::string>::const_iterator i = _arch_str.begin(); 
      i!=_arch_str.end();
      i++){
    //the parser won't read blanks lolz
    if(i->second[0]!='\0'){
      *config_out<<"%"<<i->first<<" = \'"<<i->second<<"\';"<<std::endl;
    }
  }

  for(std::map<std::string, std::vector<std::string>>::const_iterator i = _arch_str_array.begin();
      i!= _arch_str_array.end();
      i++){
    if(!i->second.empty()){
        // print to stream using the overloaded << operator
        *config_out << "%"<<i->first<<" = \'"<<i->second<<"\';"<<std::endl;
    }
}
  
  for(std::map<std::string, int>::const_iterator i = _arch_int.begin(); 
      i!=_arch_int.end();
      i++){
    *config_out<<"%"<<i->first<<" = "<<i->second<<";"<<std::endl;

  }

  for(std::map<std::string, std::vector<int>>::const_iterator i = _arch_int_array.begin();
      i!= _arch_int_array.end();
      i++){
    if(!i->second.empty()){
        // print to stream using the overloaded << operator
        *config_out << "%"<<i->first<<" = \'"<<i->second<<"\';"<<std::endl;
    }
}

  for(std::map<std::string, double>::const_iterator i = _arch_float.begin(); 
      i!=_arch_float.end();
      i++){
    *config_out<<"%"<<i->first<<" = "<<i->second<<";"<<std::endl;

  }

  for(std::map<std::string, std::vector<double>>::const_iterator i = _arch_float_array.begin();
      i!= _arch_float_array.end();
      i++){
    if(!i->second.empty()){
        // print to stream using the overloaded << operator
        *config_out << "%"<<i->first<<" = \'"<<i->second<<"\';"<<std::endl;
    }
}


  config_out->flush();

}


void Configuration::PreprocessPackets(std::ostream * o){
    /**
     * Preprocesses the packet sizes based on the flit_size variable.
     *
     * This function loops over all the packets and divides the packet size by the flit_size.
     * If the packet size after rescaling is 0, it is set to 1.
     *
     * The scaling factor allows rescaling the size of a packet to the number of flits needed to send it.
     * The size of the packet defined in the .json file is in bytes, while flit_size is in bits.
     *
     * Example: If a packet has a size of 8 BYTES and the flit size is 8 BITS, the packet size is rescaled to 8 flits.
     * Thus, 8 flits of size 8 bits are needed to send 8 BYTES over a channel.
     *
     */

    int flit_size = getIntField("flit_size");
    if (flit_size <= 0){
        throw std::invalid_argument("Flit size must be greater than 0");
    }
    
    double scaling = 8./flit_size;

    for (auto & packet : _packets){

        double temp_size = packet.size * scaling;
        double temp_pt_required = packet.pt_required * scaling; 
        
        // Round up if temp_size is not an integer
        if  (std::ceil(temp_size) != temp_size){
            temp_size = std::ceil(temp_size); //round up
            temp_pt_required = std::ceil(temp_pt_required);
        }

        // Ensure at least 1 flit
        if (temp_size == 0) { //rare case
            temp_size = 1;
            temp_pt_required = packet.pt_required; //pt_required as defined in json
                }
        
        // cast to intger because size and pt_required are integers
        packet.size = static_cast<int>(temp_size);
        packet.pt_required = static_cast<int>(temp_pt_required);

        assert(packet.size > 0);
    }
}

void Configuration::parseJSONFile(const std::string & filename){
    // The function reads the JSON file and parses it to fill the configuration fields
    // The function uses the nlohmann::json library to parse the JSON file
    // The function is based on the example provided in the nlohmann::json library documentation
    
   if((_config_file = fopen(filename.c_str(), "r")) == 0){
        std::cerr << "Error: unable to open the configuration file" << std::endl;
        exit(-1);
    }

    fseek(_config_file, 0, SEEK_END);
    long fsize = ftell(_config_file);
    fseek(_config_file, 0, SEEK_SET);

    std::string json_str;
    json_str.resize(fsize);
    fread(&json_str[0], 1, fsize, _config_file);
    fclose(_config_file);

    nlohmann::json j = nlohmann::json::parse(json_str);

    if (j.find("arch") != j.end()){
        // the JSON file has two main fields: "arch" and "packets"
        // the "arch" field contains the architectural details of the network
        // the "packets" field contains the list of packets to be sent over the network during the simulation
        nlohmann::json arch = j.at("arch");
        
        // parse the "arch" field
        for (nlohmann::json::iterator it = arch.begin(); it != arch.end(); ++it){
            // call recursiveScan
            std::vector<pStringInt> intFields;
            std::vector<pStringFloat> floatFields;
            std::vector<pStringString> strFields;
            std::vector<pStringIntArray> intArrayFields;
            std::vector<pStringFloatArray> floatArrayFields;
            std::vector<pStringStringArray> strArrayFields;
            recursiveScan(it.value(), it.key(), intFields, floatFields, strFields, intArrayFields, floatArrayFields, strArrayFields);

            for(auto & p : intFields){
                addIntField(p.first, p.second);
            }

            for(auto & p : floatFields){
                addFloatField(p.first, p.second);
            }

            for(auto & p : strFields){
                addStrField(p.first, p.second);
            }

            for(auto & p : intArrayFields){
                addIntArray(p.first, p.second);
            }

            for(auto & p : floatArrayFields){
                addFloatArray(p.first, p.second);
            }

            for (auto & p : strFields){
                assignArch(p.first, p.second);
            }
            
        }


        if (j.find("workload") != j.end()){
            nlohmann::json units = j.at("workload");
            // parse the "workload" field: it comprises of both packets and computing workloads
            for (nlohmann::json::iterator it = units.begin(); it != units.end(); ++it){
                if(it->at("type") == "COMP_OP"){
                    if(it->at("node").is_number_integer() && it->at("id").is_number_integer() && it->at("dep").is_array() && it->at("ct_required").is_number_integer() && it->at("type").is_string()){
                        // check that the depencies are valid intergers
                        for (auto & dep : it->at("dep")){
                            if(!dep.is_number_integer()){
                                parseError("Invalid dependency in the workload document");
                            }
                        }
                        assert (it->at("size") >= it->at("weight_size"));

                        int layer_id = -1;
                        if (it->find("layer_id") != it->end()){
                            layer_id = it->at("layer_id");
                        }
                        
                        // if the workload also has a input_range and output_range field, fill them in:
                        // the structure of those fields it the following:
                        // input_range: { "spatial_min" : [x, y], "spatial_max" : [x, y], "ch_bounds" : [min, max] } 
                        // OSS: in geeral spatial_min and spatial_max have the same dimensions (either 2 or 1), while ch_bounds is a 2-element array only for
                        // convolutional layers, while is an empty array for fully connected layers
                        if(it->find("input_range") != it->end() && it->find("output_range") != it->end()){
                            // assume the input_range and output_range fields are correctly formatted
                            std::vector<int> spatial_min = it->at("input_range").at("spatial_min");
                            std::vector<int> spatial_max = it->at("input_range").at("spatial_max");
                            std::vector<int> ch_bounds = it->at("input_range").at("ch_bounds");
                            std::vector<int> spatial_min_out = it->at("output_range").at("spatial_min");
                            std::vector<int> spatial_max_out = it->at("output_range").at("spatial_max");
                            std::vector<int> ch_bounds_out = it->at("output_range").at("ch_bounds");
                            
                            ComputingWorkload::Bounds input_range = ComputingWorkload::Bounds{spatial_min, spatial_max, ch_bounds};
                            ComputingWorkload::Bounds output_range = ComputingWorkload::Bounds{spatial_min_out, spatial_max_out, ch_bounds_out};



                            addComputingWorkload(it->at("node"), it->at("id"),layer_id , it->at("size"), it->at("weight_size"), input_range, output_range, it->at("dep"), it->at("ct_required"), it->at("type"));
                        }
                        else 
                        {
                            std::vector<int> empty;
                            ComputingWorkload::Bounds input_range = ComputingWorkload::Bounds{empty, empty, empty};
                            ComputingWorkload::Bounds output_range = ComputingWorkload::Bounds{empty, empty, empty};
                            addComputingWorkload(it->at("node"), it->at("id"),layer_id , it->at("size"), it->at("weight_size"), input_range, output_range, it->at("dep"), it->at("ct_required"), it->at("type"));
                        }
                        
                    }
                    else{
                        parseError("Invalid workload element in the workload document");
                    }
                }else
                {
                    if(it->at("src").is_number_integer() && it->at("dst").is_number_integer() && it->at("size").is_number_integer() && it->at("id").is_number_integer() && it->at("dep").is_array() && it->at("cl").is_number_integer() && it->at("type").is_string() && it->at("pt_required").is_number_integer()){
                        // check that the depencies are valid intergers
                        for (auto & dep : it->at("dep")){
                            if(!dep.is_number_integer()){
                                parseError("Invalid dependency in the workload document");
                            }
                        }
                        addPacket(it->at("src"), it->at("dst"), it->at("size"), it->at("id"), it->at("dep"),it->at("type"), it->at("cl"), it-> at("pt_required"));
                    }
                    else{
                        parseError("Invalid packet element in the workload document");
                    }
                } 
            }
        }
        
    }
    else{
        // inside the power_spec file, directly parse the fields
        // parse the JSON file
        for (nlohmann::json::iterator it = j.begin(); it != j.end(); ++it){
            // call recursiveScan
            std::vector<pStringInt> intFields;
            std::vector<pStringFloat> floatFields;
            std::vector<pStringString> strFields;
            std::vector<pStringIntArray> intArrayFields;
            std::vector<pStringFloatArray> floatArrayFields;
            std::vector<pStringStringArray> strArrayFields;
            recursiveScan(it.value(), it.key(), intFields, floatFields, strFields, intArrayFields, floatArrayFields, strArrayFields);

            for(auto & p : intFields){
                addIntField(p.first, p.second);
            }

            for(auto & p : floatFields){
                addFloatField(p.first, p.second);
            }

            for(auto & p : strFields){
                addStrField(p.first, p.second);
            }

            for(auto & p : intArrayFields){
                addIntArray(p.first, p.second);
            }

            for(auto & p : floatArrayFields){
                addFloatArray(p.first, p.second);
            }

            for (auto & p : strFields){
                assignArch(p.first, p.second);
            }
            
        }
    }

}


void Configuration::parseString(std::string const & str)
{
  // all fields defined in the configutation file
}

    
void Configuration::parseError(const std::string &msg, unsigned int line) const {
    if (line == 0) {
        std::cerr << "Parse error: " << msg << std::endl;
    } else {
        std::cerr << "Parse error on line " << line << ": " << msg << std::endl;
    }
}

//============================================================


bool ParseArgs(Configuration * cf, const SimulationContext& context, int argc, char * * argv)
{
  bool rc = false;

  //all dashed variables are ignored by the arg parser
  *(context.gDumpFile) << "Parsing arguments" << std::endl;
  for(int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    size_t pos = arg.find('=');
    bool dash = (argv[i][0] =='-');
    if(pos == std::string::npos && !dash) {
      // parse config file
      cf->parseJSONFile( argv[i] );
      std::ifstream in(argv[i]);
      *(context.gDumpFile) << "BEGIN Configuration File: " << argv[i] << std::endl;
      while (!in.eof()) {
	char c;
	in.get(c);
	*(context.gDumpFile) << c ;
      }
      *(context.gDumpFile) << "END Configuration File: " << argv[i] << std::endl;
      rc = true;
    } /*else if(pos != std::string::npos)  {
      // override individual parameter
      *(context.gDumpFile) << "OVERRIDE Parameter: " << arg << std::endl;
      cf->ParseString(argv[i]);
    }*/
  }

  return rc;
}


std::vector<std::string> tokenize_str(std::string const & data)
{
  std::vector<std::string> values;

  // no elements, no braces --> empty list
  if(data.empty()) {
    return values;
  }

  // doesn't start with an opening brace --> treat as single element
  // note that this element can potentially contain nested lists 
  if(data[0] != '{') {
    values.push_back(data);
    return values;
  }

  size_t start = 1;
  int nested = 0;

  size_t curr = start;

  while(std::string::npos != (curr = data.find_first_of("{,}", curr))) {
    
    if(data[curr] == '{') {
      ++nested;
    } else if((data[curr] == '}') && nested) {
      --nested;
    } else if(!nested) {
      if(curr > start) {
	std::string token = data.substr(start, curr - start);
	values.push_back(token);
      }
      start = curr + 1;
    }
    ++curr;
  }
  assert(!nested);

  return values;
}

std::vector<int> tokenize_int(std::string const & data)
{
  std::vector<int> values;

  // no elements, no braces --> empty list
  if(data.empty()) {
    return values;
  }

  // doesn't start with an opening brace --> treat as single element
  // note that this element can potentially contain nested lists 
  if(data[0] != '{') {
    values.push_back(atoi(data.c_str()));
    return values;
  }

  size_t start = 1;
  int nested = 0;

  size_t curr = start;

  while(std::string::npos != (curr = data.find_first_of("{,}", curr))) {
    
    if(data[curr] == '{') {
      ++nested;
    } else if((data[curr] == '}') && nested) {
      --nested;
    } else if(!nested) {
      if(curr > start) {
	std::string token = data.substr(start, curr - start);
	values.push_back(atoi(token.c_str()));
      }
      start = curr + 1;
    }
    ++curr;
  }
  assert(!nested);

  return values;
}

std::vector<double> tokenize_float(std::string const & data)
{
  std::vector<double> values;

  // no elements, no braces --> empty list
  if(data.empty()) {
    return values;
  }

  // doesn't start with an opening brace --> treat as single element
  // note that this element can potentially contain nested lists 
  if(data[0] != '{') {
    values.push_back(atof(data.c_str()));
    return values;
  }

  size_t start = 1;
  int nested = 0;

  size_t curr = start;

  while(std::string::npos != (curr = data.find_first_of("{,}", curr))) {
    
    if(data[curr] == '{') {
      ++nested;
    } else if((data[curr] == '}') && nested) {
      --nested;
    } else if(!nested) {
      if(curr > start) {
	std::string token = data.substr(start, curr - start);
	values.push_back(atof(token.c_str()));
      }
      start = curr + 1;
    }
    ++curr;
  }
  assert(!nested);

  return values;
}


void recursiveScan(json const & j, std::string const & key, std::vector<pStringInt> & intFields, std::vector<pStringFloat> & floatFields, std::vector<pStringString> & strFields, std::vector<pStringIntArray> & intArrayFields, std::vector<pStringFloatArray> & floatArrayFields, std::vector<pStringStringArray> & strArrayFields){
    // The function scans the JSON object and returns a vector of pairs with the key-value pairs
    // If the object contains sub-objects, the function recursively scans them until the leaf values are reached:
    // the final names for the fields will be constructed by concatenating the keys of the sub-objects
    
    if(j.is_object()){
        for(json::const_iterator it = j.begin(); it != j.end(); ++it){
            std::string new_key = key + "." + it.key();
            recursiveScan(it.value(), new_key, intFields, floatFields, strFields, intArrayFields, floatArrayFields, strArrayFields);
        }
    }
    else if(j.is_number_integer()){
        intFields.push_back(std::make_pair(key, j.get<int>()));
    }
    else if(j.is_number_float()){
        floatFields.push_back(std::make_pair(key, j.get<double>()));
    }
    else if(j.is_string()){
        strFields.push_back(std::make_pair(key, j.get<std::string>()));
    }
    else if(j.is_array()){
        // verify the type contained in the array considering the first element
        if( j[0].is_number_integer()){
            std::vector<int> intArray;
            for(json::const_iterator it = j.begin(); it != j.end(); ++it){
                intArray.push_back(it.value().get<int>());
            }
            intArrayFields.push_back(std::make_pair(key, intArray));
        }
        else if(j[0].is_number_float()){
            std::vector<double> floatArray;
            for(json::const_iterator it = j.begin(); it != j.end(); ++it){
                floatArray.push_back(it.value().get<double>());
            }
            floatArrayFields.push_back(std::make_pair(key, floatArray));
        }
        else if (j[0].is_string()){
            std::vector<std::string> strArray;
            for(json::const_iterator it = j.begin(); it != j.end(); ++it){
                strArray.push_back(it.value().get<std::string>());
            }
            strArrayFields.push_back(std::make_pair(key, strArray));
        }
        else{
            std::cerr << "Error: unsupported type in the JSON object" << std::endl;
            exit(-1);
        }
        
    }
    else{
        std::cerr << "Error: unsupported type in the JSON object" << std::endl;
        exit(-1);
    }
}
        