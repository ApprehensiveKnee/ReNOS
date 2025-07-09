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

#include <sstream>
#include <cmath>
#include <fstream>
#include <limits>
#include <cstdlib>
#include <ctime>

#include "booksim_config.hpp"
#include "trafficmanager.hpp"
#include "batchtrafficmanager.hpp"
#include "random_utils.hpp" 
#include "vc.hpp"
#include "packet_reply_info.hpp"

TrafficManager * TrafficManager::New(Configuration const & config,
                                     vector<Network *> const & net,
                                     const SimulationContext& context,
                                     const tRoutingParameters& par)
{
    TrafficManager * result = NULL;
    string sim_type = config.getStrField("sim_type");
    if((sim_type == "latency") || (sim_type == "throughput")) {
        result = new TrafficManager(config, net, context, par);
    } else if(sim_type == "batch") {
        result = new BatchTrafficManager(config, net, context, par);
    } else {
        cerr << "Unknown simulation type: " << sim_type << endl;
    } 
    return result;
}

TrafficManager::TrafficManager( const Configuration &config, const vector<Network *> & net, const SimulationContext& context, const tRoutingParameters& par )
    : Module( 0, "traffic_manager" ), _context(&context), _net(net), _empty_network(false), _deadlock_timer(0), _reset_time(0), _drain_time(-1), _cur_id(0), _cur_pid(0)
{
    _clock.reset();
    _nodes = _net[0]->NumNodes( );
    _routers = _net[0]->NumRouters( );

    _vcs = config.getIntField("num_vcs");
    _subnets = config.getIntField("subnets");
 
    _subnet.resize(NUM_FLIT_TYPES);
    _subnet[commType::READ_REQ] = config.getIntField("read_request_subnet");
    _subnet[commType::READ_ACK] = config.getIntField("read_reply_subnet");
    _subnet[commType::READ] = config.getIntField("read_subnet");
    _subnet[commType::WRITE_ACK] = config.getIntField("write_request_subnet");
    _subnet[commType::WRITE_ACK] = config.getIntField("write_reply_subnet");
    _subnet[commType::WRITE] = config.getIntField("write_subnet");

    // ============ Create the message pools ============
    flit_pool = FlitPool();
    credit_pool = CreditPool();
    packet_reply_pool = PacketReplyPool();

    // ============ Message priorities ============ 

    string priority = config.getStrField( "priority" );

    if ( priority == "class" ) {
        _pri_type = class_based;
    } else if ( priority == "age" ) {
        _pri_type = age_based;
    } else if ( priority == "network_age" ) {
        _pri_type = network_age_based;
    } else if ( priority == "local_age" ) {
        _pri_type = local_age_based;
    } else if ( priority == "queue_length" ) {
        _pri_type = queue_length_based;
    } else if ( priority == "hop_count" ) {
        _pri_type = hop_count_based;
    } else if ( priority == "sequence" ) {
        _pri_type = sequence_based;
    } else if ( priority == "none" ) {
        _pri_type = none;
    } else {
        error( "Unkown priority value: " + priority );
    }

    // ============ Routing ============ 

    string rf = config.getStrField("routing_function") + "_" + config.getStrField("topology");
    map<string, tRoutingFunction>::const_iterator rf_iter = par.gRoutingFunctionMap.find(rf);
    if(rf_iter == par.gRoutingFunctionMap.end()) {
        error("Invalid routing function: " + rf);
    }
    _rf = rf_iter->second;
  
    _lookahead_routing = !config.getIntField("routing_delay");
    _noq = config.getIntField("noq");
    if(_noq) {
        if(!_lookahead_routing) {
            error("NOQ requires lookahead routing to be enabled.");
        }
    }

    // ============ Traffic ============ 

    // ================ Additions for User-Defined Traffic ================
    
    _classes = config.getIntField("classes");

    _user_defined_traffic = config.getIntField("user_defined_traffic");
    vector<string> injection_process;
    if (_user_defined_traffic){
        // get the packets from the configuration
        _packets = config.getPackets();
        _classes = 1;
        _traffic = vector<string>(1, "user_defined");
        injection_process = vector<string>(1, "dependent");
        
    }else{
        _packets = vector<Packet>();
        _traffic = config.getStrArray("traffic");
        injection_process = config.getStrArray("injection_process");
    }

    _landed_packets.resize(_classes);
    for (int s = 0; s <_classes; ++s){
        _landed_packets[s].resize(_nodes);
    }

    _useful_pocessing_spots.resize(_classes);
    for (int s = 0; s <_classes; ++s){
        _useful_pocessing_spots[s].resize(_nodes, 0);
    }

    _to_process_packets.resize(_classes);
    for (int s = 0; s <_classes; ++s){
        _to_process_packets[s].resize(_nodes);
    }

    _processed_packets.resize(_classes);
    for (int s = 0; s <_classes; ++s){
        _processed_packets[s].resize(_nodes);
    }

    _use_read_write = config.getIntArray("use_read_write");
    if(_use_read_write.empty()) {
        _use_read_write.push_back(config.getIntField("use_read_write"));
    }
    _use_read_write.resize(_classes, _use_read_write.back());

    _write_fraction = config.getFloatArray("write_fraction");
    if(_write_fraction.empty()) {
        _write_fraction.push_back(config.getFloatField("write_fraction"));
    }
    _write_fraction.resize(_classes, _write_fraction.back());

    _read_request_size = config.getIntArray("read_request_size");
    if(_read_request_size.empty()) {
        _read_request_size.push_back(config.getIntField("read_request_size"));
    }
    _read_request_size.resize(_classes, _read_request_size.back());

    _read_reply_size = config.getIntArray("read_reply_size");
    if(_read_reply_size.empty()) {
        _read_reply_size.push_back(config.getIntField("read_reply_size"));
    }
    _read_reply_size.resize(_classes, _read_reply_size.back());

    _write_request_size = config.getIntArray("write_request_size");
    if(_write_request_size.empty()) {
        _write_request_size.push_back(config.getIntField("write_request_size"));
    }
    _write_request_size.resize(_classes, _write_request_size.back());

    _write_reply_size = config.getIntArray("write_reply_size");
    if(_write_reply_size.empty()) {
        _write_reply_size.push_back(config.getIntField("write_reply_size"));
    }
    _write_reply_size.resize(_classes, _write_reply_size.back());

    _read_request_time = config.getIntArray("read_request_time");
    if(_read_request_time.empty()) {
        _read_request_time.push_back(config.getIntField("read_request_time"));
    }
    _read_request_time.resize(_classes, _read_request_time.back());

    _write_request_time = config.getIntArray("write_request_time");
    if(_write_request_time.empty()) {
        _write_request_time.push_back(config.getIntField("write_request_time"));
    }
    _write_request_time.resize(_classes, _write_request_time.back());

    _read_reply_time = config.getIntArray("read_reply_time");
    if(_read_reply_time.empty()) {
        _read_reply_time.push_back(config.getIntField("read_reply_time"));
    }
    _read_reply_time.resize(_classes, _read_reply_time.back());

    _write_reply_time = config.getIntArray("write_reply_time");
    if(_write_reply_time.empty()) {
        _write_reply_time.push_back(config.getIntField("write_reply_time"));
    }
    _write_reply_time.resize(_classes, _write_reply_time.back());




    string packet_size_str = config.getStrField("packet_size");
    if(packet_size_str.empty()) {
        _packet_size.push_back(vector<int>(1, config.getIntField("packet_size")));
    } else {
        vector<string> packet_size_strings = tokenize_str(packet_size_str);
        for(size_t i = 0; i < packet_size_strings.size(); ++i) {
            _packet_size.push_back(tokenize_int(packet_size_strings[i]));
        }
    }
    _packet_size.resize(_classes, _packet_size.back());

    string packet_size_rate_str = config.getStrField("packet_size_rate");
    if(packet_size_rate_str.empty()) {
        int rate = config.getIntField("packet_size_rate");
        assert(rate >= 0);
        for(int c = 0; c < _classes; ++c) {
            int size = _packet_size[c].size();
            _packet_size_rate.push_back(vector<int>(size, rate));
            _packet_size_max_val.push_back(size * rate - 1);
        }
    } else {
        vector<string> packet_size_rate_strings = tokenize_str(packet_size_rate_str);
        packet_size_rate_strings.resize(_classes, packet_size_rate_strings.back());
        for(int c = 0; c < _classes; ++c) {
            vector<int> rates = tokenize_int(packet_size_rate_strings[c]);
            rates.resize(_packet_size[c].size(), rates.back());
            _packet_size_rate.push_back(rates);
            int size = rates.size();
            int max_val = -1;
            for(int i = 0; i < size; ++i) {
                int rate = rates[i];
                assert(rate >= 0);
                max_val += rate;
            }
            _packet_size_max_val.push_back(max_val);
        }
    }
  
    for(int c = 0; c < _classes; ++c) {
        if(_use_read_write[c]) {
            _packet_size[c] = 
                vector<int>(1, (_read_request_size[c] + _read_reply_size[c] +
                                _write_request_size[c] + _write_reply_size[c]) / 2);
            _packet_size_rate[c] = vector<int>(1, 1);
            _packet_size_max_val[c] = 0;
        }
    }

    _load = config.getFloatArray("injection_rate"); 
    if(_load.empty()) {
        _load.push_back(config.getFloatField("injection_rate"));
    }
    _load.resize(_classes, _load.back());

    if(config.getIntField("injection_rate_uses_flits")) {
        for(int c = 0; c < _classes; ++c)
            _load[c] /= _GetAveragePacketSize(c);
    }

    _traffic.resize(_classes, _traffic.back());

    _traffic_pattern.resize(_classes);
    _params.resize(_classes);

    _class_priority = config.getIntArray("class_priority"); 
    if(_class_priority.empty()) {
        _class_priority.push_back(config.getIntField("class_priority"));
    }
    _class_priority.resize(_classes, _class_priority.back());

    injection_process.resize(_classes, injection_process.back());

    _injection_process.resize(_classes);

    
    for(int c = 0; c < _classes; ++c) {

        _traffic_pattern[c] = TrafficPattern::New(_traffic[c], _nodes, &config);

        if (_user_defined_traffic){
            // ============  Reconfiguration ============
            int reconf = config.getIntField("reconfiguration");
            int pe_mem_size = config.getIntField("max_pe_mem");
            int pe_mem_threshold = config.getIntField("threshold_pe_mem");
            double reconf_cycles = config.getFloatField("reconf_cycles");
            if (!reconf){
                pe_mem_size = 0;
                reconf_cycles =  -1; // fix for now
                
            }else{
                assert(reconf_cycles > 0.);
            }

            int reconf_batch_size = config.getIntField("reconf_batch_size");
            int flit_size = config.getIntField("flit_size");
            double comp_cycles = config.getFloatField("ANY_comp_cycles");

            _params[c] = DependentInjectionProcessParameters::New(
                _nodes,
                pe_mem_size,
                pe_mem_threshold,
                reconf_cycles,
                comp_cycles,
                reconf_batch_size,
                flit_size,
                &_clock,
                _traffic_pattern[c],
                &(_processed_packets[c])
            );
        }

        
        _injection_process[c] = _user_defined_traffic ? InjectionProcess::NewUserDefined(injection_process[c], _params[c], _context, &config) : InjectionProcess::New(injection_process[c], _nodes, _load[c], &config);

    }

    // ============ Injection VC states  ============ 
    _buf_states.resize(_nodes);
    _last_vc.resize(_nodes);
    _last_class.resize(_nodes);

    for ( int source = 0; source < _nodes; ++source ) {
        _buf_states[source].resize(_subnets);
        _last_class[source].resize(_subnets, 0);
        _last_vc[source].resize(_subnets);
        for ( int subnet = 0; subnet < _subnets; ++subnet ) {
            ostringstream tmp_name;
            tmp_name << "terminal_buf_state_" << source << "_" << subnet;
            BufferState * bs = new BufferState(config, context, this, tmp_name.str( ) );
            int vc_alloc_delay = config.getIntField("vc_alloc_delay");
            int sw_alloc_delay = config.getIntField("sw_alloc_delay");
            int router_latency = config.getIntField("routing_delay") + (config.getIntField("speculative") ? max(vc_alloc_delay, sw_alloc_delay) : (vc_alloc_delay + sw_alloc_delay));
            int min_latency = 1 + _net[subnet]->GetInject(source)->getLatency() + router_latency + _net[subnet]->GetInjectCred(source)->getLatency();
            bs->setMinLatency(min_latency);
            _buf_states[source][subnet] = bs;
            _last_vc[source][subnet].resize(_classes, -1);
        }
    }

#ifdef TRACK_FLOWS
    _outstanding_credits.resize(_classes);
    for(int c = 0; c < _classes; ++c) {
        _outstanding_credits[c].resize(_subnets, vector<int>(_nodes, 0));
    }
    _outstanding_classes.resize(_nodes);
    for(int n = 0; n < _nodes; ++n) {
        _outstanding_classes[n].resize(_subnets, vector<queue<int> >(_vcs));
    }
#endif

    // ============ Injection queues ============ 

    _qtime.resize(_nodes);
    _qdrained.resize(_nodes);
    _partial_packets.resize(_nodes);

    for ( int s = 0; s < _nodes; ++s ) {
        _qtime[s].resize(_classes);
        _qdrained[s].resize(_classes);
        _partial_packets[s].resize(_classes);
    }

    _total_in_flight_flits.resize(_classes);
    _measured_in_flight_flits.resize(_classes);
    _retired_packets.resize(_classes);

    _packet_seq_no.resize(_nodes);
    _repliesPending.resize(_nodes);
    _requestsOutstanding.resize(_nodes);

    _hold_switch_for_packet = config.getIntField("hold_switch_for_packet");

    // ============ Simulation parameters ============ 

    _total_sims = config.getIntField( "sim_count" );

    _router.resize(_subnets);
    for (int i=0; i < _subnets; ++i) {
        _router[i] = _net[i]->GetRouters();
    }

    //seed the network
    int seed;
    if(config.getStrField("seed") == "time") {
      seed = int(time(NULL));
      *(_context->gDumpFile) << "SEED: seed=" << seed << endl;
    } else {
      seed = config.getIntField("seed");
    }
    randomSeed(seed);

    _measure_latency = (config.getStrField("sim_type") == "latency");

    _sample_period = config.getIntField( "sample_period" );
    _max_samples    = config.getIntField( "max_samples" );
    _warmup_periods = config.getIntField( "warmup_periods" );

    _measure_stats = config.getIntArray( "measure_stats" );
    if(_measure_stats.empty()) {
        _measure_stats.push_back(config.getIntField("measure_stats"));
    }
    _measure_stats.resize(_classes, _measure_stats.back());
    _pair_stats = (config.getIntField("pair_stats")==1);

    _latency_thres = config.getFloatArray( "latency_thres" );
    if(_latency_thres.empty()) {
        _latency_thres.push_back(config.getFloatField("latency_thres"));
    }
    _latency_thres.resize(_classes, _latency_thres.back());

    _warmup_threshold = config.getFloatArray( "warmup_thres" );
    if(_warmup_threshold.empty()) {
        _warmup_threshold.push_back(config.getFloatField("warmup_thres"));
    }
    _warmup_threshold.resize(_classes, _warmup_threshold.back());

    _acc_warmup_threshold = config.getFloatArray( "acc_warmup_thres" );
    if(_acc_warmup_threshold.empty()) {
        _acc_warmup_threshold.push_back(config.getFloatField("acc_warmup_thres"));
    }
    _acc_warmup_threshold.resize(_classes, _acc_warmup_threshold.back());

    _stopping_threshold = config.getFloatArray( "stopping_thres" );
    if(_stopping_threshold.empty()) {
        _stopping_threshold.push_back(config.getFloatField("stopping_thres"));
    }
    _stopping_threshold.resize(_classes, _stopping_threshold.back());

    _acc_stopping_threshold = config.getFloatArray( "acc_stopping_thres" );
    if(_acc_stopping_threshold.empty()) {
        _acc_stopping_threshold.push_back(config.getFloatField("acc_stopping_thres"));
    }
    _acc_stopping_threshold.resize(_classes, _acc_stopping_threshold.back());

    _include_queuing = config.getIntField( "include_queuing" );

    _print_csv_results = config.getIntField( "print_csv_results" );
    _deadlock_warn_timeout = config.getIntField( "deadlock_warn_timeout" );

    string watch_file = config.getStrField( "watch_file" );
    if((watch_file != "") && (watch_file != "-")) {
        _LoadWatchList(watch_file);
    }

    vector<int> watch_flits = config.getIntArray("watch_flits");
    for(size_t i = 0; i < watch_flits.size(); ++i) {
        _flits_to_watch.insert(watch_flits[i]);
    }
  
    vector<int> watch_packets = config.getIntArray("watch_packets");
    for(size_t i = 0; i < watch_packets.size(); ++i) {
        _packets_to_watch.insert(watch_packets[i]);
    }

    string stats_out_file = config.getStrField( "stats_out" );
    if(stats_out_file == "") {
        _stats_out = NULL;
    } else if(stats_out_file == "-") {
        _stats_out = &cout;
    } else {
        _stats_out = new ofstream(stats_out_file.c_str());
        config.WriteMatlabFile(_stats_out);
    }

  
#ifdef TRACK_FLOWS
    _injected_flits.resize(_classes, vector<int>(_nodes, 0));
    _ejected_flits.resize(_classes, vector<int>(_nodes, 0));
    string injected_flits_out_file = config.getStrField( "injected_flits_out" );
    if(injected_flits_out_file == "") {
        _injected_flits_out = NULL;
    } else {
        _injected_flits_out = new ofstream(injected_flits_out_file.c_str());
    }
    string received_flits_out_file = config.getStrField( "received_flits_out" );
    if(received_flits_out_file == "") {
        _received_flits_out = NULL;
    } else {
        _received_flits_out = new ofstream(received_flits_out_file.c_str());
    }
    string stored_flits_out_file = config.getStrField( "stored_flits_out" );
    if(stored_flits_out_file == "") {
        _stored_flits_out = NULL;
    } else {
        _stored_flits_out = new ofstream(stored_flits_out_file.c_str());
    }
    string sent_flits_out_file = config.getStrField( "sent_flits_out" );
    if(sent_flits_out_file == "") {
        _sent_flits_out = NULL;
    } else {
        _sent_flits_out = new ofstream(sent_flits_out_file.c_str());
    }
    string outstanding_credits_out_file = config.getStrField( "outstanding_credits_out" );
    if(outstanding_credits_out_file == "") {
        _outstanding_credits_out = NULL;
    } else {
        _outstanding_credits_out = new ofstream(outstanding_credits_out_file.c_str());
    }
    string ejected_flits_out_file = config.getStrField( "ejected_flits_out" );
    if(ejected_flits_out_file == "") {
        _ejected_flits_out = NULL;
    } else {
        _ejected_flits_out = new ofstream(ejected_flits_out_file.c_str());
    }
    string active_packets_out_file = config.getStrField( "active_packets_out" );
    if(active_packets_out_file == "") {
        _active_packets_out = NULL;
    } else {
        _active_packets_out = new ofstream(active_packets_out_file.c_str());
    }
#endif

#ifdef TRACK_CREDITS
    string used_credits_out_file = config.getStrField( "used_credits_out" );
    if(used_credits_out_file == "") {
        _used_credits_out = NULL;
    } else {
        _used_credits_out = new ofstream(used_credits_out_file.c_str());
    }
    string free_credits_out_file = config.getStrField( "free_credits_out" );
    if(free_credits_out_file == "") {
        _free_credits_out = NULL;
    } else {
        _free_credits_out = new ofstream(free_credits_out_file.c_str());
    }
    string max_credits_out_file = config.getStrField( "max_credits_out" );
    if(max_credits_out_file == "") {
        _max_credits_out = NULL;
    } else {
        _max_credits_out = new ofstream(max_credits_out_file.c_str());
    }
#endif

    // ============ Statistics ============ 

    _plat_stats.resize(_classes);
    _overall_min_plat.resize(_classes, 0.0);
    _overall_avg_plat.resize(_classes, 0.0);
    _overall_max_plat.resize(_classes, 0.0);

    _nlat_stats.resize(_classes);
    _overall_min_nlat.resize(_classes, 0.0);
    _overall_avg_nlat.resize(_classes, 0.0);
    _overall_max_nlat.resize(_classes, 0.0);

    _flat_stats.resize(_classes);
    _overall_min_flat.resize(_classes, 0.0);
    _overall_avg_flat.resize(_classes, 0.0);
    _overall_max_flat.resize(_classes, 0.0);

    _frag_stats.resize(_classes);
    _overall_min_frag.resize(_classes, 0.0);
    _overall_avg_frag.resize(_classes, 0.0);
    _overall_max_frag.resize(_classes, 0.0);

    if(_pair_stats){
        _pair_plat.resize(_classes);
        _pair_nlat.resize(_classes);
        _pair_flat.resize(_classes);
    }
  
    _hop_stats.resize(_classes);
    _overall_hop_stats.resize(_classes, 0.0);
  
    _sent_packets.resize(_classes);
    _overall_min_sent_packets.resize(_classes, 0.0);
    _overall_avg_sent_packets.resize(_classes, 0.0);
    _overall_max_sent_packets.resize(_classes, 0.0);
    _accepted_packets.resize(_classes);
    _overall_min_accepted_packets.resize(_classes, 0.0);
    _overall_avg_accepted_packets.resize(_classes, 0.0);
    _overall_max_accepted_packets.resize(_classes, 0.0);

    _sent_flits.resize(_classes);
    _overall_min_sent.resize(_classes, 0.0);
    _overall_avg_sent.resize(_classes, 0.0);
    _overall_max_sent.resize(_classes, 0.0);
    _accepted_flits.resize(_classes);
    _overall_min_accepted.resize(_classes, 0.0);
    _overall_avg_accepted.resize(_classes, 0.0);
    _overall_max_accepted.resize(_classes, 0.0);

#ifdef TRACK_STALLS
    _buffer_busy_stalls.resize(_classes);
    _buffer_conflict_stalls.resize(_classes);
    _buffer_full_stalls.resize(_classes);
    _buffer_reserved_stalls.resize(_classes);
    _crossbar_conflict_stalls.resize(_classes);
    _overall_buffer_busy_stalls.resize(_classes, 0);
    _overall_buffer_conflict_stalls.resize(_classes, 0);
    _overall_buffer_full_stalls.resize(_classes, 0);
    _overall_buffer_reserved_stalls.resize(_classes, 0);
    _overall_crossbar_conflict_stalls.resize(_classes, 0);
#endif

    for ( int c = 0; c < _classes; ++c ) {
        ostringstream tmp_name;

        tmp_name << "plat_stat_" << c;
        _plat_stats[c] = new Stats( this, tmp_name.str( ), 1.0, 1000 );
        _stats[tmp_name.str()] = _plat_stats[c];
        tmp_name.str("");

        tmp_name << "nlat_stat_" << c;
        _nlat_stats[c] = new Stats( this, tmp_name.str( ), 1.0, 1000 );
        _stats[tmp_name.str()] = _nlat_stats[c];
        tmp_name.str("");

        tmp_name << "flat_stat_" << c;
        _flat_stats[c] = new Stats( this, tmp_name.str( ), 1.0, 1000 );
        _stats[tmp_name.str()] = _flat_stats[c];
        tmp_name.str("");

        tmp_name << "frag_stat_" << c;
        _frag_stats[c] = new Stats( this, tmp_name.str( ), 1.0, 100 );
        _stats[tmp_name.str()] = _frag_stats[c];
        tmp_name.str("");

        tmp_name << "hop_stat_" << c;
        _hop_stats[c] = new Stats( this, tmp_name.str( ), 1.0, 20 );
        _stats[tmp_name.str()] = _hop_stats[c];
        tmp_name.str("");

        if(_pair_stats){
            _pair_plat[c].resize(_nodes*_nodes);
            _pair_nlat[c].resize(_nodes*_nodes);
            _pair_flat[c].resize(_nodes*_nodes);
        }

        _sent_packets[c].resize(_nodes, 0);
        _accepted_packets[c].resize(_nodes, 0);
        _sent_flits[c].resize(_nodes, 0);
        _accepted_flits[c].resize(_nodes, 0);

#ifdef TRACK_STALLS
        _buffer_busy_stalls[c].resize(_subnets*_routers, 0);
        _buffer_conflict_stalls[c].resize(_subnets*_routers, 0);
        _buffer_full_stalls[c].resize(_subnets*_routers, 0);
        _buffer_reserved_stalls[c].resize(_subnets*_routers, 0);
        _crossbar_conflict_stalls[c].resize(_subnets*_routers, 0);
#endif
        if(_pair_stats){
            for ( int i = 0; i < _nodes; ++i ) {
                for ( int j = 0; j < _nodes; ++j ) {
                    tmp_name << "pair_plat_stat_" << c << "_" << i << "_" << j;
                    _pair_plat[c][i*_nodes+j] = new Stats( this, tmp_name.str( ), 1.0, 250 );
                    _stats[tmp_name.str()] = _pair_plat[c][i*_nodes+j];
                    tmp_name.str("");
	  
                    tmp_name << "pair_nlat_stat_" << c << "_" << i << "_" << j;
                    _pair_nlat[c][i*_nodes+j] = new Stats( this, tmp_name.str( ), 1.0, 250 );
                    _stats[tmp_name.str()] = _pair_nlat[c][i*_nodes+j];
                    tmp_name.str("");
	  
                    tmp_name << "pair_flat_stat_" << c << "_" << i << "_" << j;
                    _pair_flat[c][i*_nodes+j] = new Stats( this, tmp_name.str( ), 1.0, 250 );
                    _stats[tmp_name.str()] = _pair_flat[c][i*_nodes+j];
                    tmp_name.str("");
                }
            }
        }
    }

    _slowest_flit.resize(_classes, -1);
    _slowest_packet.resize(_classes, -1);

 

}

TrafficManager::~TrafficManager( )
{


    for ( int source = 0; source < _nodes; ++source ) {
        for ( int subnet = 0; subnet < _subnets; ++subnet ) {
            delete _buf_states[source][subnet];
        }
    }
  
    for ( int c = 0; c < _classes; ++c ) {
        delete _plat_stats[c];
        delete _nlat_stats[c];
        delete _flat_stats[c];
        delete _frag_stats[c];
        delete _hop_stats[c];

        delete _traffic_pattern[c];
        delete _injection_process[c];
        if(_pair_stats){
            for ( int i = 0; i < _nodes; ++i ) {
                for ( int j = 0; j < _nodes; ++j ) {
                    delete _pair_plat[c][i*_nodes+j];
                    delete _pair_nlat[c][i*_nodes+j];
                    delete _pair_flat[c][i*_nodes+j];
                }
            }
        }
    }
  
    if((_context->gWatchOut) && ((_context->gWatchOut) != &cout)) delete (_context->gWatchOut);
    if(_stats_out && (_stats_out != &cout)) delete _stats_out;

#ifdef TRACK_FLOWS
    if(_injected_flits_out) delete _injected_flits_out;
    if(_received_flits_out) delete _received_flits_out;
    if(_stored_flits_out) delete _stored_flits_out;
    if(_sent_flits_out) delete _sent_flits_out;
    if(_outstanding_credits_out) delete _outstanding_credits_out;
    if(_ejected_flits_out) delete _ejected_flits_out;
    if(_active_packets_out) delete _active_packets_out;
#endif

#ifdef TRACK_CREDITS
    if(_used_credits_out) delete _used_credits_out;
    if(_free_credits_out) delete _free_credits_out;
    if(_max_credits_out) delete _max_credits_out;
#endif

    packet_reply_pool.freeAllReplies();
    flit_pool.freeAllFlits();
    credit_pool.freeAllCredits();
}


void TrafficManager::_RetireFlit( Flit *f, int dest )
{
    _deadlock_timer = 0;

    assert(_total_in_flight_flits[f->cl].count(f->id) > 0);
    _total_in_flight_flits[f->cl].erase(f->id);
  
    if(f->record) {
        assert(_measured_in_flight_flits[f->cl].count(f->id) > 0);
        _measured_in_flight_flits[f->cl].erase(f->id);
    }

    if ( f->watch ) { 
        *(_context->gWatchOut) << getTime() << " | "
                   << "node" << dest << " | "
                   << "Retiring flit " << f->id 
                   << " (packet " << f->pid
                   << ", src = " << f->src 
                   << ", dest = " << f->dst
                   << ", hops = " << f->hops
                   << ", flat = " << f->atime - f->itime
                   << ")." << endl;
    }

    if ( f->head && ( f->dst != dest ) ) {
        ostringstream err;
        err << "Flit " << f->id << " arrived at incorrect output " << dest;
        error( err.str( ) );
    }
  
    if((_slowest_flit[f->cl] < 0) ||
       (_flat_stats[f->cl]->Max() < (f->atime - f->itime)))
        _slowest_flit[f->cl] = f->id;
    _flat_stats[f->cl]->AddSample( f->atime - f->itime);
    if(_pair_stats){
        _pair_flat[f->cl][f->src*_nodes+dest]->AddSample( f->atime - f->itime );
    }
      
    if ( f->tail ) {
        Flit * head;
        if(f->head) {
            head = f;
        } else {
            map<int, Flit *>::iterator iter = _retired_packets[f->cl].find(f->pid);
            assert(iter != _retired_packets[f->cl].end());
            head = iter->second;
            _retired_packets[f->cl].erase(iter);
            assert(head->head);
            assert(f->pid == head->pid);
        }
        if ( f->watch ) { 
            *(_context->gWatchOut) << getTime() << " | "
                       << "node" << dest << " | "
                       << "Retiring packet " << f->pid
                       << " / rpid  " << f->rpid
                       << " (plat = " << f->atime - head->ctime
                       << ", nlat = " << f->atime - head->itime
                       << ", frag = " << (f->atime - head->atime) - (f->id - head->id) // NB: In the spirit of solving problems using ugly hacks, we compute the packet length by taking advantage of the fact that the IDs of flits within a packet are contiguous.
                       << ", src = " << head->src
                       << ", dest = " << head->dst
                       << ")." << endl;
        }

        // append the packet id to the set of landed packets
        // if the packet is a request, assert that it is not already there
        // if the packet is a reply, assert that there is one request already
        if(_use_read_write[f->cl]) {
            int processing_time = 0;
            if(f->type == commType::READ_REQ || f->type == commType::WRITE_REQ || f->type == commType::READ || f->type == commType::WRITE) {
                // search in the destination node landed packets list
                auto it = std::find_if(_landed_packets[f->cl][dest].begin(), _landed_packets[f->cl][dest].end(), [f](const std::tuple<int, int, int, int> & p) { return get<0>(p) == f->rpid && get<1>(p) == f->type; });
                assert(it == _landed_packets[f->cl][dest].end());
                _landed_packets[f->cl][dest].insert(std::make_tuple(f->rpid, f->type, _clock.time(),f->size));

                // ================ MANAGING OF PROCESSING TIMES FOR PACKETS ================ 
                
                if(f->type == commType::WRITE || f->type == commType::READ){
                    processing_time = f->data_ptime_expected;
                    *(_context->gDumpFile) << " Bulk packet with id: "<<f->rpid << " and type: "<< f->type << " arrived at: "<< dest << " at time: " << _clock.time() << " from node: "<< f->src << ", size: "<< f->size << std::endl;
                    *(_context->gDumpFile) << "Processing time: " << processing_time << std::endl;
                    if (_params[f->cl]->localMemSize > 0 && f->type == commType::WRITE && f->data_dep != -1) {
                        // finalize the write for reconfiguration
                        _injection_process[f->cl]->finalizeWrite(dest, f->data_dep, f->rpid);
                    }
                }
                else{
                    *(_context->gDumpFile) << " Read/Write REQ packet with id: "<<f->rpid << " and type: "<< f->type << " arrived at: "<< dest << " at time: " << _clock.time() << " from node: "<< f->src << ", size: "<< f->size << std::endl;
                    processing_time = (f->type == commType::WRITE_REQ) ? _write_request_time[f->cl] : _read_request_time[f->cl];
                    
                }
                _to_process_packets[f->cl][dest].insert(std::make_tuple(f->rpid, f->type, _clock.time() + processing_time)); 
                // ==========================================================================
            } else if (f->type == commType::READ_ACK || f->type == commType::WRITE_ACK) {
                // search in the reply source node (destination node of the replied packet) landed packets list
                int req_type1, req_type2;
                if (f->type == commType::READ_ACK) {
                    req_type1 = commType::READ_REQ;
                    req_type2 = commType::READ;
                    processing_time = _read_reply_time[f->cl];
                } else {
                    req_type1 = commType::WRITE_REQ;
                    req_type2 = commType::WRITE;
                    processing_time = _write_reply_time[f->cl];
                }
                *(_context->gDumpFile) << " Reply packet with id: "<<f->rpid << " and type: "<< f->type << " arrived at: "<< dest << " at time: " << _clock.time() << " from node: "<< f->src << std::endl;
                auto it = std::find_if(_landed_packets[f->cl][f->src].begin(), _landed_packets[f->cl][f->src].end(), [f, req_type1, req_type2](const std::tuple<int, int, int, int> & p) { return get<0>(p) == f->rpid && (get<1>(p) == req_type1 || get<1>(p) == req_type2); });
                assert(it != _landed_packets[f->cl][f->src].end());
                // if the packet is a reply and its corresponding request if a WRITE, deallocate from 
                // the local memory the space used to store the output data (packet size)
                if (_params[f->cl]->localMemSize > 0 && f->type == commType::WRITE_ACK && f->data_dep != -1) {
                    
                    // when a write reply is received, we finalize the communication for reconfiguration
                    _injection_process[f->cl]->finalizeCommunication(f->dst, f->data_dep, f->rpid);

                }
            }
            _injection_process[f->cl]->requestInterrupt(dest, processing_time);
        }

        if (_context->logger) {
            _context->logger->register_event(EventType::IN_TRAFFIC, _clock.time(), f->rpid, f->type);
        }

        //code the source of request, look carefully, its tricky ;
        if (/*f->type == commType::READ_REQ || f->type == commType::WRITE_REQ ||*/ f->type == commType::READ || f->type == commType::WRITE) {
            PacketReplyInfo* rinfo = PacketReplyInfo::newReply(packet_reply_pool);
            rinfo->source = f->src;
            rinfo->src = f->src;
            rinfo->dst = f->dst;
            rinfo->time = f->atime; 
            rinfo->record = f->record;
            rinfo->type = f->type;
            rinfo->rpid = f->rpid;
            rinfo->data_dep = f->data_dep;
            _repliesPending[dest].push_back(rinfo);

            if (_context->logger) {
                commType rtype = (f->type == commType::READ) ? commType::READ_ACK : commType::WRITE_ACK;
                EventInfo* info = new TrafficEventInfo(f->rpid, rtype, dest, f->src, f->size);
                _context->logger->add_tevent_info(info);
            }



        }else {
            if(f->type == commType::READ_ACK || f->type == commType::WRITE_ACK ){
                _requestsOutstanding[dest]--;
            } else if(f->type == commType::ANY) {
                _requestsOutstanding[f->src]--;
            }
        }

        // AUTOMATIC GENERATION OF READ_REQ AND WRITEs AFTER RECEIVED MESSAGES
        if ((f->type == commType::WRITE_REQ || f->type == commType::READ_REQ) && _traffic_pattern[f->cl]->check_user_defined()) {
            // size of the write (data to be tranferred) is taken from
            // the size field of the landed packet element that originated the read request
            int size = f->data_size;
            // same with time
            int time = f->data_ptime_expected;
            // no dependecies to be resolved
            std::vector<int> dep(1, -1);
            dep[0] = f->type == commType::READ_REQ ? f->data_dep : -1;
            int data_dep = f->data_dep;
            int type = (f->type == commType::WRITE_REQ) ? commType::READ_REQ : commType::WRITE;
            if (f->type == commType::WRITE_REQ){
                *(_context->gDumpFile) << "Generating READ_REQ (id: "<< f->rpid <<") packet at time: " << _clock.time() << " from node: " << dest << " to node: " << f->src << std::endl;
            } else {
                *(_context->gDumpFile) << "Generating WRITE (id: "<< f->rpid <<") packet at time: " << _clock.time() << " from node: " << dest << " to node: " << f->src << std::endl;
            }
            Packet* p = new Packet{f->rpid, f->dst, f->src, size, dep, f->cl, type, time, 0 , data_dep}; 
            // append this new packet to the list of packets that are waiting to be processed
            _injection_process[f->cl]->addToWaitingQueue(f->dst, p);
            
        }

        // Only record statistics once per packet (at tail)
        // and based on the simulation state
        if ( ( _sim_state == warming_up ) || f->record ) {
      
            _hop_stats[f->cl]->AddSample( f->hops );

            if((_slowest_packet[f->cl] < 0) ||
               (_plat_stats[f->cl]->Max() < (f->atime - head->itime)))
                _slowest_packet[f->cl] = f->pid;
            _plat_stats[f->cl]->AddSample( f->atime - head->ctime);
            _nlat_stats[f->cl]->AddSample( f->atime - head->itime);
            _frag_stats[f->cl]->AddSample( (f->atime - head->atime) - (f->id - head->id) );
   
            if(_pair_stats){
                _pair_plat[f->cl][f->src*_nodes+dest]->AddSample( f->atime - head->ctime );
                _pair_nlat[f->cl][f->src*_nodes+dest]->AddSample( f->atime - head->itime );
            }
        }
    
        if(f != head) {
            head->freeFlit(flit_pool);
        }
    
    }
  
    if(f->head && !f->tail) {
        _retired_packets[f->cl].insert(make_pair(f->pid, f));
    } else {
        f->freeFlit(flit_pool);
    }
}

int TrafficManager::_IssuePacket( int source, int cl )
{
    int result = 0;

    if(_use_read_write[cl]){ //use read and write
        //check queue for waiting replies.
        //check to make sure it is on time yet
        if (!_repliesPending[source].empty()) {
            // if the reply queue is not empty (there are valid replies waiting), then
            // the test function has not been called, meaning that the processing time has not decurred.
            // But in general this processing time is used to model the time it takes the processor
            // to elaborate the packet, not the node of the NOC. So it should still decur even when
            // the replies are being processed by the source.
            if(_repliesPending[source].front()->time <= _clock.time()) {
                result = -1;
            }
        } else {

            
            // if the traffic pattern is user defined, the injection process
            // cannot be simply modeled using injection rate, but must take into account
            // the presence in the net of not-yet-arrived packets from which next packets
            // depend on, as well as the time necessary for each node to actually process
            // the sent. This will be modelled in the injection process itself.

            // Produce a packet
            if(_injection_process[cl]->test(source)) {
                
                if(_traffic_pattern[cl]->check_user_defined()){
                    // user defined type for the next packet
                    result = _traffic_pattern[cl]->cur_packet->type;
                } else {
                    //coin toss to determine request type.
                    result = (randomFloat() < _write_fraction[cl]) ? 2 : 1;
                }
	
                _requestsOutstanding[source]++;
            }
        }
    } else { //normal mode
        if(_traffic_pattern[cl]->check_user_defined()){
            error("Custom traffic pattern is used without read/write enabled");
        }
        result = _injection_process[cl]->test(source) ? 1 : 0;
        _requestsOutstanding[source]++;
    } 
    if(result != 0) {
        _packet_seq_no[source]++;
    }
    return result;
}

void TrafficManager::_GeneratePacket( int source, int stype, 
                                      int cl, int time )
{
    assert(stype!=0);

    commType packet_type = commType::ANY;
    int size = _GetNextPacketSize(cl); //size of the packets, user defined for UserDefined traffic pattern
    int pid = _cur_pid++;
    assert(_cur_pid);
    // check that the packet id is the same as the one referenced by the traffic pattern
    int packet_destination = -1;
    if (!_traffic_pattern[cl]->check_user_defined()) {
        packet_destination = _traffic_pattern[cl]->dest(source);
    }
    int rpid = -1;
    int data_dep = -1;
    

    bool record = false;
    bool watch = (_context->gWatchOut) && (_packets_to_watch.count(pid) > 0);
    if(_use_read_write[cl]){
        if(stype > 0) {    
            packet_destination = _traffic_pattern[cl]->dest(source);
            rpid = _traffic_pattern[cl]->cur_packet->id;
            data_dep = _traffic_pattern[cl]->cur_packet->data_dep;
            if (stype == 1) {
                packet_type = commType::READ_REQ;
                size = _read_request_size[cl];
            } else if (stype == 2) {
                packet_type = commType::WRITE_REQ;
                size = _write_request_size[cl];
                // data_dep = _traffic_pattern[cl]->cur_packet->dep[0];
            } else if (stype == 5) {
                packet_type = commType::READ;
                size = (_traffic_pattern[cl]->check_user_defined()) ? _traffic_pattern[cl]->cur_packet->size : _read_request_size[cl];
            } else if (stype == 6) {
                packet_type = commType::WRITE;
                size = (_traffic_pattern[cl]->check_user_defined()) ? _traffic_pattern[cl]->cur_packet->size : _write_request_size[cl];
            } else {
                ostringstream err;
                err << "Invalid packet type: " << packet_type;
                error( err.str( ) );
            }
        } else {
            int processing_time = 0;
            PacketReplyInfo* rinfo = _repliesPending[source].front();
            if (rinfo->type == commType::READ_REQ || rinfo->type == commType::READ) {//read reply
                size = _read_reply_size[cl];
                packet_type = commType::READ_ACK;
                processing_time = _read_reply_time[cl];

            } else if(rinfo->type == commType::WRITE_REQ || rinfo->type == commType::WRITE) {  //write reply
                size = _write_reply_size[cl];
                packet_type = commType::WRITE_ACK;
                processing_time = _write_reply_time[cl];
            } else {
                ostringstream err;
                err << "Invalid packet type: " << rinfo->type;
                error( err.str( ) );
            }
            packet_destination = rinfo->source;
            time = rinfo->time;
            record = rinfo->record;
            rpid = rinfo->rpid;
            data_dep = rinfo->data_dep;
            _repliesPending[source].pop_front();
            rinfo->freeReply(packet_reply_pool);

            // request an interrupt if the node is computing or reconfiguring
            _injection_process[cl]->requestInterrupt(source, processing_time);
        }
    }

    if ((packet_destination <0) || (packet_destination >= _nodes)) {
        ostringstream err;
        err << "Incorrect packet destination " << packet_destination
            << " for stype " << packet_type;
        error( err.str( ) );
    }

    if ( ( _sim_state == running ) ||
         ( ( _sim_state == draining ) && ( time < _drain_time ) ) ) {
        record = _measure_stats[cl];
    }

    int subnetwork = ((packet_type == commType::ANY) ? 
                      randomInt(_subnets-1) :
                      _subnet[packet_type]);
  
    if ( watch ) { 
        *(_context->gWatchOut) << getTime() << " | "
                   << "node" << source << " | "
                   << "Enqueuing packet " << pid << "/ rpid " << rpid
                   << " at time " << time
                   << "." << endl;
    }

  
    for ( int i = 0; i < size; ++i ) {
        Flit * f  = Flit::newFlit(flit_pool);
        f->id     = _cur_id++;
        assert(_cur_id);
        f->pid    = pid;
        f->rpid   = rpid;
        f->size   = size;
        f->data_size = _traffic_pattern[cl]->check_user_defined() ? _traffic_pattern[cl]->cur_packet->size : 0;
        f->data_ptime_expected = _traffic_pattern[cl]->check_user_defined() ? _traffic_pattern[cl]->cur_packet->pt_required : 0;
        f->watch  = watch | ((_context->gWatchOut) && (_flits_to_watch.count(f->id) > 0));
        f->subnetwork = subnetwork;
        f->src    = source;
        f->ctime  = time;
        f->record = record;
        f->cl     = cl;
        f->data_dep = data_dep;

        _total_in_flight_flits[f->cl].insert(make_pair(f->id, f));
        if(record) {
            _measured_in_flight_flits[f->cl].insert(make_pair(f->id, f));
        }
    
        if(_context->gTrace){
            *(_context->gDumpFile)<<"New Flit "<<f->src<<endl;
        }
        f->type = packet_type;

        if ( i == 0 ) { // Head flit
            f->head = true;
            //packets are only generated to nodes smaller or equal to limit
            f->dst = packet_destination;
        } else {
            f->head = false;
            f->dst = -1;
        }
        switch( _pri_type ) {
        case class_based:
            f->priority = _class_priority[cl];
            assert(f->priority >= 0);
            break;
        case age_based:
            f->priority = numeric_limits<int>::max() - time;
            assert(f->priority >= 0);
            break;
        case sequence_based:
            f->priority = numeric_limits<int>::max() - _packet_seq_no[source];
            assert(f->priority >= 0);
            break;
        default:
            f->priority = 0;
        }
        if ( i == ( size - 1 ) ) { // Tail flit
            f->tail = true;
        } else {
            f->tail = false;
        }
    
        f->vc  = -1;

        if ( f->watch ) { 
            *(_context->gWatchOut) << getTime() << " | "
                       << "node" << source << " | "
                       << "Enqueuing flit " << f->id
                       << " (packet " << f->pid << " / rpid "<< f->rpid 
                       << ") at time " << time
                       << "." << endl;
        }

        _partial_packets[source][cl].push_back( f );
    }
}

void TrafficManager::_Inject(){
    //*(_context->gDumpFile) << "==================== Inject ====================" << std::endl;
    for ( int input = 0; input < _nodes; ++input ) {
        for ( int c = 0; c < _classes; ++c ) {
            // Potentially generate packets for any (input,class)
            // that is currently empty
            // This already checks that there are no contentions for the node
            if ( _partial_packets[input][c].empty() ) {
                bool generated = false;
                int i = 0;
                while( !generated && ( _qtime[input][c] <= _clock.time() ) ) {
                    int stype = _IssuePacket( input, c );
                    //*(_context->gDumpFile) << "Time: " << _clock.time() << " | Node: " << input << " | Class: " << c << " | Stype: " << stype << std::endl;
            
                    if ( stype != 0 ) { //generate a packet (only requests)
                        _GeneratePacket( input, stype, c, 
                                         _include_queuing==1 ? 
                                         _qtime[input][c] : _clock.time() );
                        generated = true;
                    }
                    // only advance time if this is not a reply packet
                    if(!_use_read_write[c] || (stype >= 0)){
                        ++_qtime[input][c];
                    }
                }
	
                if ( ( _sim_state == draining ) && 
                     ( _qtime[input][c] > _drain_time ) ) {
                    _qdrained[input][c] = true;
                }
            }
        }
    }
}

void TrafficManager::_Step( )
{
    bool flits_in_flight = false;
    for(int c = 0; c < _classes; ++c) {
        flits_in_flight |= !_total_in_flight_flits[c].empty();
    }
    if(flits_in_flight && (_deadlock_timer++ >= _deadlock_warn_timeout)){
        _deadlock_timer = 0;
        *(_context->gDumpFile) << "WARNING: Possible network deadlock.\n";
    }

    vector<map<int, Flit *> > flits(_subnets);

    // For each subnet and each node, read the flits arriving at the node
    for ( int subnet = 0; subnet < _subnets; ++subnet ) {
        for ( int n = 0; n < _nodes; ++n ) {
            Flit * const f = _net[subnet]->ReadFlit( n );
            if ( f ) {
                if(f->watch) {
                    *(_context->gWatchOut) << getTime() << " | "
                               << "node" << n << " | "
                               << "Ejecting flit " << f->id
                               << " (packet " << f->pid << ")"
                               << " from VC " << f->vc
                               << "." << endl;
                }
                
                flits[subnet].insert(make_pair(n, f));
                if((_sim_state == warming_up) || (_sim_state == running)) {
                    ++_accepted_flits[f->cl][n];
                    if(f->tail) {
                        ++_accepted_packets[f->cl][n];
                    }
                }
            }

            Credit * const c = _net[subnet]->ReadCredit( n );
            if ( c ) {
#ifdef TRACK_FLOWS
                for(set<int>::const_iterator iter = c->vc.begin(); iter != c->vc.end(); ++iter) {
                    int const vc = *iter;
                    assert(!_outstanding_classes[n][subnet][vc].empty());
                    int cl = _outstanding_classes[n][subnet][vc].front();
                    _outstanding_classes[n][subnet][vc].pop();
                    assert(_outstanding_credits[cl][subnet][n] > 0);
                    --_outstanding_credits[cl][subnet][n];
                }
#endif
                _buf_states[n][subnet]->processCredit(c);
                c->freeCredit(credit_pool);
            }
        }
        _net[subnet]->readInputs( );
    }
  
    if ( !_empty_network ) {
        // Until the empty (draning) flag is on, we keep injecting flits for each step
        _Inject();
    }

    for(int subnet = 0; subnet < _subnets; ++subnet) {

        for(int n = 0; n < _nodes; ++n) {

            Flit * f = NULL;

            BufferState * const dest_buf = _buf_states[n][subnet];

            // Check for the last class that had a flit pass through the selected node
            int const last_class = _last_class[n][subnet];

            int class_limit = _classes;

            if(_hold_switch_for_packet) {
                list<Flit *> const & pp = _partial_packets[n][last_class];
                // Se la lista non è vuota e il flit in cima non è head flit e la destinazione per il VC non è piena
                if(!pp.empty() && !pp.front()->head && 
                   !dest_buf->isFullFor(pp.front()->vc)) {
                    // The flit considered is the one on top of the list
                    f = pp.front();
                    assert(f->vc == _last_vc[n][subnet][last_class]);

                    // if we're holding the connection, we don't need to check that class 
                    // again in the for loop
                    --class_limit;
                }
            }

            for(int i = 1; i <= class_limit; ++i) {
                
                // Round robin over classes
                int const c = (last_class + i) % _classes;

                list<Flit *> const & pp = _partial_packets[n][c];
                
                // se la lista per quella classe e quel nodo è vuota, passa alla prossima classe
                if(pp.empty()) {
                    continue;
                }

                Flit * const cf = pp.front();  
                assert(cf);
                assert(cf->cl == c);

                // se il flit non appartiene alla subnet corrente, passa alla prossima classe
                if(cf->subnetwork != subnet) {
                    continue;
                }

                // se esiste un flit in coda per l'ultima classe che non sia head e per cui la destinazione per il VC non sia piena E ...
                // se la priorità del flit in coda per l'ultima classe è maggiore di quella del flit corrente, passa alla prossima classe
                if(f && (f->priority >= cf->priority)) {
                    continue;
                }

                // in caso contrario, se il flit è di testa e non è ancora stato assegnato un VC,
                // cerchiamo il primo VC disponibile
                if(cf->head && cf->vc == -1) { // Find first available VC
	  
                    OutSet route_set;
                    _rf(_context, NULL, _net[subnet]->par, cf, -1, &route_set, true);
                    set<OutSet::sSetElement> const & os = route_set.getOutSet();
                    assert(os.size() == 1);
                    OutSet::sSetElement const & se = *os.begin();
                    assert(se.output_port == -1);
                    int vc_start = se.vc_start;
                    int vc_end = se.vc_end;
                    int vc_count = vc_end - vc_start + 1;
                    if(_noq) {
                        assert(_lookahead_routing);
                        const FlitChannel * inject = _net[subnet]->GetInject(n);
                        const Router * router = inject->getSnkRouter();
                        assert(router);
                        int in_channel = inject->getSnkPort();

                        // NOTE: Because the lookahead is not for injection, but for the 
                        // first hop, we have to temporarily set cf's VC to be non-negative 
                        // in order to avoid seting of an assertion in the routing function.
                        cf->vc = vc_start;
                        _rf(_context, router, _net[subnet]->par, cf, in_channel, &cf->la_route_set, false);
                        cf->vc = -1;

                        if(cf->watch) {
                            *(_context->gWatchOut) << getTime() << " | "
                                       << "node" << n << " | "
                                       << "Generating lookahead routing info for flit " << cf->id
                                       << " (NOQ)." << endl;
                        }
                        set<OutSet::sSetElement> const sl = cf->la_route_set.getOutSet();
                        assert(sl.size() == 1);
                        int next_output = sl.begin()->output_port;
                        vc_count /= router->NumOutputs();
                        vc_start += next_output * vc_count;
                        vc_end = vc_start + vc_count - 1;
                        assert(vc_start >= se.vc_start && vc_start <= se.vc_end);
                        assert(vc_end >= se.vc_start && vc_end <= se.vc_end);
                        assert(vc_start <= vc_end);
                    }
                    if(cf->watch) {
                        *(_context->gWatchOut) << getTime() << " | " << getFullName() << " | "
                                   << "Finding output VC for flit " << cf->id
                                   << ":" << endl;
                    }
                    for(int i = 1; i <= vc_count; ++i) {
                        int const lvc = _last_vc[n][subnet][c];
                        int const vc =
                            (lvc < vc_start || lvc > vc_end) ?
                            vc_start :
                            (vc_start + (lvc - vc_start + i) % vc_count);
                        assert((vc >= vc_start) && (vc <= vc_end));
                        if(!dest_buf->isAvailableFor(vc)) {
                            if(cf->watch) {
                                *(_context->gWatchOut) << getTime() << " | " << getFullName() << " | "
                                           << "  Output VC " << vc << " is busy." << endl;
                            }
                        } else {
                            if(dest_buf->isFullFor(vc)) {
                                if(cf->watch) {
                                    *(_context->gWatchOut) << getTime() << " | " << getFullName() << " | "
                                               << "  Output VC " << vc << " is full." << endl;
                                }
                            } else {
                                if(cf->watch) {
                                    *(_context->gWatchOut) << getTime() << " | " << getFullName() << " | "
                                               << "  Selected output VC " << vc << "." << endl;
                                }
                                cf->vc = vc;
                                break;
                            }
                        }
                    }
                }

                if(cf->vc == -1) {
                    if(cf->watch) {
                        *(_context->gWatchOut) << getTime() << " | " << getFullName() << " | "
                                   << "No output VC found for flit " << cf->id
                                   << "." << endl;
                    }
                } else {
                    if(dest_buf->isFullFor(cf->vc)) {
                        if(cf->watch) {
                            *(_context->gWatchOut) << getTime() << " | " << getFullName() << " | "
                                       << "Selected output VC " << cf->vc
                                       << " is full for flit " << cf->id
                                       << "." << endl;
                        }
                    } else {
                        f = cf;
                    }
                }
            }
            
            if(f) {

                assert(f->subnetwork == subnet);

                int const c = f->cl;

                if(f->head) {
	  
                    if (_lookahead_routing) {
                        if(!_noq) {
                            const FlitChannel * inject = _net[subnet]->GetInject(n);
                            const Router * router = inject->getSnkRouter();
                            assert(router);
                            int in_channel = inject->getSnkPort();
                            _rf(_context, router, _net[subnet]->par, f, in_channel, &f->la_route_set, false);
                            if(f->watch) {
                                *(_context->gWatchOut) << getTime() << " | "
                                           << "node" << n << " | "
                                           << "Generating lookahead routing info for flit " << f->id
                                           << "." << endl;
                            }
                        } else if(f->watch) {
                            *(_context->gWatchOut) << getTime() << " | "
                                       << "node" << n << " | "
                                       << "Already generated lookahead routing info for flit " << f->id
                                       << " (NOQ)." << endl;
                        }
                    } else {
                        f->la_route_set.clear();
                    }

                    dest_buf->takeBuffer(f->vc);
                    _last_vc[n][subnet][c] = f->vc;
                }
	
                _last_class[n][subnet] = c;

                _partial_packets[n][c].pop_front();

#ifdef TRACK_FLOWS
                ++_outstanding_credits[c][subnet][n];
                _outstanding_classes[n][subnet][f->vc].push(c);
#endif

                dest_buf->sendingFlit(f);
	
                if(_pri_type == network_age_based) {
                    f->priority = numeric_limits<int>::max() - _clock.time();
                    assert(f->priority >= 0);
                }
	
                if(f->watch) {
                    *(_context->gWatchOut) << getTime() << " | "
                               << "node" << n << " | "
                               << "Injecting flit " << f->id
                               << " into subnet " << subnet
                               << " at time " << _clock.time()
                               << " with priority " << f->priority
                               << "." << endl;
                }
                f->itime = _clock.time();

                if (_context->logger && f->head) {
                    _context->logger->register_event(EventType::OUT_TRAFFIC, _clock.time(), f->rpid, f->type);
                }
                


                // Pass VC "back"
                if(!_partial_packets[n][c].empty() && !f->tail) {
                    Flit * const nf = _partial_packets[n][c].front();
                    nf->vc = f->vc;
                }
	
                if((_sim_state == warming_up) || (_sim_state == running)) {
                    ++_sent_flits[c][n];
                    if(f->head) {
                        ++_sent_packets[c][n];
                    }
                }
	
#ifdef TRACK_FLOWS
                ++_injected_flits[c][n];
#endif
	
                _net[subnet]->WriteFlit(f, n);
	
            }
        }
    }

    for(int subnet = 0; subnet < _subnets; ++subnet) {
        for(int n = 0; n < _nodes; ++n) {
            map<int, Flit *>::const_iterator iter = flits[subnet].find(n);
            if(iter != flits[subnet].end()) {
                Flit * const f = iter->second;

                f->atime = _clock.time();
                if(f->watch) {
                    *(_context->gWatchOut) << getTime() << " | "
                               << "node" << n << " | "
                               << "Injecting credit for VC " << f->vc 
                               << " into subnet " << subnet 
                               << "." << endl;
                }
                Credit * const c = Credit::newCredit(credit_pool);
                c->vc.insert(f->vc);
                _net[subnet]->WriteCredit(c, n);
	
#ifdef TRACK_FLOWS
                ++_ejected_flits[f->cl][n];
#endif
	
                _RetireFlit(f, n);
            }
        }
        flits[subnet].clear();
        _net[subnet]->evaluate( );
        _net[subnet]->writeOutputs( );
    }
    
    // ==========================================
    for(int c = 0; c < _classes; ++c) {
        for(int n=0; n < _nodes; ++n) {
            // for each node, check in the _to_process_packets list if there are packets that have 
            // elapsed the processing time
            auto it = _to_process_packets[c][n].begin();
            while(it != _to_process_packets[c][n].end()) {

                // find the corresponding packet in the landed packets list
                auto it_land = std::find_if(_landed_packets[c][n].begin(), _landed_packets[c][n].end(), [it](const std::tuple<int, int, int, int> & p) { return (get<0>(*it) == get<0>(p)) && (get<1>(*it) == get<1>(p));});
                _useful_pocessing_spots[c][n] += (_injection_process[c]->isIdle(n) ? 1 : 0);


                // if the two times match, then the packet has elapsed the processing time
                if(get<2>(*it) < get<2>(*it_land) + _useful_pocessing_spots[c][n]) {
                    // insert the packet in the processed packets list
                    _processed_packets[c][n].insert(*it);
                    // erase the packet from the to process packets list
                    it = _to_process_packets[c][n].erase(it);
                    // reset the useful processing spots counter
                    _useful_pocessing_spots[c][n] = 0;
                } else {
                    ++it;
                }

                // // if the packet has elapsed the processing time, then it can be moved to the _processed_packets list
                // if(get<2>(*it) <= _clock.time()) {
                //     _processed_packets[c][n].insert(*it);
                //     it = _to_process_packets[c][n].erase(it);
                // } else {
                //     ++it;
                // }
            }
        }
    }

    // print the elements of processed packets at each time step
    // *(_context->gDumpFile) << "==================== Processed Packets ====================" << std::endl;
    // *(_context->gDumpFile) << "                      Time: " << _clock.time() << std::endl;
    // for(int c = 0; c < _classes; ++c) {
    //     for(int n = 0; n < _nodes; ++n) {
    //         for(auto it = _processed_packets[c][n].begin(); it != _processed_packets[c][n].end(); ++it) {
    //             *(_context->gDumpFile) << " Class: " << c << " | Node: " << n << " | Request ID: " << get<0>(*it) << " | Type: " << get<1>(*it) << " | Time: " << get<2>(*it) << std::endl;
    //         }
    //     }
    // }
    // *(_context->gDumpFile) << "=============================================================" << std::endl;
        
    // ==========================================

    _clock.tick( );
    assert(_clock.time());
    if(_context->gTrace){
        *(_context->gDumpFile)<<"TIME "<<_clock.time()<<endl;
    }

}
  
bool TrafficManager::_PacketsOutstanding( ) const
{
    for ( int c = 0; c < _classes; ++c ) {
        if ( _measure_stats[c] ) {
            if ( _measured_in_flight_flits[c].empty() ) {
	
                for ( int s = 0; s < _nodes; ++s ) {
                    if ( !_qdrained[s][c] ) {
#ifdef DEBUG_DRAIN
                        *(_context->gDumpFile) << "waiting on queue " << s << " class " << c;
                        *(_context->gDumpFile) << ", time = " << _clock.time() << " qtime = " << _qtime[s][c] << endl;
#endif
                        return true;
                    }
                }
            } else {
#ifdef DEBUG_DRAIN
                *(_context->gDumpFile) << "in flight = " << _measured_in_flight_flits[c].size() << endl;
#endif
                return true;
            }
        }
    }
    return false;
}

void TrafficManager::_ClearStats( )
{
    _slowest_flit.assign(_classes, -1);
    _slowest_packet.assign(_classes, -1);

    for ( int c = 0; c < _classes; ++c ) {

        _plat_stats[c]->Clear( );
        _nlat_stats[c]->Clear( );
        _flat_stats[c]->Clear( );

        _frag_stats[c]->Clear( );

        _sent_packets[c].assign(_nodes, 0);
        _accepted_packets[c].assign(_nodes, 0);
        _sent_flits[c].assign(_nodes, 0);
        _accepted_flits[c].assign(_nodes, 0);

#ifdef TRACK_STALLS
        _buffer_busy_stalls[c].assign(_subnets*_routers, 0);
        _buffer_conflict_stalls[c].assign(_subnets*_routers, 0);
        _buffer_full_stalls[c].assign(_subnets*_routers, 0);
        _buffer_reserved_stalls[c].assign(_subnets*_routers, 0);
        _crossbar_conflict_stalls[c].assign(_subnets*_routers, 0);
#endif
        if(_pair_stats){
            for ( int i = 0; i < _nodes; ++i ) {
                for ( int j = 0; j < _nodes; ++j ) {
                    _pair_plat[c][i*_nodes+j]->Clear( );
                    _pair_nlat[c][i*_nodes+j]->Clear( );
                    _pair_flat[c][i*_nodes+j]->Clear( );
                }
            }
        }
        _hop_stats[c]->Clear();

    }

    _reset_time = _clock.time();
}

void TrafficManager::_ComputeStats( const vector<int> & stats, int *sum, int *min, int *max, int *min_pos, int *max_pos ) const 
{
    int const count = stats.size();
    assert(count > 0);

    if(min_pos) {
        *min_pos = 0;
    }
    if(max_pos) {
        *max_pos = 0;
    }

    if(min) {
        *min = stats[0];
    }
    if(max) {
        *max = stats[0];
    }

    *sum = stats[0];

    for ( int i = 1; i < count; ++i ) {
        int curr = stats[i];
        if ( min  && ( curr < *min ) ) {
            *min = curr;
            if ( min_pos ) {
                *min_pos = i;
            }
        }
        if ( max && ( curr > *max ) ) {
            *max = curr;
            if ( max_pos ) {
                *max_pos = i;
            }
        }
        *sum += curr;
    }
}

void TrafficManager::_DisplayRemaining( ostream & os ) const 
{
    for(int c = 0; c < _classes; ++c) {

        map<int, Flit *>::const_iterator iter;
        int i;

        os << "Class " << c << ":" << endl;

        os << "Remaining flits: ";
        for ( iter = _total_in_flight_flits[c].begin( ), i = 0;
              ( iter != _total_in_flight_flits[c].end( ) ) && ( i < 10 );
              iter++, i++ ) {
            os << iter->first << " ";
        }
        if(_total_in_flight_flits[c].size() > 10)
            os << "[...] ";
    
        os << "(" << _total_in_flight_flits[c].size() << " flits)" << endl;
    
        os << "Measured flits: ";
        for ( iter = _measured_in_flight_flits[c].begin( ), i = 0;
              ( iter != _measured_in_flight_flits[c].end( ) ) && ( i < 10 );
              iter++, i++ ) {
            os << iter->first << " ";
        }
        if(_measured_in_flight_flits[c].size() > 10)
            os << "[...] ";
    
        os << "(" << _measured_in_flight_flits[c].size() << " flits)" << endl;
    
    }
}


bool TrafficManager::_SingleSim( )
{
    // Otherwise, perform a single simulation in the standard way
    int converged = 0;
  
    //once warmed up, we require 3 converging runs to end the simulation 
    vector<double> prev_latency(_classes, 0.0);
    vector<double> prev_accepted(_classes, 0.0);
    bool clear_last = false;
    int total_phases = 0;
    while( ( total_phases < _max_samples ) && 
           ( ( _sim_state != running ) || 
             ( converged < 3 ) ) ) {
    
        if ( clear_last || (( ( _sim_state == warming_up ) && ( ( total_phases % 2 ) == 0 ) )) ) {
            clear_last = false;
            _ClearStats( );
        }
    
    
        for ( int iter = 0; iter < _sample_period; ++iter )
            _Step( );
    
        //*(_context->gDumpFile) << _sim_state << endl;

        UpdateStats();
        DisplayStats();
    
        int lat_exc_class = -1;
        int lat_chg_exc_class = -1;
        int acc_chg_exc_class = -1;
    
        for(int c = 0; c < _classes; ++c) {
      
            if(_measure_stats[c] == 0) {
                continue;
            }

            double cur_latency = _plat_stats[c]->Average( );

            int total_accepted_count;
            _ComputeStats( _accepted_flits[c], &total_accepted_count );
            double total_accepted_rate = (double)total_accepted_count / (double)(_clock.time() - _reset_time);
            double cur_accepted = total_accepted_rate / (double)_nodes;

            double latency_change = fabs((cur_latency - prev_latency[c]) / cur_latency);
            prev_latency[c] = cur_latency;

            double accepted_change = fabs((cur_accepted - prev_accepted[c]) / cur_accepted);
            prev_accepted[c] = cur_accepted;

            double latency = (double)_plat_stats[c]->Sum();
            double count = (double)_plat_stats[c]->NumSamples();
      
            map<int, Flit *>::const_iterator iter;
            for(iter = _total_in_flight_flits[c].begin(); 
                iter != _total_in_flight_flits[c].end(); 
                iter++) {
                latency += (double)(_clock.time() - iter->second->ctime);
                count++;
            }
      
            if((lat_exc_class < 0) &&
               (_latency_thres[c] >= 0.0) &&
               ((latency / count) > _latency_thres[c])) {
                lat_exc_class = c;
            }
      
            *(_context->gDumpFile) << "latency change    = " << latency_change << endl;
            if(lat_chg_exc_class < 0) {
                if((_sim_state == warming_up) &&
                   (_warmup_threshold[c] >= 0.0) &&
                   (latency_change > _warmup_threshold[c])) {
                    lat_chg_exc_class = c;
                } else if((_sim_state == running) &&
                          (_stopping_threshold[c] >= 0.0) &&
                          (latency_change > _stopping_threshold[c])) {
                    lat_chg_exc_class = c;
                }
            }
      
            *(_context->gDumpFile) << "throughput change = " << accepted_change << endl;
            if(acc_chg_exc_class < 0) {
                if((_sim_state == warming_up) &&
                   (_acc_warmup_threshold[c] >= 0.0) &&
                   (accepted_change > _acc_warmup_threshold[c])) {
                    acc_chg_exc_class = c;
                } else if((_sim_state == running) &&
                          (_acc_stopping_threshold[c] >= 0.0) &&
                          (accepted_change > _acc_stopping_threshold[c])) {
                    acc_chg_exc_class = c;
                }
            }
      
        }
    
        // Fail safe for latency mode, throughput will ust continue
        if ( _measure_latency && ( lat_exc_class >= 0 ) ) {
      
            *(_context->gDumpFile) << "Average latency for class " << lat_exc_class << " exceeded " << _latency_thres[lat_exc_class] << " cycles. Aborting simulation." << endl;
            converged = 0; 
            _sim_state = draining;
            _drain_time = _clock.time();
            if(_stats_out) {
                WriteStats(*_stats_out);
            }
            break;
      
        }
    
        if ( _sim_state == warming_up ) {
            if ( ( _warmup_periods > 0 ) ? 
                 ( total_phases + 1 >= _warmup_periods ) :
                 ( ( !_measure_latency || ( lat_chg_exc_class < 0 ) ) &&
                   ( acc_chg_exc_class < 0 ) ) ) {
                *(_context->gDumpFile) << "Warmed up ..." <<  "Time used is " << _clock.time() << " cycles" <<endl;
                clear_last = true;
                _sim_state = running;
            }
        } else if(_sim_state == running) {
            if ( ( !_measure_latency || ( lat_chg_exc_class < 0 ) ) &&
                 ( acc_chg_exc_class < 0 ) ) {
                ++converged;
            } else {
                converged = 0;
            }
        }
        ++total_phases;
    }
  
    if ( _sim_state == running ) {
        ++converged;
    
        _sim_state  = draining;
        _drain_time = _clock.time();

        if ( _measure_latency ) {
            *(_context->gDumpFile) << "Draining all recorded packets ..." << endl;
            int empty_steps = 0;
            while( _PacketsOutstanding( ) ) { 
                _Step( ); 
	
                ++empty_steps;
	
                if ( empty_steps % 1000 == 0 ) {
	  
                    int lat_exc_class = -1;
	  
                    for(int c = 0; c < _classes; c++) {
	    
                        double threshold = _latency_thres[c];
	    
                        if(threshold < 0.0) {
                            continue;
                        }
	    
                        double acc_latency = _plat_stats[c]->Sum();
                        double acc_count = (double)_plat_stats[c]->NumSamples();
	    
                        map<int, Flit *>::const_iterator iter;
                        for(iter = _total_in_flight_flits[c].begin(); 
                            iter != _total_in_flight_flits[c].end(); 
                            iter++) {
                            acc_latency += (double)(_clock.time() - iter->second->ctime);
                            acc_count++;
                        }
	    
                        if((acc_latency / acc_count) > threshold) {
                            lat_exc_class = c;
                            break;
                        }
                    }
	  
                    if(lat_exc_class >= 0) {
                        *(_context->gDumpFile) << "Average latency for class " << lat_exc_class << " exceeded " << _latency_thres[lat_exc_class] << " cycles. Aborting simulation." << endl;
                        converged = 0; 
                        _sim_state = warming_up;
                        if(_stats_out) {
                            WriteStats(*_stats_out);
                        }
                        break;
                    }
	  
                    _DisplayRemaining( ); 
	  
                }
            }
        }
    } else {
        *(_context->gDumpFile) << "Too many sample periods needed to converge" << endl;
    }
  
    return ( converged > 0 );
}



int TrafficManager::RunUserDefined(){

    // Ensure that the traffic pattern is user defined (for all classes)
    bool is_user_defined = true;
    for(int c = 0; c < _classes; ++c) {
        if(!(_traffic_pattern[c]->check_user_defined())) {
            is_user_defined = false;
        }
    }
    assert(is_user_defined);

    // The list of packets that need to be injected is stored
    // in the _traffic_pattern object and are injected by the 
    // _injection_process object (which is called by the _IssurePacket method inside
    // the _Step method).
    
    // Differently from the _SingleSim method, in this case the simulation
    // is carried out until all the packets are drained

    _sim_state = running;

    // clear the stats
    _ClearStats();


    for(int c = 0; c< _classes; c++){
        _traffic_pattern[c]->reset();
        _injection_process[c]->reset();
    }


    bool reached_end = false;
    for( int c = 0; c < _classes; c++){
            reached_end |= _injection_process[c]->reached_end;
    }

    bool packets_left = false;
    for(int c = 0; c < _classes; ++c) {
        packets_left |= !_total_in_flight_flits[c].empty();
    }


    int total_steps(0);

    while((packets_left || !reached_end)){

        // Perform a single simulation step
        _Step();

        // *(_context->gDumpFile) << "Landed packets:" << std::endl;
        // for(int c = 0; c < _classes; c++){
        //     *(_context->gDumpFile) << "Class " << c << ":" << std::endl;
        //     // print all the elements of _landed_packets
        //     for(auto it = _landed_packets[c].begin(); it != _landed_packets[c].end(); ++it){
        //         *(_context->gDumpFile) << "Packet " << it->first << " landed at time " << it->second << std::endl;
        //     }
        // }

        ++total_steps;

        // Print activity
        
        // if(total_steps % 1000 == 0){
        //     _DisplayRemaining();
        // }

        packets_left = false;
        for(int c = 0; c < _classes; ++c) {
            packets_left |= !_total_in_flight_flits[c].empty();
        }
        // also check for outstanding replies
        for (int i=0;i<_nodes;i++) {
            packets_left |= !_repliesPending[i].empty();
        }
        // also check for packets to be still processed
        for(int c = 0; c < _classes; ++c) {
            for(int n = 0; n < _nodes; ++n) {
                packets_left |= !_to_process_packets[c][n].empty();
            }
        }
        //reached_end variable is computed as
        //the logic or of the _reached_end variable of the 
        //traffic pattern for the different classes
        reached_end = false;
        for( int c = 0; c < _classes; c++){
            reached_end |= _injection_process[c]->reached_end;
        }

    }
     //wait until all the credits are drained as well
    while(credit_pool.OutStanding()!=0){
        _Step();
    }

    *(_context->gDumpFile) << "=======================================================" << std::endl;

    UpdateStats();
    DisplayStats();

    // Extract the stats as in the _SingleSim method
    //for the love of god don't ever say "Time taken" anywhere else
    //the power script depend on it
    *(_context->gDumpFile) << "Time elapsed is " << _clock.time() << " cycles" <<endl; 

    if(_stats_out) {
        WriteStats(*_stats_out);
    }
    // _UpdateOverallStats();
    
  
    // DisplayOverallStats();
    // if(_print_csv_results) {
    //     DisplayOverallStatsCSV();
    // }

    if (_context->logger) {
        _context->logger->end_simulation(_clock.time());
    }
  
    // return the total latency of the simulation
    return _clock.time();

}



int TrafficManager::Run( )
{

    for ( int sim = 0; sim < _total_sims; ++sim ) {

        _clock.reset( );

        //remove any pending request from the previous simulations
        _requestsOutstanding.assign(_nodes, 0);
        for (int i=0;i<_nodes;i++) {
            while(!_repliesPending[i].empty()) {
                _repliesPending[i].front()->freeReply(packet_reply_pool);
                _repliesPending[i].pop_front();
            }
        }

        //reset queuetime for all sources
        for ( int s = 0; s < _nodes; ++s ) {
            _qtime[s].assign(_classes, 0);
            _qdrained[s].assign(_classes, false);
        }

        // warm-up ...
        // reset stats, all packets after warmup_time marked
        // converge
        // draing, wait until all packets finish
        _sim_state    = warming_up;
  
        _ClearStats( );

        for(int c = 0; c < _classes; ++c) {
            _traffic_pattern[c]->reset();
            _injection_process[c]->reset();
        }

        if ( !_SingleSim( ) ) {
            *(_context->gDumpFile) << "Simulation unstable, ending ..." << endl;
            return false;
        }

        // Empty any remaining packets
        *(_context->gDumpFile) << "Draining remaining packets ..." << endl;
        _empty_network = true;
        int empty_steps = 0;

        bool packets_left = false;
        for(int c = 0; c < _classes; ++c) {
            packets_left |= !_total_in_flight_flits[c].empty();
        }

        while( packets_left ) { 
            _Step( ); 

            ++empty_steps;

            if ( empty_steps % 1000 == 0 ) {
                _DisplayRemaining( ); 
            }
      
            packets_left = false;
            for(int c = 0; c < _classes; ++c) {
                packets_left |= !_total_in_flight_flits[c].empty();
            }
        }
        //wait until all the credits are drained as well
        while(credit_pool.OutStanding()!=0){
            _Step();
        }
        _empty_network = false;

        //for the love of god don't ever say "Time taken" anywhere else
        //the power script depend on it
        *(_context->gDumpFile) << "Time taken is " << _clock.time() << " cycles" <<endl; 

        if(_stats_out) {
            WriteStats(*_stats_out);
        }
        _UpdateOverallStats();
    }
  
    DisplayOverallStats();
    if(_print_csv_results) {
        DisplayOverallStatsCSV();
    }
  
    return _clock.time();
}

void TrafficManager::_UpdateOverallStats() {
    for ( int c = 0; c < _classes; ++c ) {
    
        if(_measure_stats[c] == 0) {
            continue;
        }
    
        _overall_min_plat[c] += _plat_stats[c]->Min();
        _overall_avg_plat[c] += _plat_stats[c]->Average();
        _overall_max_plat[c] += _plat_stats[c]->Max();
        _overall_min_nlat[c] += _nlat_stats[c]->Min();
        _overall_avg_nlat[c] += _nlat_stats[c]->Average();
        _overall_max_nlat[c] += _nlat_stats[c]->Max();
        _overall_min_flat[c] += _flat_stats[c]->Min();
        _overall_avg_flat[c] += _flat_stats[c]->Average();
        _overall_max_flat[c] += _flat_stats[c]->Max();
    
        _overall_min_frag[c] += _frag_stats[c]->Min();
        _overall_avg_frag[c] += _frag_stats[c]->Average();
        _overall_max_frag[c] += _frag_stats[c]->Max();

        _overall_hop_stats[c] += _hop_stats[c]->Average();

        int count_min, count_sum, count_max;
        double rate_min, rate_sum, rate_max;
        double rate_avg;
        double time_delta = (double)(_drain_time - _reset_time);
        _ComputeStats( _sent_flits[c], &count_sum, &count_min, &count_max );
        rate_min = (double)count_min / time_delta;
        rate_sum = (double)count_sum / time_delta;
        rate_max = (double)count_max / time_delta;
        rate_avg = rate_sum / (double)_nodes;
        _overall_min_sent[c] += rate_min;
        _overall_avg_sent[c] += rate_avg;
        _overall_max_sent[c] += rate_max;
        _ComputeStats( _sent_packets[c], &count_sum, &count_min, &count_max );
        rate_min = (double)count_min / time_delta;
        rate_sum = (double)count_sum / time_delta;
        rate_max = (double)count_max / time_delta;
        rate_avg = rate_sum / (double)_nodes;
        _overall_min_sent_packets[c] += rate_min;
        _overall_avg_sent_packets[c] += rate_avg;
        _overall_max_sent_packets[c] += rate_max;
        _ComputeStats( _accepted_flits[c], &count_sum, &count_min, &count_max );
        rate_min = (double)count_min / time_delta;
        rate_sum = (double)count_sum / time_delta;
        rate_max = (double)count_max / time_delta;
        rate_avg = rate_sum / (double)_nodes;
        _overall_min_accepted[c] += rate_min;
        _overall_avg_accepted[c] += rate_avg;
        _overall_max_accepted[c] += rate_max;
        _ComputeStats( _accepted_packets[c], &count_sum, &count_min, &count_max );
        rate_min = (double)count_min / time_delta;
        rate_sum = (double)count_sum / time_delta;
        rate_max = (double)count_max / time_delta;
        rate_avg = rate_sum / (double)_nodes;
        _overall_min_accepted_packets[c] += rate_min;
        _overall_avg_accepted_packets[c] += rate_avg;
        _overall_max_accepted_packets[c] += rate_max;

#ifdef TRACK_STALLS
        _ComputeStats(_buffer_busy_stalls[c], &count_sum);
        rate_sum = (double)count_sum / time_delta;
        rate_avg = rate_sum / (double)(_subnets*_routers);
        _overall_buffer_busy_stalls[c] += rate_avg;
        _ComputeStats(_buffer_conflict_stalls[c], &count_sum);
        rate_sum = (double)count_sum / time_delta;
        rate_avg = rate_sum / (double)(_subnets*_routers);
        _overall_buffer_conflict_stalls[c] += rate_avg;
        _ComputeStats(_buffer_full_stalls[c], &count_sum);
        rate_sum = (double)count_sum / time_delta;
        rate_avg = rate_sum / (double)(_subnets*_routers);
        _overall_buffer_full_stalls[c] += rate_avg;
        _ComputeStats(_buffer_reserved_stalls[c], &count_sum);
        rate_sum = (double)count_sum / time_delta;
        rate_avg = rate_sum / (double)(_subnets*_routers);
        _overall_buffer_reserved_stalls[c] += rate_avg;
        _ComputeStats(_crossbar_conflict_stalls[c], &count_sum);
        rate_sum = (double)count_sum / time_delta;
        rate_avg = rate_sum / (double)(_subnets*_routers);
        _overall_crossbar_conflict_stalls[c] += rate_avg;
#endif

    }
}

void TrafficManager::WriteStats(ostream & os) const {
  
    os << "%=================================" << endl;

    for(int c = 0; c < _classes; ++c) {
    
        if(_measure_stats[c] == 0) {
            continue;
        }
    
        //c+1 due to matlab array starting at 1
        os << "plat(" << c+1 << ") = " << _plat_stats[c]->Average() << ";" << endl
           << "plat_hist(" << c+1 << ",:) = " << *_plat_stats[c] << ";" << endl
           << "nlat(" << c+1 << ") = " << _nlat_stats[c]->Average() << ";" << endl
           << "nlat_hist(" << c+1 << ",:) = " << *_nlat_stats[c] << ";" << endl
           << "flat(" << c+1 << ") = " << _flat_stats[c]->Average() << ";" << endl
           << "flat_hist(" << c+1 << ",:) = " << *_flat_stats[c] << ";" << endl
           << "frag_hist(" << c+1 << ",:) = " << *_frag_stats[c] << ";" << endl
           << "hops(" << c+1 << ",:) = " << *_hop_stats[c] << ";" << endl;
        if(_pair_stats){
            os<< "pair_sent(" << c+1 << ",:) = [ ";
            for(int i = 0; i < _nodes; ++i) {
                for(int j = 0; j < _nodes; ++j) {
                    os << _pair_plat[c][i*_nodes+j]->NumSamples() << " ";
                }
            }
            os << "];" << endl
               << "pair_plat(" << c+1 << ",:) = [ ";
            for(int i = 0; i < _nodes; ++i) {
                for(int j = 0; j < _nodes; ++j) {
                    os << _pair_plat[c][i*_nodes+j]->Average( ) << " ";
                }
            }
            os << "];" << endl
               << "pair_nlat(" << c+1 << ",:) = [ ";
            for(int i = 0; i < _nodes; ++i) {
                for(int j = 0; j < _nodes; ++j) {
                    os << _pair_nlat[c][i*_nodes+j]->Average( ) << " ";
                }
            }
            os << "];" << endl
               << "pair_flat(" << c+1 << ",:) = [ ";
            for(int i = 0; i < _nodes; ++i) {
                for(int j = 0; j < _nodes; ++j) {
                    os << _pair_flat[c][i*_nodes+j]->Average( ) << " ";
                }
            }
        }

        double time_delta = (double)(_drain_time - _reset_time);

        os << "];" << endl
           << "sent_packets(" << c+1 << ",:) = [ ";
        for ( int d = 0; d < _nodes; ++d ) {
            os << (double)_sent_packets[c][d] / time_delta << " ";
        }
        os << "];" << endl
           << "accepted_packets(" << c+1 << ",:) = [ ";
        for ( int d = 0; d < _nodes; ++d ) {
            os << (double)_accepted_packets[c][d] / time_delta << " ";
        }
        os << "];" << endl
           << "sent_flits(" << c+1 << ",:) = [ ";
        for ( int d = 0; d < _nodes; ++d ) {
            os << (double)_sent_flits[c][d] / time_delta << " ";
        }
        os << "];" << endl
           << "accepted_flits(" << c+1 << ",:) = [ ";
        for ( int d = 0; d < _nodes; ++d ) {
            os << (double)_accepted_flits[c][d] / time_delta << " ";
        }
        os << "];" << endl
           << "sent_packet_size(" << c+1 << ",:) = [ ";
        for ( int d = 0; d < _nodes; ++d ) {
            os << (double)_sent_flits[c][d] / (double)_sent_packets[c][d] << " ";
        }
        os << "];" << endl
           << "accepted_packet_size(" << c+1 << ",:) = [ ";
        for ( int d = 0; d < _nodes; ++d ) {
            os << (double)_accepted_flits[c][d] / (double)_accepted_packets[c][d] << " ";
        }
        os << "];" << endl;
#ifdef TRACK_STALLS
        os << "buffer_busy_stalls(" << c+1 << ",:) = [ ";
        for ( int d = 0; d < _subnets*_routers; ++d ) {
            os << (double)_buffer_busy_stalls[c][d] / time_delta << " ";
        }
        os << "];" << endl
           << "buffer_conflict_stalls(" << c+1 << ",:) = [ ";
        for ( int d = 0; d < _subnets*_routers; ++d ) {
            os << (double)_buffer_conflict_stalls[c][d] / time_delta << " ";
        }
        os << "];" << endl
           << "buffer_full_stalls(" << c+1 << ",:) = [ ";
        for ( int d = 0; d < _subnets*_routers; ++d ) {
            os << (double)_buffer_full_stalls[c][d] / time_delta << " ";
        }
        os << "];" << endl
           << "buffer_reserved_stalls(" << c+1 << ",:) = [ ";
        for ( int d = 0; d < _subnets*_routers; ++d ) {
            os << (double)_buffer_reserved_stalls[c][d] / time_delta << " ";
        }
        os << "];" << endl
           << "crossbar_conflict_stalls(" << c+1 << ",:) = [ ";
        for ( int d = 0; d < _subnets*_routers; ++d ) {
            os << (double)_crossbar_conflict_stalls[c][d] / time_delta << " ";
        }
        os << "];" << endl;
#endif
    }
}

void TrafficManager::UpdateStats() {
#if defined(TRACK_FLOWS) || defined(TRACK_STALLS)
    for(int c = 0; c < _classes; ++c) {
#ifdef TRACK_FLOWS
        {
            char trail_char = (c == _classes - 1) ? '\n' : ',';
            if(_injected_flits_out) *_injected_flits_out << _injected_flits[c] << trail_char;
            _injected_flits[c].assign(_nodes, 0);
            if(_ejected_flits_out) *_ejected_flits_out << _ejected_flits[c] << trail_char;
            _ejected_flits[c].assign(_nodes, 0);
        }
#endif
        for(int subnet = 0; subnet < _subnets; ++subnet) {
#ifdef TRACK_FLOWS
            if(_outstanding_credits_out) *_outstanding_credits_out << _outstanding_credits[c][subnet] << ',';
            if(_stored_flits_out) *_stored_flits_out << vector<int>(_nodes, 0) << ',';
#endif
            for(int router = 0; router < _routers; ++router) {
                Router * const r = _router[subnet][router];
#ifdef TRACK_FLOWS
                char trail_char = 
                    ((router == _routers - 1) && (subnet == _subnets - 1) && (c == _classes - 1)) ? '\n' : ',';
                if(_received_flits_out) *_received_flits_out << r->GetReceivedFlits(c) << trail_char;
                if(_stored_flits_out) *_stored_flits_out << r->GetStoredFlits(c) << trail_char;
                if(_sent_flits_out) *_sent_flits_out << r->GetSentFlits(c) << trail_char;
                if(_outstanding_credits_out) *_outstanding_credits_out << r->GetOutstandingCredits(c) << trail_char;
                if(_active_packets_out) *_active_packets_out << r->GetActivePackets(c) << trail_char;
                r->ResetFlowStats(c);
#endif
#ifdef TRACK_STALLS
                _buffer_busy_stalls[c][subnet*_routers+router] += r->GetBufferBusyStalls(c);
                _buffer_conflict_stalls[c][subnet*_routers+router] += r->GetBufferConflictStalls(c);
                _buffer_full_stalls[c][subnet*_routers+router] += r->GetBufferFullStalls(c);
                _buffer_reserved_stalls[c][subnet*_routers+router] += r->GetBufferReservedStalls(c);
                _crossbar_conflict_stalls[c][subnet*_routers+router] += r->GetCrossbarConflictStalls(c);
                r->ResetStallStats(c);
#endif
            }
        }
    }
#ifdef TRACK_FLOWS
    if(_injected_flits_out) *_injected_flits_out << flush;
    if(_received_flits_out) *_received_flits_out << flush;
    if(_stored_flits_out) *_stored_flits_out << flush;
    if(_sent_flits_out) *_sent_flits_out << flush;
    if(_outstanding_credits_out) *_outstanding_credits_out << flush;
    if(_ejected_flits_out) *_ejected_flits_out << flush;
    if(_active_packets_out) *_active_packets_out << flush;
#endif
#endif

#ifdef TRACK_CREDITS
    for(int s = 0; s < _subnets; ++s) {
        for(int n = 0; n < _nodes; ++n) {
            BufferState const * const bs = _buf_states[n][s];
            for(int v = 0; v < _vcs; ++v) {
                if(_used_credits_out) *_used_credits_out << bs->OccupancyFor(v) << ',';
                if(_free_credits_out) *_free_credits_out << bs->AvailableFor(v) << ',';
                if(_max_credits_out) *_max_credits_out << bs->LimitFor(v) << ',';
            }
        }
        for(int r = 0; r < _routers; ++r) {
            Router const * const rtr = _router[s][r];
            char trail_char = 
                ((r == _routers - 1) && (s == _subnets - 1)) ? '\n' : ',';
            if(_used_credits_out) *_used_credits_out << rtr->UsedCredits() << trail_char;
            if(_free_credits_out) *_free_credits_out << rtr->FreeCredits() << trail_char;
            if(_max_credits_out) *_max_credits_out << rtr->MaxCredits() << trail_char;
        }
    }
    if(_used_credits_out) *_used_credits_out << flush;
    if(_free_credits_out) *_free_credits_out << flush;
    if(_max_credits_out) *_max_credits_out << flush;
#endif

}

void TrafficManager::DisplayStats(ostream & os) const {
  
    for(int c = 0; c < _classes; ++c) {
    
        if(_measure_stats[c] == 0) {
            continue;
        }
    
        *(_context->gDumpFile) << "Class " << c << ":" << endl;
    
        *(_context->gDumpFile) 
            << "Packet latency average = " << _plat_stats[c]->Average() << endl
            << "\tminimum = " << _plat_stats[c]->Min() << endl
            << "\tmaximum = " << _plat_stats[c]->Max() << endl
            << "Network latency average = " << _nlat_stats[c]->Average() << endl
            << "\tminimum = " << _nlat_stats[c]->Min() << endl
            << "\tmaximum = " << _nlat_stats[c]->Max() << endl
            << "Slowest packet = " << _slowest_packet[c] << endl
            << "Flit latency average = " << _flat_stats[c]->Average() << endl
            << "\tminimum = " << _flat_stats[c]->Min() << endl
            << "\tmaximum = " << _flat_stats[c]->Max() << endl
            << "Slowest flit = " << _slowest_flit[c] << endl
            << "Fragmentation average = " << _frag_stats[c]->Average() << endl
            << "\tminimum = " << _frag_stats[c]->Min() << endl
            << "\tmaximum = " << _frag_stats[c]->Max() << endl;
    
        int count_sum, count_min, count_max;
        double rate_sum, rate_min, rate_max;
        double rate_avg;
        int sent_packets, sent_flits, accepted_packets, accepted_flits;
        int min_pos, max_pos;
        double time_delta = (double)(_clock.time() - _reset_time);
        _ComputeStats(_sent_packets[c], &count_sum, &count_min, &count_max, &min_pos, &max_pos);
        rate_sum = (double)count_sum / time_delta;
        rate_min = (double)count_min / time_delta;
        rate_max = (double)count_max / time_delta;
        rate_avg = rate_sum / (double)_nodes;
        sent_packets = count_sum;
        *(_context->gDumpFile) << "Injected packet rate average = " << rate_avg << endl
             << "\tminimum = " << rate_min 
             << " (at node " << min_pos << ")" << endl
             << "\tmaximum = " << rate_max
             << " (at node " << max_pos << ")" << endl;
        _ComputeStats(_accepted_packets[c], &count_sum, &count_min, &count_max, &min_pos, &max_pos);
        rate_sum = (double)count_sum / time_delta;
        rate_min = (double)count_min / time_delta;
        rate_max = (double)count_max / time_delta;
        rate_avg = rate_sum / (double)_nodes;
        accepted_packets = count_sum;
        *(_context->gDumpFile) << "Accepted packet rate average = " << rate_avg << endl
             << "\tminimum = " << rate_min 
             << " (at node " << min_pos << ")" << endl
             << "\tmaximum = " << rate_max
             << " (at node " << max_pos << ")" << endl;
        _ComputeStats(_sent_flits[c], &count_sum, &count_min, &count_max, &min_pos, &max_pos);
        rate_sum = (double)count_sum / time_delta;
        rate_min = (double)count_min / time_delta;
        rate_max = (double)count_max / time_delta;
        rate_avg = rate_sum / (double)_nodes;
        sent_flits = count_sum;
        *(_context->gDumpFile) << "Injected flit rate average = " << rate_avg << endl
             << "\tminimum = " << rate_min 
             << " (at node " << min_pos << ")" << endl
             << "\tmaximum = " << rate_max
             << " (at node " << max_pos << ")" << endl;
        _ComputeStats(_accepted_flits[c], &count_sum, &count_min, &count_max, &min_pos, &max_pos);
        rate_sum = (double)count_sum / time_delta;
        rate_min = (double)count_min / time_delta;
        rate_max = (double)count_max / time_delta;
        rate_avg = rate_sum / (double)_nodes;
        accepted_flits = count_sum;
        *(_context->gDumpFile) << "Accepted flit rate average= " << rate_avg << endl
             << "\tminimum = " << rate_min 
             << " (at node " << min_pos << ")" << endl
             << "\tmaximum = " << rate_max
             << " (at node " << max_pos << ")" << endl;
    
        *(_context->gDumpFile) << "Injected packet length average = " << (double)sent_flits / (double)sent_packets << endl
             << "Accepted packet length average = " << (double)accepted_flits / (double)accepted_packets << endl;

        *(_context->gDumpFile) << "Total in-flight flits = " << _total_in_flight_flits[c].size()
             << " (" << _measured_in_flight_flits[c].size() << " measured)"
             << endl;
    
#ifdef TRACK_STALLS
        _ComputeStats(_buffer_busy_stalls[c], &count_sum);
        rate_sum = (double)count_sum / time_delta;
        rate_avg = rate_sum / (double)(_subnets*_routers);
        os << "Buffer busy stall rate = " << rate_avg << endl;
        _ComputeStats(_buffer_conflict_stalls[c], &count_sum);
        rate_sum = (double)count_sum / time_delta;
        rate_avg = rate_sum / (double)(_subnets*_routers);
        os << "Buffer conflict stall rate = " << rate_avg << endl;
        _ComputeStats(_buffer_full_stalls[c], &count_sum);
        rate_sum = (double)count_sum / time_delta;
        rate_avg = rate_sum / (double)(_subnets*_routers);
        os << "Buffer full stall rate = " << rate_avg << endl;
        _ComputeStats(_buffer_reserved_stalls[c], &count_sum);
        rate_sum = (double)count_sum / time_delta;
        rate_avg = rate_sum / (double)(_subnets*_routers);
        os << "Buffer reserved stall rate = " << rate_avg << endl;
        _ComputeStats(_crossbar_conflict_stalls[c], &count_sum);
        rate_sum = (double)count_sum / time_delta;
        rate_avg = rate_sum / (double)(_subnets*_routers);
        os << "Crossbar conflict stall rate = " << rate_avg << endl;
#endif
    
    }
}

void TrafficManager::DisplayOverallStats( ostream & os ) const {

    os << "====== Overall Traffic Statistics ======" << endl;
    for ( int c = 0; c < _classes; ++c ) {

        if(_measure_stats[c] == 0) {
            continue;
        }

        os << "====== Traffic class " << c << " ======" << endl;
    
        os << "Packet latency average = " << _overall_avg_plat[c] / (double)_total_sims
           << " (" << _total_sims << " samples)" << endl;
        os << "\tminimum = " << _overall_min_plat[c] / (double)_total_sims
           << " (" << _total_sims << " samples)" << endl;
        os << "\tmaximum = " << _overall_max_plat[c] / (double)_total_sims
           << " (" << _total_sims << " samples)" << endl;

        os << "Network latency average = " << _overall_avg_nlat[c] / (double)_total_sims
           << " (" << _total_sims << " samples)" << endl;
        os << "\tminimum = " << _overall_min_nlat[c] / (double)_total_sims
           << " (" << _total_sims << " samples)" << endl;
        os << "\tmaximum = " << _overall_max_nlat[c] / (double)_total_sims
           << " (" << _total_sims << " samples)" << endl;

        os << "Flit latency average = " << _overall_avg_flat[c] / (double)_total_sims
           << " (" << _total_sims << " samples)" << endl;
        os << "\tminimum = " << _overall_min_flat[c] / (double)_total_sims
           << " (" << _total_sims << " samples)" << endl;
        os << "\tmaximum = " << _overall_max_flat[c] / (double)_total_sims
           << " (" << _total_sims << " samples)" << endl;

        os << "Fragmentation average = " << _overall_avg_frag[c] / (double)_total_sims
           << " (" << _total_sims << " samples)" << endl;
        os << "\tminimum = " << _overall_min_frag[c] / (double)_total_sims
           << " (" << _total_sims << " samples)" << endl;
        os << "\tmaximum = " << _overall_max_frag[c] / (double)_total_sims
           << " (" << _total_sims << " samples)" << endl;

        os << "Injected packet rate average = " << _overall_avg_sent_packets[c] / (double)_total_sims
           << " (" << _total_sims << " samples)" << endl;
        os << "\tminimum = " << _overall_min_sent_packets[c] / (double)_total_sims
           << " (" << _total_sims << " samples)" << endl;
        os << "\tmaximum = " << _overall_max_sent_packets[c] / (double)_total_sims
           << " (" << _total_sims << " samples)" << endl;
    
        os << "Accepted packet rate average = " << _overall_avg_accepted_packets[c] / (double)_total_sims
           << " (" << _total_sims << " samples)" << endl;
        os << "\tminimum = " << _overall_min_accepted_packets[c] / (double)_total_sims
           << " (" << _total_sims << " samples)" << endl;
        os << "\tmaximum = " << _overall_max_accepted_packets[c] / (double)_total_sims
           << " (" << _total_sims << " samples)" << endl;

        os << "Injected flit rate average = " << _overall_avg_sent[c] / (double)_total_sims
           << " (" << _total_sims << " samples)" << endl;
        os << "\tminimum = " << _overall_min_sent[c] / (double)_total_sims
           << " (" << _total_sims << " samples)" << endl;
        os << "\tmaximum = " << _overall_max_sent[c] / (double)_total_sims
           << " (" << _total_sims << " samples)" << endl;
    
        os << "Accepted flit rate average = " << _overall_avg_accepted[c] / (double)_total_sims
           << " (" << _total_sims << " samples)" << endl;
        os << "\tminimum = " << _overall_min_accepted[c] / (double)_total_sims
           << " (" << _total_sims << " samples)" << endl;
        os << "\tmaximum = " << _overall_max_accepted[c] / (double)_total_sims
           << " (" << _total_sims << " samples)" << endl;
    
        os << "Injected packet size average = " << _overall_avg_sent[c] / _overall_avg_sent_packets[c]
           << " (" << _total_sims << " samples)" << endl;

        os << "Accepted packet size average = " << _overall_avg_accepted[c] / _overall_avg_accepted_packets[c]
           << " (" << _total_sims << " samples)" << endl;
    
        os << "Hops average = " << _overall_hop_stats[c] / (double)_total_sims
           << " (" << _total_sims << " samples)" << endl;
    
#ifdef TRACK_STALLS
        os << "Buffer busy stall rate = " << (double)_overall_buffer_busy_stalls[c] / (double)_total_sims
           << " (" << _total_sims << " samples)" << endl
           << "Buffer conflict stall rate = " << (double)_overall_buffer_conflict_stalls[c] / (double)_total_sims
           << " (" << _total_sims << " samples)" << endl
           << "Buffer full stall rate = " << (double)_overall_buffer_full_stalls[c] / (double)_total_sims
           << " (" << _total_sims << " samples)" << endl
           << "Buffer reserved stall rate = " << (double)_overall_buffer_reserved_stalls[c] / (double)_total_sims
           << " (" << _total_sims << " samples)" << endl
           << "Crossbar conflict stall rate = " << (double)_overall_crossbar_conflict_stalls[c] / (double)_total_sims
           << " (" << _total_sims << " samples)" << endl;
#endif
    
    }
  
}

string TrafficManager::_OverallStatsCSV(int c) const
{
    ostringstream os;
    os << _traffic[c]
       << ',' << _use_read_write[c]
       << ',' << _load[c]
       << ',' << _overall_min_plat[c] / (double)_total_sims
       << ',' << _overall_avg_plat[c] / (double)_total_sims
       << ',' << _overall_max_plat[c] / (double)_total_sims
       << ',' << _overall_min_nlat[c] / (double)_total_sims
       << ',' << _overall_avg_nlat[c] / (double)_total_sims
       << ',' << _overall_max_nlat[c] / (double)_total_sims
       << ',' << _overall_min_flat[c] / (double)_total_sims
       << ',' << _overall_avg_flat[c] / (double)_total_sims
       << ',' << _overall_max_flat[c] / (double)_total_sims
       << ',' << _overall_min_frag[c] / (double)_total_sims
       << ',' << _overall_avg_frag[c] / (double)_total_sims
       << ',' << _overall_max_frag[c] / (double)_total_sims
       << ',' << _overall_min_sent_packets[c] / (double)_total_sims
       << ',' << _overall_avg_sent_packets[c] / (double)_total_sims
       << ',' << _overall_max_sent_packets[c] / (double)_total_sims
       << ',' << _overall_min_accepted_packets[c] / (double)_total_sims
       << ',' << _overall_avg_accepted_packets[c] / (double)_total_sims
       << ',' << _overall_max_accepted_packets[c] / (double)_total_sims
       << ',' << _overall_min_sent[c] / (double)_total_sims
       << ',' << _overall_avg_sent[c] / (double)_total_sims
       << ',' << _overall_max_sent[c] / (double)_total_sims
       << ',' << _overall_min_accepted[c] / (double)_total_sims
       << ',' << _overall_avg_accepted[c] / (double)_total_sims
       << ',' << _overall_max_accepted[c] / (double)_total_sims
       << ',' << _overall_avg_sent[c] / _overall_avg_sent_packets[c]
       << ',' << _overall_avg_accepted[c] / _overall_avg_accepted_packets[c]
       << ',' << _overall_hop_stats[c] / (double)_total_sims;

#ifdef TRACK_STALLS
    os << ',' << (double)_overall_buffer_busy_stalls[c] / (double)_total_sims
       << ',' << (double)_overall_buffer_conflict_stalls[c] / (double)_total_sims
       << ',' << (double)_overall_buffer_full_stalls[c] / (double)_total_sims
       << ',' << (double)_overall_buffer_reserved_stalls[c] / (double)_total_sims
       << ',' << (double)_overall_crossbar_conflict_stalls[c] / (double)_total_sims;
#endif

    return os.str();
}

void TrafficManager::DisplayOverallStatsCSV(ostream & os) const {
    for(int c = 0; c < _classes; ++c) {
        os << "results:" << c << ',' << _OverallStatsCSV() << endl;
    }
}

//read the watchlist
void TrafficManager::_LoadWatchList(const string & filename){
    ifstream watch_list;
    watch_list.open(filename.c_str());
  
    string line;
    if(watch_list.is_open()) {
        while(!watch_list.eof()) {
            getline(watch_list, line);
            if(line != "") {
                if(line[0] == 'p') {
                    _packets_to_watch.insert(atoi(line.c_str()+1));
                } else {
                    _flits_to_watch.insert(atoi(line.c_str()));
                }
            }
        }
    
    } else {
        error("Unable to open flit watch file: " + filename);
    }
}

int TrafficManager::_GetNextPacketSize(int cl) const
{
    assert(cl >= 0 && cl < _classes);

    if(_traffic_pattern[cl]->check_user_defined()){
        return _traffic_pattern[cl]->cur_packet->size;
    }

    vector<int> const & psize = _packet_size[cl];
    int sizes = psize.size();

    if(sizes == 1) {
        return psize[0];
    }

    vector<int> const & prate = _packet_size_rate[cl];
    int max_val = _packet_size_max_val[cl];

    int pct = randomInt(max_val);

    for(int i = 0; i < (sizes - 1); ++i) {
        int const limit = prate[i];
        if(limit > pct) {
            return psize[i];
        } else {
            pct -= limit;
        }
    }
    assert(prate.back() > pct);
    return psize.back();
}

double TrafficManager::_GetAveragePacketSize(int cl) const
{   // if the user defined traffic is used...
    // loop over the packets of the config data and compute the average size
    if(_traffic_pattern[cl]->check_user_defined()){
        vector<Packet>::const_iterator it = _packets.begin();
        int sum = 0;
        for(; it != _packets.end(); ++it) {
            if(it->cl == cl) {
                sum += it->size;
            }
        }
        return (double)sum / _packets.size();
    }


    vector<int> const & psize = _packet_size[cl];
    int sizes = psize.size();
    if(sizes == 1) {
        return (double)psize[0];
    }
    vector<int> const & prate = _packet_size_rate[cl];
    int sum = 0;
    for(int i = 0; i < sizes; ++i) {
        sum += psize[i] * prate[i];
    }
    return (double)sum / (double)(_packet_size_max_val[cl] + 1);
}
