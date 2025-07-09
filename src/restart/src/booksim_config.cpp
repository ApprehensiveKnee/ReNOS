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

/*booksim_config.cpp
 *
 *Contains all the configurable parameters in a network
 *
 */


#include "booksim_config.hpp"

BookSimConfig::BookSimConfig( )
{ 
  //========================================================
  // Network options
  //========================================================

  // Channel length listing file
  addStrField( "channel_file", "" ) ;

  // Physical sub-networks
  _arch_int["subnets"] = 1;

  //==== Topology options =======================
  addStrField( "topology", "torus" );
  _arch_int["k"] = 8; //network radix
  _arch_int["n"] = 2; //network dimension
  _arch_int["c"] = 1; //concentration
  addStrField( "routing_function", "none" );

  //simulator tries to correclty adjust latency for node/router placement 
  _arch_int["use_noc_latency"] = 1;


  //used for noc latency calcualtion for network with concentration
  _arch_int["x"] = 8; //number of routers in X
  _arch_int["y"] = 8; //number of routers in Y
  _arch_int["xr"] = 1; //number of nodes per router in X only if c>1
  _arch_int["yr"] = 1; //number of nodes per router in Y only if c>1


  _arch_int["link_failures"] = 0; //legacy
  _arch_int["fail_seed"]     = 0; //legacy
  addStrField( "fail_seed", "" ); // workaround to allow special "time" value

  //==== Single-node options ===============================

  _arch_int["in_ports"]  = 5;
  _arch_int["out_ports"] = 5;

  //========================================================
  // Router options
  //========================================================

  //==== General options ===================================

  addStrField( "router", "iq" ); 

  _arch_int["output_delay"] = 0;
  _arch_int["credit_delay"] = 0;
  _arch_float["internal_speedup"] = 1.0;

  //with switch speedup flits requires otuput buffering
  //full output buffer will cancel switch allocation requests
  //default setting is unlimited
  _arch_int["output_buffer_size"] = -1;

  // enable next-hop-output queueing
  _arch_int["noq"] = 0;

  //==== Input-queued ======================================

  // Control of virtual channel speculation
  _arch_int["speculative"] = 0 ;
  _arch_int["spec_check_elig"] = 1 ;
  _arch_int["spec_check_cred"] = 1 ;
  _arch_int["spec_mask_by_reqs"] = 0 ;
  addStrField("spec_sw_allocator", "prio");
  
  _arch_int["num_vcs"]         = 16;  
  _arch_int["vc_buf_size"]     = 8;  //per vc buffer size
  _arch_int["buf_size"]        = -1; //shared buffer size
  addStrField("buffer_policy", "private"); //buffer sharing policy

  _arch_int["private_bufs"] = -1;
  _arch_int["private_buf_size"] = 1;
  addStrField("private_buf_size", "");
  _arch_int["private_buf_start_vc"] = -1;
  addStrField("private_buf_start_vc", "");
  _arch_int["private_buf_end_vc"] = -1;
  addStrField("private_buf_end_vc", "");

  _arch_int["max_held_slots"] = -1;

  _arch_int["feedback_aging_scale"] = 1;
  _arch_int["feedback_offset"] = 0;

  _arch_int["wait_for_tail_credit"] = 0; // reallocate a VC before a tail credit?
  _arch_int["vc_busy_when_full"] = 0; // mark VCs as in use when they have no credit available
  _arch_int["vc_prioritize_empty"] = 0; // prioritize empty VCs over non-empty ones in VC allocation
  _arch_int["vc_priority_donation"] = 0; // allow high-priority flits to donate their priority to low-priority that they are queued up behind
  _arch_int["vc_shuffle_requests"] = 0; // rearrange VC allocator requests to avoid unfairness

  _arch_int["hold_switch_for_packet"] = 0; // hold a switch config for the entire packet

  _arch_int["input_speedup"]     = 1;  // expansion of input ports into crossbar
  _arch_int["output_speedup"]    = 1;  // expansion of output ports into crossbar

  _arch_int["routing_delay"]    = 1;  
  _arch_int["vc_alloc_delay"]   = 1;  
  _arch_int["sw_alloc_delay"]   = 1;  
  _arch_int["st_prepare_delay"] = 0;
  _arch_int["st_final_delay"]   = 1;

  //==== Event-driven =====================================

  _arch_int["vct"] = 0; 

  //==== Allocators ========================================

  addStrField( "vc_allocator", "islip" ); 
  addStrField( "sw_allocator", "islip" ); 
  
  addStrField( "arb_type", "round_robin" );
  
  _arch_int["alloc_iters"] = 1;
  
  //==== Traffic ========================================

  _arch_int["classes"] = 1;

  addStrField( "traffic", "uniform" );

  _arch_int["class_priority"] = 0;
  addStrField("class_priority", ""); // workaraound to allow for vector specification

  _arch_int["perm_seed"] = 0; // seed value for random permuation trafficpattern generator
  addStrField("perm_seed", ""); // workaround to allow special "time" value

  _arch_float["injection_rate"]       = 0.1;
  addStrField("injection_rate", ""); // workaraound to allow for vector specification
  
  _arch_int["injection_rate_uses_flits"] = 0;

  // number of flits per packet
  _arch_int["packet_size"] = 1;
  addStrField("packet_size", ""); // workaraound to allow for vector specification

  // if multiple values are specified per class, set probabilities for each
  _arch_int["packet_size_rate"] = 1;
  addStrField("packet_size_rate", ""); // workaraound to allow for vector specification

  addStrField( "injection_process", "bernoulli" );

  _arch_float["burst_alpha"] = 0.5; // burst interval
  _arch_float["burst_beta"]  = 0.5; // burst length
  _arch_float["burst_r1"] = -1.0; // burst rate

  addStrField( "priority", "none" );  // message priorities

  _arch_int["batch_size"] = 1000;
  _arch_int["batch_count"] = 1;
  _arch_int["max_outstanding_requests"] = 0; // 0 = unlimited

  // Use read/write request reply scheme
  _arch_int["use_read_write"] = 0;
  addStrField("use_read_write", ""); // workaraound to allow for vector specification
  _arch_float["write_fraction"] = 0.5;
  addStrField("write_fraction", "");

  // Control assignment of packets to VCs
  _arch_int["read_request_begin_vc"] = 0;
  _arch_int["read_request_end_vc"] = 5;
  _arch_int["write_request_begin_vc"] = 2;
  _arch_int["write_request_end_vc"] = 7;
  _arch_int["read_reply_begin_vc"] = 8;
  _arch_int["read_reply_end_vc"] = 13;
  _arch_int["write_reply_begin_vc"] = 10;
  _arch_int["write_reply_end_vc"] = 15;

  // Control Injection of Packets into Replicated Networks
  _arch_int["read_request_subnet"] = 0;
  _arch_int["read_reply_subnet"] = 0;
  _arch_int["read_subnet"] = 0;
  _arch_int["write_request_subnet"] = 0;
  _arch_int["write_reply_subnet"] = 0;
  _arch_int["write_subnet"] = 0;

  // Set packet length in flits
  _arch_int["read_request_size"]  = 1;
  addStrField("read_request_size", ""); // workaraound to allow for vector specification
  _arch_int["write_request_size"] = 1;
  addStrField("write_request_size", ""); // workaraound to allow for vector specification
  _arch_int["read_reply_size"]    = 1;
  addStrField("read_reply_size", ""); // workaraound to allow for vector specification
  _arch_int["write_reply_size"]   = 1;
  addStrField("write_reply_size", ""); // workaraound to allow for vector specification

  // Set packet processing times
  _arch_int["read_request_time"]  = 1;
  addStrField("read_request_time", "");
  _arch_int["write_request_time"] = 1;
  addStrField("write_request_time", "");
  _arch_int["read_reply_time"] = 0;
  addStrField("read_reply_time", "");
  _arch_int["write_reply_time"] = 0;
  addStrField("write_reply_time", "");

  //==== Simulation parameters ==========================

  // types:
  //   latency    - average + latency distribution for a particular injection rate
  //   throughput - sustained throughput for a particular injection rate

  addStrField( "sim_type", "latency" );

  _arch_int["warmup_periods"] = 3; // number of samples periods to "warm-up" the simulation

  _arch_int["sample_period"] = 1000; // how long between measurements
  _arch_int["max_samples"]   = 10;   // maximum number of sample periods in a simulation

  // whether or not to measure statistics for a given traffic class
  _arch_int["measure_stats"] = 1;
  addStrField("measure_stats", ""); // workaround to allow for vector specification
  //whether to enable per pair statistics, caution N^2 memory usage
  _arch_int["pair_stats"] = 0;

  // if avg. latency exceeds the threshold, assume unstable
  _arch_float["latency_thres"] = 500.0;
  addStrField("latency_thres", ""); // workaround to allow for vector specification

   // consider warmed up once relative change in latency / throughput between successive iterations is smaller than this
  _arch_float["warmup_thres"] = 0.05;
  addStrField("warmup_thres", ""); // workaround to allow for vector specification
  _arch_float["acc_warmup_thres"] = 0.05;
  addStrField("acc_warmup_thres", ""); // workaround to allow for vector specification

  // consider converged once relative change in latency / throughput between successive iterations is smaller than this
  _arch_float["stopping_thres"] = 0.05;
  addStrField("stopping_thres", ""); // workaround to allow for vector specification
  _arch_float["acc_stopping_thres"] = 0.05;
  addStrField("acc_stopping_thres", ""); // workaround to allow for vector specification

  _arch_int["sim_count"]     = 1;   // number of simulations to perform


  _arch_int["include_queuing"] =1; // non-zero includes source queuing latency

  //  _arch_int["reorder"]         = 0;  // know what you're doing

  //_arch_int["flit_timing"]     = 0;  // know what you're doing
  //_arch_int["split_packets"]   = 0;  // know what you're doing

  _arch_int["seed"]            = 0; //random seed for simulation, e.g. traffic 
  addStrField("seed", ""); // workaround to allow special "time" value

  _arch_int["print_activity"] = 0;

  _arch_int["print_csv_results"] = 0;

  _arch_int["deadlock_warn_timeout"] = 256;

  _arch_int["viewer_trace"] = 0;

  addStrField("watch_file", "");
  
  addStrField("watch_flits", "");
  addStrField("watch_packets", "");
  addStrField("watch_transactions", "");

  addStrField("watch_out", "");

  addStrField("stats_out", "");

  addIntField("logger", 0);

  // ========== Additions for user-defined traffic ==========
  _arch_int["user_defined_traffic"] = 0;
  _arch_int["resort_waiting_queues"] = 1;
  // ========================================================

  //==== PE parameters ==========================

  _arch_float["ANY_comp_cycles"] = 1.; // cycles to process a FLOP [cycle/FLOP]
  _arch_float["FC_comp_cycles"] = 1.; // cycles to process a FLOP [cycle/FLOP]
  _arch_float["CONV_comp_cycles"] = 1.; // cycles to process a FLOP [cycle/FLOP]

  // ============ Reconfiguration ============

  addIntField("reconfiguration", 0);
  addIntField("max_pe_mem", 256000);
  addIntField("threshold_pe_mem", 200000);
  addIntField("flit_size", 64);
  addIntField("reconf_batch_size", 0);
  addFloatField("reconf_cycles", 0.);

  // ============ Reconfiguration ============

#ifdef TRACK_FLOWS
  addStrField("injected_flits_out", "");
  addStrField("received_flits_out", "");
  addStrField("stored_flits_out", "");
  addStrField("sent_flits_out", "");
  addStrField("outstanding_credits_out", "");
  addStrField("ejected_flits_out", "");
  addStrField("active_packets_out", "");
#endif

#ifdef TRACK_CREDITS
  addStrField("used_credits_out", "");
  addStrField("free_credits_out", "");
  addStrField("max_credits_out", "");
#endif

  // batch only -- packet sequence numbers
  addStrField("sent_packets_out", "");
  
  //==================Power model params=====================
  _arch_int["sim_power"] = 0;
  addStrField("power_output_file","pwr_tmp");
  addStrField("tech_file", "");
  _arch_int["channel_width"] = 128;
  _arch_int["channel_sweep"] = 0;

  //==================Network file===========================
  addStrField("network_file","");
}



PowerConfig::PowerConfig( )
{ 

  _arch_int["H_INVD2"] = 0;
  _arch_int["W_INVD2"] = 0;
  _arch_int["H_DFQD1"] = 0;
  _arch_int["W_DFQD1"] = 0;
  _arch_int["H_ND2D1"] = 0;
  _arch_int["W_ND2D1"] = 0;
  _arch_int["H_SRAM"] = 0;
  _arch_int["W_SRAM"] = 0;
  _arch_float["Vdd"] = 0;
  _arch_float["R"] = 0;
  _arch_float["IoffSRAM"] = 0;
  _arch_float["IoffP"] = 0;
  _arch_float["IoffN"] = 0;
  _arch_float["Cg_pwr"] = 0;
  _arch_float["Cd_pwr"] = 0;
  _arch_float["Cgdl"] = 0;
  _arch_float["Cg"] = 0;
  _arch_float["Cd"] = 0;
  _arch_float["LAMBDA"] = 0;
  _arch_float["MetalPitch"] = 0;
  _arch_float["Rw"] = 0;
  _arch_float["Cw_gnd"] = 0;
  _arch_float["Cw_cpl"] = 0;
  _arch_float["wire_length"] = 0;

}
