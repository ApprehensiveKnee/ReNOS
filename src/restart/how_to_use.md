# <span style="color:sandybrown"><b>Restart: an extension of Booksim 2.0, the Cycle accurate NoC simulator to host trace-like simulations.</b></span>

This is a simple extension of Booksim 2.0, the cycle accurate NoC simulator, to include the support for trace-like simulations. 
Booksim itself is an extremely flexible simulator: as the original authors of the code themselves specify, "most of the simulator’s components are designed to be modular so tasks such as adding a new routing algorithm, topology, or router microarchitecture should not require a complete redesign of the code".\
For these reasons, as well as the fact that the components of the simulated NoC are threated at a somewhat higher level of abstraction than other simulators, Booksim represents a good starting point for the main goal of this project.

## <b> 1.  <span style="color:sandybrown">Main objective of the code</b></span>

The primary goal is to implement a NoC simulator to use for processing application-specific workloads, in order to estimante the required resources - both in terms of power consumption and latency - for the correct and opimized execution of said application on a NoC-based architecture. The idea is to use the estimates produced by the simulation as data for optimization of the mapping of the application's tasks to the PEs represented by the nodes of the NoC

## <b> 2. <span style="color:sandybrown">Running the code </span> </b>

###  2.1.  How to compile
The code can be complied with the following comands, in the main directory of the project:
```bash

$ mkdir build
$ cd build
$ cmake ..
$ make

``` 
To clean the build directory, use the following command:
```bash
$ make clean
```

The code runs using a configuration file, which is passed as an argument to the executable. The option to modify the configuration file by specifying different values for the parameters in the command line directly after the name of the configuration file has been tweaked and is not currently available.

To run the code use the following command, after moving into the `bin` directory created by the compilation process:
```bash
$ ./run_restart <configuration_file>
```

### 2.2. Dependencies

Two additional requirements are needed to run the code:

- <a href=https://github.com/catchorg/Catch2>Catch2</a>: a unit testing framework for C++ , which is included in the `test` directory of the project. The library is not needed for the main code to run. The compilation will still give errors if the library is not found: to avoid this, the `CMakeLists.txt` file should be modified to remove the lines that include the testing directory.

- <a href=https://github.com/nlohmann/json>JSON for Modern C++</a>: a JSON, header-only parser for C++ to allow the parsing of the configuration file (differently from the original Booksim, which defined a custom format for the configuration `.txt` file).

Both of these libraries can be installed using the most common package managers, such as `apt` for Ubuntu or `brew` for MacOS.

### <b>2.3. Configuration file</b>

As already mentioned, the format of accepted configuration files has been switched to JSON.
It can be placed in any directory, provided that the path to it is correctly specified in the command line when running the code.
The file should have the following structure:

```json
{
    "arch": {
        ...
    },
    "packets": {
        ...
    }
}   
```

where the `arch` subdocument <em>contains the general parameters for the architecture of the NoC</em>, while the `packets` object contains <em>list of packets that will be sent through the network</em>.
In general, all the parameters that could be specified in the original Booksim configuration file can be specified also in the JSON file, with the same name, by inserting them in `arch` object. For example:

```json

"arch": {
    "topology" : "torus",
    "k" : 3,
    "n" : 2,
    "use_read_write" : 1,
    "routing_function" : "dim_order",
    "num_vcs" : 16,
    "traffic" : "uniform",
    "injection_rate" : 0.15
},
```

The `packets` object should contain a list of packets, <em>described at very high</em> level, each of which is a JSON object with the following structure:

```json
{
    "id" : 0, // the id of the packet
    "src" : 0, // the source node of the packet
    "dst" : 1, // the destination node of the packet
    "size" : 1, // the size of the packet in terms of flits
    "dep" : -1, // the dependency of the packet, i.e. the id of the last packet that must be received before injecting this one
    "type" : 2, // the type of the packet (READ = 1, WRITE = 2)
    "cl" : 0, // the class of the packet
    "pt_required" : 5 // the number of cycles required to process the packet
}
```

For the simulation to use this list of packets, the `use_read_write` parameter must be set to 1 in the `arch` object.
Additionally, the `user_defined_traffic` parameter must be set to 1, which takes care of explicitly setting `traffic:"user_defined"` and `injection_process:"dependent"`. In this case, other parameters such as `injection_rate` are ignored.

```json

"arch": {
    "topology" : "torus",
    "k" : 3,
    "n" : 2,
    "use_read_write" : 1,
    "routing_function" : "dim_order",
    "num_vcs" : 16,
    "user_defined_traffic" : 1,
    "injection_rate" : 1 //ignored
},
```

## <b>3. <span style="color:sandybrown">Original usage of the simulator</b></span>

User guide for the orginal code of Booksim 2.0 can be found <a href=https://srb.iau.ir/Files/booksim_manual.pdf>here</a>. We hereby report some of the most critical information included in the documentation.
First of all, the config file is used for specifying the topology, routing algorithm, flow control, and traffic

Consider for example the following configuration file:
```json
{
    "arch": {
        "topology" : "torus",
        "k" : 3,
        "n" : 2,
        "use_read_write" : 1,
        "routing_function" : "dim_order",
        "num_vcs" : 16,
        "traffic" : "uniform",
        "injection_rate" : 0.15
    },
}
```
In this case, the NoC simulated is a 3x3 torus, with 16 virtual channels, using dimension
order routing, and uniform traffic with an injection rate of 0.15, meaning that, on average, the simulator will inject 0.15 packets per simulation cycle per node. Using the native implementation of Booksim, the default packet size is 1 flit.

Other parameters may be set and are described below, while default values are specified in the `network/src/booksim_config.hpp` file.


### <b> 3.1 Parameters</b>

1. <b> Topology</b>:
    The ```topology``` parameter determines the underlying topology of the network, together with a set of auxiliary parameters that describe its size:
    
    - `k` : the number of routers per dimensions (network radix)
    - `n` : dimensions of the network
    - `c` : network concentration, i.e. the number of nodes sharing a single router (>1 only in networks with concentration, ex. <em>cmesh</em>)

    The channel latency must be configured within the source code of the topology files. By default, all topologies have a channel latency of 1 cycle. Examples of available topologes are : 

    - <em>k</em>-ary <em>n</em>-fly topology &rarr; `fly`
    - <em>k</em>-ary <em>n</em>-mesh topology &rarr; `mesh`
    - <em>k</em>-ary <em>n</em>-cube(torus) topology &rarr; `torus`
    - concentrated <em>k</em>-ary <em>n</em>-mesh topology &rarr; `cmesh`
    - fat tree topology with 3 levels &rarr; `fat tree`
    - flattened butterfly &rarr; `flattened butterfly`
    - a topology based on the paper “Technology-driven, highly-scalable dragonfly topol-
ogy.” ISCA 2008 &rarr; `dragonfly`
    - quad tree topology &rarr; `quad tree`
    - a topology based on an user input file specifying connectivity of nodes and routers &rarr; `anynet`

2. <b>Physical sub-networks</b>:
    The ```physical_subnetworks``` parameter defines the number of physical sub-networks present in the network (defaults to one). All sub-networks receive the same configuration parameters and thus are identical. Traffic sources maintain an injection queue for each sub-network. The packet generation process is unaffected. It enqueues generated packets into the proper sub-network queue according to a division function in the traffic manager. At every cycle, flits at the head of each queue attempt to be injected. Traffic destinations can eject one flit from each sub-network each cycle.

3. <b>Routing algorithms</b>:
    The ```routing_function``` parameter selects a routing algorithm for the topology. Many routing algorithms need multiple virtual channels for deadlock freedom. Many can be found in the `routefunc.cpp` file.

4. <b> Flow control</b>:
    The simulator supports basic virtual-channel flow contro with credit-based backpressure.

    - `num_vcs` : the number of virtual channels per physical channel. Default is 16.
    - `vc_buf_size` : the depth of the virtual channel buffers in flits. Default is 8.
    - `wait_for_tail_credit` : if set to 1, do not reallocate the VC until the tail flit has left th    at VC. This conservative approach prevents a dependency from being formed between two packets sharing the same VC in succession. Default is 0.

5. <b> Router organization</b>: 
    The simulator supports two different router microarchitectures. The IQ(input-queued) router and the event-driven router. The microarchitecture is selected by the `router` parameter. The IQ router is the default.
    Both routers share a small set of options:

    - `credit_delay` : the processing delay for a credit (does not include the wire delay for transmission).

    - `internal_speedup` : arbitray speedup factor for the router over the chanel transmission rate. For example, a speedup 1.5 means that, on average, 1.5 flits can be forwarded by the router in the time required for a single flit to be transmitted across a channel.

    1. <b> Input-queued router</b>:
    The IQ router implements the router described in the book "Principles and Practices of Interconnection Networks" by William Dally and Brian Towles. The following are some parameters specifc for this type of router:
    
    - `input_speedup` : an integer speedup of the input ports in space. A speedup of 2, for example, gives each input two input ports into the crossbar.

    - `output_speedup` : an integer speedup of the output ports in space. Similar to `input speedup`.

    - `routing_delay` : the delay in cycles for the routing computation.

    - `hold_switch_for_packet`

    - `speculative` : enable speculative VC allocation. (i.e., allow switch allocation to occur in
    parallel with VC allocation for header flits).

    - `alloc iters` : for the islip, pim and select allocators, allocation can be improved by per-
    forming multiple iterations of the algorithm;

    - `arb type` : if the VC or switch allocator is a separable input- or output-first allocator, this parameter selects the type of arbiter to use.

    - `sw_allocator`: the type of allocator used for switch allocation. See later for a list of the possible types.

    - `sw_alloc_delay`: the delay (in cycles) of switch allocation.

    - `vc_allocator` : the type of allocator used for virtual-channel-allocation. See later for a list of the possible types.

    - `vc_alloc_delay` : the delay (in cycles) of virtual-channel allocation.

    2. <b> The event-driven router</b>:
    The event-driven router (router = event) is a microarchitecture designed specifically to support a large number of virtual channels (VCs) eﬃciently. Instead of continuously polling the state of the virtual channels, as in the input-queued router, only changes in VC state are tracked. The eﬃciency then comes from the fact that the number of state changes per cycle is constant and independent of the number of VCs.

6. <b> Allocators </b>: many of the allocators used in the simulator are configurable and several allocation algorithms are configuarable:
    
    - `max_size` : maximum-size matching.

    - `islip` : iSLIP separable allocator.

    - `pim` : parallel iterative matching separable allocator.

    - `loa` : lonely output allocator.

    - `wavefron` : wavefront allocator.

    - `separable_input_first`: separable input-firt allocator.

    - `separable_output_first` : separable output-first allocator.

    - `select`: priority-based allocator. Allocation is performed as in iSLIP, but with preference towards higher priority packets. 

7. <b> Traffic </b>:  

    -   `injection_rate`: the rate at which packet are injected into the simulator. For instance, setting `injection_rate = 0.25` means that each source injects a new packet in one out of every four simulator cycles

    - `injection_process`: the type of injection process used in the simulation, either `bernulli` or `on–off` process. The `dependent` injection process has also been added.

    - `const_flits_per_packet`: sets the number of flits per packet when the `use_read_write` option is unset.

    - `use_read_write` : enable the automatic generation of reply packets of the same type of the received packet. In this case, packet sizes are defined by the options `{read|write}_{request|rely}_size` (for every injection process expect `dependent`). Furthermore, the mapping of packet types to VCs can be customized using the `{read|write}_{request|rely}_{begin|end}_vc`.

    - `traffic` : used to specify the traffic patter, a comprensive list can be found directly on the guide. The `user_defined` value have been added to support custom packet lists.


## 4. <b><span style="color:sandybrown"> Compiling for the list of packets</span></b>

For now, the list of packets gets processed in a really simple way: each packet is 
assinged to a queue based on the source node each packet departs from. At each clock
 cycle it is checked wether the dependencies for the packet at the head of the queue 
 for an idle source have been resolved (meaning all the previous packets carring data
  or asking for computation - whose results are necessary for the packet - have succesfully 
  been received and processed). If so, the packets are injected. Of course, this if a very simple implementation,
 that still has room for improvement: in particular, supposed there is a packet buried deep down in the queue
 of a source node, whose head packet is still waiting for its dependence packet to be recieved. This
 front packets blocks stalls the injection of non-dependent packets that come later in the list, but that could
 already be injected in the net to make usage of idle resources, thus degrading the performance.
