/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: routefunc.hpp
//  Description: header file for the declaration of the routing functions
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
//  Date:  09/10/2024
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <map>
#include <cstdlib>
#include <cassert>


#include "routefunc.hpp"
#include "kncube.hpp"
#include "random_utils.hpp"
#include "misc_utils.hpp"
#include "fattree.hpp"
#include "tree4.hpp"
#include "qtree.hpp"
#include "cmesh.hpp"



// ============================================================
//  QTree: Nearest Common Ancestor
// ===
void qtree_nca(const SimulationContext*c, const Router *r, const tRoutingParameters * p, const Flit *f,
		int in_channel, OutSet* outputs, bool inject)
{
  int vcBegin = 0, vcEnd = p->gNumVCs-1;
  if ( f->type == commType::READ_REQ  || f->type == commType::READ ) {
    vcBegin = p->gReadReqBeginVC;
    vcEnd = p->gReadReqEndVC;
  } else if ( f->type == commType::WRITE_REQ  || f->type == commType::WRITE ) {
    vcBegin = p->gWriteReqBeginVC;
    vcEnd = p->gWriteReqEndVC;
  } else if ( f->type ==  commType::READ_ACK ) {
    vcBegin = p->gReadReplyBeginVC;
    vcEnd = p->gReadReplyEndVC;
  } else if ( f->type ==  commType::WRITE_ACK ) {
    vcBegin = p->gWriteReplyBeginVC;
    vcEnd = p->gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  int out_port;

  if(inject) {

    out_port = -1;

  } else {

    int height = QTree::HeightFromID( r->GetID() );
    int pos    = QTree::PosFromID( r->GetID() );
    
    int dest   = f->dst;
    
    for (int i = height+1; i < c->gN; i++) 
      dest /= c->gK;
    if ( pos == dest / c->gK ) 
      // Route down to child
      out_port = dest % c->gK ; 
    else
      // Route up to parent
      out_port = c->gK;        

  }

  outputs->clear();

  outputs->addRange( out_port, vcBegin, vcEnd );
}

// ============================================================
//  Tree4: Nearest Common Ancestor w/ Adaptive Routing Up
// ===
void tree4_anca( const SimulationContext*c, const Router *r, const tRoutingParameters * p, const Flit *f,
		int in_channel, OutSet* outputs, bool inject)
{
  int vcBegin = 0, vcEnd = p->gNumVCs-1;
  if ( f->type == commType::READ_REQ  || f->type == commType::READ ) {
    vcBegin = p->gReadReqBeginVC;
    vcEnd = p->gReadReqEndVC;
  } else if ( f->type == commType::WRITE_REQ  || f->type == commType::WRITE ) {
    vcBegin = p->gWriteReqBeginVC;
    vcEnd = p->gWriteReqEndVC;
  } else if ( f->type ==  commType::READ_ACK ) {
    vcBegin = p->gReadReplyBeginVC;
    vcEnd = p->gReadReplyEndVC;
  } else if ( f->type ==  commType::WRITE_ACK ) {
    vcBegin = p->gWriteReplyBeginVC;
    vcEnd = p->gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  int range = 1;
  
  int out_port;

  if(inject) {

    out_port = -1;

  } else {

    int dest = f->dst;
    
    const int NPOS = 16;
    
    int rH = r->GetID( ) / NPOS;
    int rP = r->GetID( ) % NPOS;
    
    if ( rH == 0 ) {
      dest /= 16;
      out_port = 2 * dest + randomInt(1);
    } else if ( rH == 1 ) {
      dest /= 4;
      if ( dest / 4 == rP / 2 )
	out_port = dest % 4;
      else {
	out_port = c->gK;
	range = c->gK;
      }
    } else {
      if ( dest/4 == rP )
	out_port = dest % 4;
      else {
	out_port = c->gK;
	range = 2;
      }
    }
    
    //  cout << "Router("<<rH<<","<<rP<<"): id= " << f->id << " dest= " << f->dst << " out_port = "
    //       << out_port << endl;

  }

  outputs->clear( );

  for (int i = 0; i < range; ++i) 
    outputs->addRange( out_port + i, vcBegin, vcEnd );
}

// ============================================================
//  Tree4: Nearest Common Ancestor w/ Random Routing Up
// ===
void tree4_nca( const SimulationContext*c, const Router *r, const tRoutingParameters * p, const Flit *f,
		int in_channel, OutSet* outputs, bool inject)
{
  int vcBegin = 0, vcEnd = p->gNumVCs-1;
  if ( f->type == commType::READ_REQ  || f->type == commType::READ ) {
    vcBegin = p->gReadReqBeginVC;
    vcEnd = p->gReadReqEndVC;
  } else if ( f->type == commType::WRITE_REQ  || f->type == commType::WRITE ) {
    vcBegin = p->gWriteReqBeginVC;
    vcEnd = p->gWriteReqEndVC;
  } else if ( f->type ==  commType::READ_ACK ) {
    vcBegin = p->gReadReplyBeginVC;
    vcEnd = p->gReadReplyEndVC;
  } else if ( f->type ==  commType::WRITE_ACK ) {
    vcBegin = p->gWriteReplyBeginVC;
    vcEnd = p->gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  int out_port;

  if(inject) {

    out_port = -1;

  } else {

    int dest = f->dst;
    
    const int NPOS = 16;
    
    int rH = r->GetID( ) / NPOS;
    int rP = r->GetID( ) % NPOS;
    
    if ( rH == 0 ) {
      dest /= 16;
      out_port = 2 * dest + randomInt(1);
    } else if ( rH == 1 ) {
      dest /= 4;
      if ( dest / 4 == rP / 2 )
	out_port = dest % 4;
      else
	out_port = c->gK + randomInt(c->gK-1);
    } else {
      if ( dest/4 == rP )
	out_port = dest % 4;
      else
	out_port = c->gK + randomInt(1);
    }
    
    //  cout << "Router("<<rH<<","<<rP<<"): id= " << f->id << " dest= " << f->dst << " out_port = "
    //       << out_port << endl;

  }

  outputs->clear( );

  outputs->addRange( out_port, vcBegin, vcEnd );
}

// ============================================================
//  FATTREE: Nearest Common Ancestor w/ Random  Routing Up
// ===
void fattree_nca( const SimulationContext*c, const Router *r, const tRoutingParameters * p, const Flit *f,
		int in_channel, OutSet* outputs, bool inject)
{
  int vcBegin = 0, vcEnd = p->gNumVCs-1;
  if ( f->type == commType::READ_REQ  || f->type == commType::READ ) {
    vcBegin = p->gReadReqBeginVC;
    vcEnd = p->gReadReqEndVC;
  } else if ( f->type == commType::WRITE_REQ  || f->type == commType::WRITE ) {
    vcBegin = p->gWriteReqBeginVC;
    vcEnd = p->gWriteReqEndVC;
  } else if ( f->type ==  commType::READ_ACK ) {
    vcBegin = p->gReadReplyBeginVC;
    vcEnd = p->gReadReplyEndVC;
  } else if ( f->type ==  commType::WRITE_ACK ) {
    vcBegin = p->gWriteReplyBeginVC;
    vcEnd = p->gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  int out_port;

  if(inject) {

    out_port = -1;

  } else {
    
    int dest = f->dst;
    int router_id = r->GetID(); //routers are numbered with smallest at the top level
    int routers_per_level = powi(c->gK, c->gN-1);
    int pos = router_id%routers_per_level;
    int router_depth  = router_id/ routers_per_level; //which level
    int routers_per_neighborhood = powi(c->gK,c->gN-router_depth-1);
    int router_neighborhood = pos/routers_per_neighborhood; //coverage of this tree
    int router_coverage = powi(c->gK, c->gN-router_depth);  //span of the tree from this router
    

    //NCA reached going down
    if(dest <(router_neighborhood+1)* router_coverage && 
       dest >=router_neighborhood* router_coverage){
      //down ports are numbered first

      //ejection
      if(router_depth == c->gN-1){
	out_port = dest%c->gK;
      } else {	
	//find the down port for the destination
	int router_branch_coverage = powi(c->gK, c->gN-(router_depth+1)); 
	out_port = (dest-router_neighborhood* router_coverage)/router_branch_coverage;
      }
    } else {
      //up ports are numbered last
      assert(in_channel<c->gK);//came from a up channel
      out_port = c->gK+randomInt(c->gK-1);
    }
  }  
  outputs->clear( );

  outputs->addRange( out_port, vcBegin, vcEnd );
}

// ============================================================
//  FATTREE: Nearest Common Ancestor w/ Adaptive Routing Up
// ===
void fattree_anca( const SimulationContext*c, const Router *r, const tRoutingParameters * p, const Flit *f,
		int in_channel, OutSet* outputs, bool inject)
{
  int vcBegin = 0, vcEnd = p->gNumVCs-1;
  if ( f->type == commType::READ_REQ  || f->type == commType::READ ) {
    vcBegin = p->gReadReqBeginVC;
    vcEnd = p->gReadReqEndVC;
  } else if ( f->type == commType::WRITE_REQ  || f->type == commType::WRITE ) {
    vcBegin = p->gWriteReqBeginVC;
    vcEnd = p->gWriteReqEndVC;
  } else if ( f->type ==  commType::READ_ACK ) {
    vcBegin = p->gReadReplyBeginVC;
    vcEnd = p->gReadReplyEndVC;
  } else if ( f->type ==  commType::WRITE_ACK ) {
    vcBegin = p->gWriteReplyBeginVC;
    vcEnd = p->gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));


  int out_port;

  if(inject) {

    out_port = -1;

  } else {


    int dest = f->dst;
    int router_id = r->GetID(); //routers are numbered with smallest at the top level
    int routers_per_level = powi(c->gK, c->gN-1);
    int pos = router_id%routers_per_level;
    int router_depth  = router_id/ routers_per_level; //which level
    int routers_per_neighborhood = powi(c->gK,c->gN-router_depth-1);
    int router_neighborhood = pos/routers_per_neighborhood; //coverage of this tree
    int router_coverage = powi(c->gK, c->gN-router_depth);  //span of the tree from this router
    

    //NCA reached going down
    if(dest <(router_neighborhood+1)* router_coverage && 
       dest >=router_neighborhood* router_coverage){
      //down ports are numbered first

      //ejection
      if(router_depth == c->gN-1){
	out_port = dest%c->gK;
      } else {	
	//find the down port for the destination
	int router_branch_coverage = powi(c->gK, c->gN-(router_depth+1)); 
	out_port = (dest-router_neighborhood* router_coverage)/router_branch_coverage;
      }
    } else {
      //up ports are numbered last
      assert(in_channel<c->gK);//came from a up channel
      out_port = c->gK;
      int random1 = randomInt(c->gK-1); // Chose two ports out of the possible at random, compare loads, choose one.
      int random2 = randomInt(c->gK-1);
      if (r->GetUsedCredit(out_port + random1) > r->GetUsedCredit(out_port + random2)){
	out_port = out_port + random2;
      }else{
	out_port =  out_port + random1;
      }
    }
  }  
  outputs->clear( );
  
  outputs->addRange( out_port, vcBegin, vcEnd );
}

// ============================================================
//  Mesh - adatpive XY,YX Routing 
//         pick xy or yx min routing adaptively at the source router
// ===

int dor_next_mesh( const SimulationContext*c, int cur, int dest, bool descending = false );

void adaptive_xy_yx_mesh( const SimulationContext*c, const Router *r, const tRoutingParameters * p, const Flit *f,
		int in_channel, OutSet* outputs, bool inject)
{
  int vcBegin = 0, vcEnd = p->gNumVCs-1;
  if ( f->type == commType::READ_REQ  || f->type == commType::READ ) {
    vcBegin = p->gReadReqBeginVC;
    vcEnd = p->gReadReqEndVC;
  } else if ( f->type == commType::WRITE_REQ  || f->type == commType::WRITE ) {
    vcBegin = p->gWriteReqBeginVC;
    vcEnd = p->gWriteReqEndVC;
  } else if ( f->type ==  commType::READ_ACK ) {
    vcBegin = p->gReadReplyBeginVC;
    vcEnd = p->gReadReplyEndVC;
  } else if ( f->type ==  commType::WRITE_ACK ) {
    vcBegin = p->gWriteReplyBeginVC;
    vcEnd = p->gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  int out_port;

  if(inject) {

    out_port = -1;

  } else if(r->GetID() == f->dst) {

    // at destination router, we don't need to separate VCs by dim order
    out_port = 2*c->gN;

  } else {

    //each class must have at least 2 vcs assigned or else xy_yx will deadlock
    int const available_vcs = (vcEnd - vcBegin + 1) / 2;
    assert(available_vcs > 0);
    
    int out_port_xy = dor_next_mesh( c, r->GetID(), f->dst, false );
    int out_port_yx = dor_next_mesh( c, r->GetID(), f->dst, true );

    // Route order (XY or YX) determined when packet is injected
    //  into the network, adaptively
    bool x_then_y;
    if(in_channel < 2*c->gN){
      x_then_y =  (f->vc < (vcBegin + available_vcs));
    } else {
      int credit_xy = r->GetUsedCredit(out_port_xy);
      int credit_yx = r->GetUsedCredit(out_port_yx);
      if(credit_xy > credit_yx) {
	x_then_y = false;
      } else if(credit_xy < credit_yx) {
	x_then_y = true;
      } else {
	x_then_y = (randomInt(1) > 0);
      }
    }
    
    if(x_then_y) {
      out_port = out_port_xy;
      vcEnd -= available_vcs;
    } else {
      out_port = out_port_yx;
      vcBegin += available_vcs;
    }

  }

  outputs->clear();

  outputs->addRange( out_port , vcBegin, vcEnd );
  
}

void xy_yx_mesh( const SimulationContext*c, const Router *r, const tRoutingParameters * p, const Flit *f,
		int in_channel, OutSet* outputs, bool inject)
{
  int vcBegin = 0, vcEnd = p->gNumVCs-1;
  if ( f->type == commType::READ_REQ  || f->type == commType::READ ) {
    vcBegin = p->gReadReqBeginVC;
    vcEnd = p->gReadReqEndVC;
  } else if ( f->type == commType::WRITE_REQ  || f->type == commType::WRITE ) {
    vcBegin = p->gWriteReqBeginVC;
    vcEnd = p->gWriteReqEndVC;
  } else if ( f->type ==  commType::READ_ACK ) {
    vcBegin = p->gReadReplyBeginVC;
    vcEnd = p->gReadReplyEndVC;
  } else if ( f->type ==  commType::WRITE_ACK ) {
    vcBegin = p->gWriteReplyBeginVC;
    vcEnd = p->gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  int out_port;

  if(inject) {

    out_port = -1;

  } else if(r->GetID() == f->dst) {

    // at destination router, we don't need to separate VCs by dim order
    out_port = 2*c->gN;

  } else {

    //each class must have at least 2 vcs assigned or else xy_yx will deadlock
    int const available_vcs = (vcEnd - vcBegin + 1) / 2;
    assert(available_vcs > 0);

    // Route order (XY or YX) determined when packet is injected
    //  into the network
    bool x_then_y = ((in_channel < 2*c->gN) ?
		     (f->vc < (vcBegin + available_vcs)) :
		     (randomInt(1) > 0));

    if(x_then_y) {
      out_port = dor_next_mesh( c, r->GetID(), f->dst, false );
      vcEnd -= available_vcs;
    } else {
      out_port = dor_next_mesh( c, r->GetID(), f->dst, true );
      vcBegin += available_vcs;
    }

  }

  outputs->clear();

  outputs->addRange( out_port , vcBegin, vcEnd );
  
}

//
// End Balfour-Schultz
//=============================================================

int dor_next_mesh(const SimulationContext*c, int cur, int dest, bool descending )
{
  if ( cur == dest ) {
    return 2*c->gN;  // Eject
  }

  int dim_left;

  if(descending) {
    for ( dim_left = ( c->gN - 1 ); dim_left > 0; --dim_left ) {
      if ( ( cur * c->gK / c->gNodes ) != ( dest * c->gK / c->gNodes ) ) { break; }
      cur = (cur * c->gK) % c->gNodes; dest = (dest * c->gK) % c->gNodes;
    }
    cur = (cur * c->gK) / c->gNodes;
    dest = (dest * c->gK) / c->gNodes;
  } else {
    for ( dim_left = 0; dim_left < ( c->gN - 1 ); ++dim_left ) {
      if ( ( cur % c->gK ) != ( dest % c->gK ) ) { break; }
      cur /= c->gK; dest /= c->gK;
    }
    cur %= c->gK;
    dest %= c->gK;
  }

  if ( cur < dest ) {
    return 2*dim_left;     // Right
  } else {
    return 2*dim_left + 1; // Left
  }
}

//=============================================================

void dor_next_torus( const SimulationContext*c, int cur, int dest, int in_port,
		     int *out_port, int *partition,
		     bool balance = false )
{
  int dim_left;
  int dir;
  int dist2;

  for ( dim_left = 0; dim_left < c->gN; ++dim_left ) {
    if ( ( cur % c->gK ) != ( dest % c->gK ) ) { break; }
    cur /= c->gK; dest /= c->gK;
  }
  
  if ( dim_left < c->gN ) {

    if ( (in_port/2) != dim_left ) {
      // Turning into a new dimension

      cur %= c->gK; dest %= c->gK;
      dist2 = c->gK - 2 * ( ( dest - cur + c->gK ) % c->gK );
      
      if ( ( dist2 > 0 ) || 
	   ( ( dist2 == 0 ) && ( randomInt( 1 ) ) ) ) {
	*out_port = 2*dim_left;     // Right
	dir = 0;
      } else {
	*out_port = 2*dim_left + 1; // Left
	dir = 1;
      }
      
      if ( partition ) {
	if ( balance ) {
	  // Cray's "Partition" allocation
	  // Two datelines: one between k-1 and 0 which forces VC 1
	  //                another between ((k-1)/2) and ((k-1)/2 + 1) which 
	  //                forces VC 0 otherwise any VC can be used
	  
	  if ( ( ( dir == 0 ) && ( cur > dest ) ) ||
	       ( ( dir == 1 ) && ( cur < dest ) ) ) {
	    *partition = 1;
	  } else if ( ( ( dir == 0 ) && ( cur <= (c->gK-1)/2 ) && ( dest >  (c->gK-1)/2 ) ) ||
		      ( ( dir == 1 ) && ( cur >  (c->gK-1)/2 ) && ( dest <= (c->gK-1)/2 ) ) ) {
	    *partition = 0;
	  } else {
	    *partition = randomInt( 1 ); // use either VC set
	  }
	} else {
	  // Deterministic, fixed dateline between nodes k-1 and 0
	  
	  if ( ( ( dir == 0 ) && ( cur > dest ) ) ||
	       ( ( dir == 1 ) && ( dest < cur ) ) ) {
	    *partition = 1;
	  } else {
	    *partition = 0;
	  }
	}
      }
    } else {
      // Inverting the least significant bit keeps
      // the packet moving in the same direction
      *out_port = in_port ^ 0x1;
    }    

  } else {
    *out_port = 2*c->gN;  // Eject
  }
}

//=============================================================

void dim_order_mesh( const SimulationContext*c, const Router *r, const tRoutingParameters * p, const Flit *f, int in_channel, OutSet *outputs, bool inject )
{
  int out_port = inject ? -1 : dor_next_mesh( c, r->GetID(), f->dst );
  
  int vcBegin = 0, vcEnd = p->gNumVCs-1;
  if ( f->type == commType::READ_REQ  || f->type == commType::READ ) {
    vcBegin = p->gReadReqBeginVC;
    vcEnd = p->gReadReqEndVC;
  } else if ( f->type == commType::WRITE_REQ  || f->type == commType::WRITE ) {
    vcBegin = p->gWriteReqBeginVC;
    vcEnd = p->gWriteReqEndVC;
  } else if ( f->type ==  commType::READ_ACK ) {
    vcBegin = p->gReadReplyBeginVC;
    vcEnd = p->gReadReplyEndVC;
  } else if ( f->type ==  commType::WRITE_ACK ) {
    vcBegin = p->gWriteReplyBeginVC;
    vcEnd = p->gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  if ( !inject && f->watch ) {
    *(c->gWatchOut) <<GetSimTime(c) << " | " << r->getFullName() << " | "
	       << "Adding VC range [" 
	       << vcBegin << "," 
	       << vcEnd << "]"
	       << " at output port " << out_port
	       << " for flit " << f->id
	       << " (input port " << in_channel
	       << ", destination " << f->pid << ")"
	       << "." << std::endl;
  }
  
  outputs->clear();

  outputs->addRange( out_port, vcBegin, vcEnd );
}

//=============================================================

void dim_order_ni_mesh( const SimulationContext*c, const Router *r, const tRoutingParameters * p, const Flit *f, int in_channel, OutSet *outputs, bool inject )
{
  int out_port = inject ? -1 : dor_next_mesh( c, r->GetID(), f->dst );
  
  int vcBegin = 0, vcEnd = p->gNumVCs-1;
  if ( f->type == commType::READ_REQ  || f->type == commType::READ ) {
    vcBegin = p->gReadReqBeginVC;
    vcEnd = p->gReadReqEndVC;
  } else if ( f->type == commType::WRITE_REQ  || f->type == commType::WRITE ) {
    vcBegin = p->gWriteReqBeginVC;
    vcEnd = p->gWriteReqEndVC;
  } else if ( f->type ==  commType::READ_ACK ) {
    vcBegin = p->gReadReplyBeginVC;
    vcEnd = p->gReadReplyEndVC;
  } else if ( f->type ==  commType::WRITE_ACK ) {
    vcBegin = p->gWriteReplyBeginVC;
    vcEnd = p->gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  // at the destination router, we don't need to separate VCs by destination
  if(inject || (r->GetID() != f->dst)) {

    int const vcs_per_dest = (vcEnd - vcBegin + 1) / c->gNodes;
    assert(vcs_per_dest > 0);

    vcBegin += f->dst * vcs_per_dest;
    vcEnd = vcBegin + vcs_per_dest - 1;

  }
  
  if( !inject && f->watch ) {
    *(c->gWatchOut) << GetSimTime(c) << " | " << r->getFullName() << " | "
	       << "Adding VC range [" 
	       << vcBegin << "," 
	       << vcEnd << "]"
	       << " at output port " << out_port
	       << " for flit " << f->id
	       << " (input port " << in_channel
	       << ", destination " << f->dst << ")"
	       << "." << std::endl;
  }
  
  outputs->clear( );
  
  outputs->addRange( out_port, vcBegin, vcEnd );
}

//=============================================================

void dim_order_pni_mesh( const SimulationContext*c, const Router *r, const tRoutingParameters * p, const Flit *f, int in_channel, OutSet *outputs, bool inject )
{
  int out_port = inject ? -1 : dor_next_mesh( c, r->GetID(), f->dst );
  
  int vcBegin = 0, vcEnd = p->gNumVCs-1;
  if ( f->type == commType::READ_REQ  || f->type == commType::READ ) {
    vcBegin = p->gReadReqBeginVC;
    vcEnd = p->gReadReqEndVC;
  } else if ( f->type == commType::WRITE_REQ  || f->type == commType::WRITE ) {
    vcBegin = p->gWriteReqBeginVC;
    vcEnd = p->gWriteReqEndVC;
  } else if ( f->type ==  commType::READ_ACK ) {
    vcBegin = p->gReadReplyBeginVC;
    vcEnd = p->gReadReplyEndVC;
  } else if ( f->type ==  commType::WRITE_ACK ) {
    vcBegin = p->gWriteReplyBeginVC;
    vcEnd = p->gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  if(inject || (r->GetID() != f->dst)) {
    int next_coord = f->dst;
    if(!inject) {
      int out_dim = out_port / 2;
      for(int d = 0; d < out_dim; ++d) {
	next_coord /= c->gK;
      }
    }
    next_coord %= c->gK;
    assert(next_coord >= 0 && next_coord < c->gK);
    int vcs_per_dest = (vcEnd - vcBegin + 1) / c->gK;
    assert(vcs_per_dest > 0);
    vcBegin += next_coord * vcs_per_dest;
    vcEnd = vcBegin + vcs_per_dest - 1;
  }

  if( !inject && f->watch ) {
    *(c->gWatchOut) << GetSimTime(c) << " | " << r->getFullName() << " | "
	       << "Adding VC range [" 
	       << vcBegin << "," 
	       << vcEnd << "]"
	       << " at output port " << out_port
	       << " for flit " << f->id
	       << " (input port " << in_channel
	       << ", destination " << f->dst << ")"
	       << "." << std::endl;
  }
  
  outputs->clear( );
  
  outputs->addRange( out_port, vcBegin, vcEnd );
}


//=============================================================

// Random intermediate in the minimal quadrant defined
// by the source and destination
int rand_min_intr_mesh( const SimulationContext*c, int src, int dest )
{
  int dist;

  int intm = 0;
  int offset = 1;

  for ( int n = 0; n < c->gN; ++n ) {
    dist = ( dest % c->gK ) - ( src % c->gK );

    if ( dist > 0 ) {
      intm += offset * ( ( src % c->gK ) + randomInt( dist ) );
    } else {
      intm += offset * ( ( dest % c->gK ) + randomInt( -dist ) );
    }

    offset *= c->gK;
    dest /= c->gK; src /= c->gK;
  }

  return intm;
}

//=============================================================

void romm_mesh( const SimulationContext*c, const Router *r, const tRoutingParameters * p, const Flit *f, int in_channel, OutSet *outputs, bool inject )
{
  int vcBegin = 0, vcEnd = p->gNumVCs-1;
  if ( f->type == commType::READ_REQ || f->type == commType::READ ) {
    vcBegin = p->gReadReqBeginVC;
    vcEnd = p->gReadReqEndVC;
  } else if ( f->type == commType::WRITE_REQ || f->type == commType::WRITE ) {
    vcBegin = p->gWriteReqBeginVC;
    vcEnd = p->gWriteReqEndVC;
  } else if ( f->type ==  commType::READ_ACK ) {
    vcBegin = p->gReadReplyBeginVC;
    vcEnd = p->gReadReplyEndVC;
  } else if ( f->type ==  commType::WRITE_ACK ) {
    vcBegin = p->gWriteReplyBeginVC;
    vcEnd = p->gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  int out_port;

  if(inject) {

    out_port = -1;

  } else {

    if ( in_channel == 2*c->gN ) {
      f->ph   = 0;  // Phase 0
      f->intm = rand_min_intr_mesh( c, f->src, f->dst );
    } 

    if ( ( f->ph == 0 ) && ( r->GetID( ) == f->intm ) ) {
      f->ph = 1; // Go to phase 1
    }

    out_port = dor_next_mesh( c, r->GetID( ), (f->ph == 0) ? f->intm : f->dst );

    // at the destination router, we don't need to separate VCs by phase
    if(r->GetID() != f->dst) {

      //each class must have at least 2 vcs assigned or else valiant valiant will deadlock
      int available_vcs = (vcEnd - vcBegin + 1) / 2;
      assert(available_vcs > 0);

      if(f->ph == 0) {
	vcEnd -= available_vcs;
      } else {
	assert(f->ph == 1);
	vcBegin += available_vcs;
      }
    }

  }

  outputs->clear( );

  outputs->addRange( out_port, vcBegin, vcEnd );
}

//=============================================================

void romm_ni_mesh( const SimulationContext*c, const Router *r, const tRoutingParameters * p, const Flit *f, int in_channel, OutSet *outputs, bool inject )
{
  int vcBegin = 0, vcEnd = p->gNumVCs-1;
  if ( f->type == commType::READ_REQ  || f->type == commType::READ ) {
    vcBegin = p->gReadReqBeginVC;
    vcEnd = p->gReadReqEndVC;
  } else if ( f->type == commType::WRITE_REQ || f->type == commType::WRITE ) {
    vcBegin = p->gWriteReqBeginVC;
    vcEnd = p->gWriteReqEndVC;
  } else if ( f->type ==  commType::READ_ACK ) {
    vcBegin = p->gReadReplyBeginVC;
    vcEnd = p->gReadReplyEndVC;
  } else if ( f->type ==  commType::WRITE_ACK ) {
    vcBegin = p->gWriteReplyBeginVC;
    vcEnd = p->gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  // at the destination router, we don't need to separate VCs by destination
  if(inject || (r->GetID() != f->dst)) {

    int const vcs_per_dest = (vcEnd - vcBegin + 1) / c->gNodes;
    assert(vcs_per_dest > 0);

    vcBegin += f->dst * vcs_per_dest;
    vcEnd = vcBegin + vcs_per_dest - 1;

  }

  int out_port;

  if(inject) {

    out_port = -1;

  } else {

    if ( in_channel == 2*c->gN ) {
      f->ph   = 0;  // Phase 0
      f->intm = rand_min_intr_mesh( c, f->src, f->dst );
    } 

    if ( ( f->ph == 0 ) && ( r->GetID( ) == f->intm ) ) {
      f->ph = 1; // Go to phase 1
    }

    out_port = dor_next_mesh( c, r->GetID( ), (f->ph == 0) ? f->intm : f->dst );

  }

  outputs->clear( );

  outputs->addRange( out_port, vcBegin, vcEnd );
}

//=============================================================

void min_adapt_mesh( const SimulationContext*c, const Router *r, const tRoutingParameters * p, const Flit *f, int in_channel, OutSet *outputs, bool inject )
{
  int vcBegin = 0, vcEnd = p->gNumVCs-1;
  if ( f->type == commType::READ_REQ  || f->type == commType::READ ) {
    vcBegin = p->gReadReqBeginVC;
    vcEnd = p->gReadReqEndVC;
  } else if ( f->type == commType::WRITE_REQ || f->type == commType::WRITE ) {
    vcBegin = p->gWriteReqBeginVC;
    vcEnd = p->gWriteReqEndVC;
  } else if ( f->type ==  commType::READ_ACK ) {
    vcBegin = p->gReadReplyBeginVC;
    vcEnd = p->gReadReplyEndVC;
  } else if ( f->type ==  commType::WRITE_ACK ) {
    vcBegin = p->gWriteReplyBeginVC;
    vcEnd = p->gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  outputs->clear( );
  
  if(inject) {
    // injection can use all VCs
    outputs->addRange(-1, vcBegin, vcEnd);
    return;
  } else if(r->GetID() == f->dst) {
    // ejection can also use all VCs
    outputs->addRange(2*c->gN, vcBegin, vcEnd);
    return;
  }

  int in_vc;

  if ( in_channel == 2*c->gN ) {
    in_vc = vcEnd; // ignore the injection VC
  } else {
    in_vc = f->vc;
  }
  
  // DOR for the escape channel (VC 0), low priority 
  int out_port = dor_next_mesh( c, r->GetID( ), f->dst );    
  outputs->addRange( out_port, 0, vcBegin, vcBegin );
  
  if ( f->watch ) {
      *(c->gWatchOut) << GetSimTime(c) << " | " << r->getFullName() << " | "
		  << "Adding VC range [" 
		  << vcBegin << "," 
		  << vcBegin << "]"
		  << " at output port " << out_port
		  << " for flit " << f->id
		  << " (input port " << in_channel
		  << ", destination " << f->dst << ")"
		  << "." << endl;
   }
  
  if ( in_vc != vcBegin ) { // If not in the escape VC
    // Minimal adaptive for all other channels
    int cur = r->GetID( );
    int dest = f->dst;
    
    for ( int n = 0; n < c->gN; ++n ) {
      if ( ( cur % c->gK ) != ( dest % c->gK ) ) { 
	// Add minimal direction in dimension 'n'
	if ( ( cur % c->gK ) < ( dest % c->gK ) ) { // Right
	  if ( f->watch ) {
	    *(c->gWatchOut) << GetSimTime(c) << " | " << r->getFullName() << " | "
			<< "Adding VC range [" 
		       << (vcBegin+1) << "," 
			<< vcEnd << "]"
			<< " at output port " << 2*n
			<< " with priority " << 1
			<< " for flit " << f->id
			<< " (input port " << in_channel
			<< ", destination " << f->dst << ")"
			<< "." << endl;
	  }
	  outputs->addRange( 2*n, vcBegin+1, vcEnd, 1 ); 
	} else { // Left
	  if ( f->watch ) {
	    *(c->gWatchOut) << GetSimTime(c) << " | " << r->getFullName() << " | "
			<< "Adding VC range [" 
		       << (vcBegin+1) << "," 
			<< vcEnd << "]"
			<< " at output port " << 2*n+1
			<< " with priority " << 1
			<< " for flit " << f->id
			<< " (input port " << in_channel
			<< ", destination " << f->dst << ")"
			<< "." << endl;
	  }
	  outputs->addRange( 2*n + 1, vcBegin+1, vcEnd, 1 ); 
	}
      }
      cur  /= c->gK;
      dest /= c->gK;
    }
  } 
}

//=============================================================

void planar_adapt_mesh( const SimulationContext*c, const Router *r, const tRoutingParameters * p, const Flit *f, int in_channel, OutSet *outputs, bool inject )
{
  int vcBegin = 0, vcEnd = p->gNumVCs-1;
  if ( f->type == commType::READ_REQ  || f->type == commType::READ ) {
    vcBegin = p->gReadReqBeginVC;
    vcEnd = p->gReadReqEndVC;
  } else if ( f->type == commType::WRITE_REQ || f->type == commType::WRITE ) {
    vcBegin = p->gWriteReqBeginVC;
    vcEnd = p->gWriteReqEndVC;
  } else if ( f->type ==  commType::READ_ACK ) {
    vcBegin = p->gReadReplyBeginVC;
    vcEnd = p->gReadReplyEndVC;
  } else if ( f->type ==  commType::WRITE_ACK ) {
    vcBegin = p->gWriteReplyBeginVC;
    vcEnd = p->gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  outputs->clear( );
  
  if(inject) {
    // injection can use all VCs
    outputs->addRange(-1, vcBegin, vcEnd);
    return;
  }

  int cur     = r->GetID( ); 
  int dest    = f->dst;

  if ( cur != dest ) {
   
    int in_vc   = f->vc;
    int vc_mult = (vcEnd - vcBegin + 1) / 3;

    // Find the first unmatched dimension -- except
    // for when we're in the first dimension because
    // of misrouting in the last adaptive plane.
    // In this case, go to the last dimension instead.

    int n;
    for ( n = 0; n < c->gN; ++n ) {
      if ( ( ( cur % c->gK ) != ( dest % c->gK ) ) &&
	   !( ( in_channel/2 == 0 ) &&
	      ( n == 0 ) &&
	      ( in_vc < vcBegin+2*vc_mult ) ) ) {
	break;
      }

      cur  /= c->gK;
      dest /= c->gK;
    }

    assert( n < c->gN );

    if ( f->watch ) {
      *(c->gWatchOut) << GetSimTime(c) << " | " << r->getFullName() << " | "
		  << "PLANAR ADAPTIVE: flit " << f->id 
		  << " in adaptive plane " << n << "." << endl;
    }

    // We're in adaptive plane n

    // Can route productively in d_{i,2}
    bool increase;
    bool fault;
    if ( ( cur % c->gK ) < ( dest % c->gK ) ) { // Increasing
      increase = true;
      if ( !r->IsFaultyOutput( 2*n ) ) {
	outputs->addRange( 2*n, vcBegin+2*vc_mult, vcEnd );
	fault = false;

	if ( f->watch ) {
	  *(c->gWatchOut) << GetSimTime(c) << " | " << r->getFullName() << " | "
		      << "PLANAR ADAPTIVE: increasing in dimension " << n
		      << "." << endl;
	}
      } else {
	fault = true;
      }
    } else { // Decreasing
      increase = false;
      if ( !r->IsFaultyOutput( 2*n + 1 ) ) {
	outputs->addRange( 2*n + 1, vcBegin+2*vc_mult, vcEnd ); 
	fault = false;

	if ( f->watch ) {
	  *(c->gWatchOut) << GetSimTime(c) << " | " << r->getFullName() << " | "
		      << "PLANAR ADAPTIVE: decreasing in dimension " << n
		      << "." << endl;
	}
      } else {
	fault = true;
      }
    }
      
    n = ( n + 1 ) % c->gN;
    cur  /= c->gK;
    dest /= c->gK;
      
    if ( !increase ) {
      vcBegin += vc_mult;
    }
    vcEnd = vcBegin + vc_mult - 1;
      
    int d1_min_c;
    if ( ( cur % c->gK ) < ( dest % c->gK ) ) { // Increasing in d_{i+1}
      d1_min_c = 2*n;
    } else if ( ( cur % c->gK ) != ( dest % c->gK ) ) {  // Decreasing in d_{i+1}
      d1_min_c = 2*n + 1;
    } else {
      d1_min_c = -1;
    }
      
    // do we want to 180?  if so, the last
    // route was a misroute in this dimension,
    // if there is no fault in d_i, just ignore
    // this dimension, otherwise continue to misroute
    if ( d1_min_c == in_channel ) { 
      if ( fault ) {
	d1_min_c = in_channel ^ 1;
      } else {
	d1_min_c = -1;
      }

      if ( f->watch ) {
	*(c->gWatchOut) << GetSimTime(c) << " | " << r->getFullName() << " | "
		    << "PLANAR ADAPTIVE: avoiding 180 in dimension " << n
		    << "." << endl;
      }
    }
      
    if ( d1_min_c != -1 ) {
      if ( !r->IsFaultyOutput( d1_min_c ) ) {
	outputs->addRange( d1_min_c, vcBegin, vcEnd );
      } else if ( fault ) {
	// major problem ... fault in d_i and d_{i+1}
	r->error( "There seem to be faults in d_i and d_{i+1}" );
      }
    } else if ( fault ) { // need to misroute!
      bool atedge;
      if ( cur % c->gK == 0 ) {
	d1_min_c = 2*n;
	atedge = true;
      } else if ( cur % c->gK == c->gK - 1 ) {
	d1_min_c = 2*n + 1;
	atedge = true;
      } else {
	d1_min_c = 2*n + randomInt( 1 ); // random misroute

	if ( d1_min_c  == in_channel ) { // don't 180
	  d1_min_c = in_channel ^ 1;
	}
	atedge = false;
      }
      
      if ( !r->IsFaultyOutput( d1_min_c ) ) {
	outputs->addRange( d1_min_c, vcBegin, vcEnd );
      } else if ( !atedge && !r->IsFaultyOutput( d1_min_c ^ 1 ) ) {
	outputs->addRange( d1_min_c ^ 1, vcBegin, vcEnd );
      } else {
	// major problem ... fault in d_i and d_{i+1}
	r->error( "There seem to be faults in d_i and d_{i+1}" );
      }
    }
  } else {
    outputs->addRange( 2*c->gN, vcBegin, vcEnd ); 
  }
}

//=============================================================
/*
  FIXME: This is broken (note that f->dr is never actually modified).
  Even if it were, this should really use f->ph instead of introducing a single-
  use field.

void limited_adapt_mesh( const Router *r, const Flit *f, int in_channel, OutSet *outputs, bool inject )
{
  outputs->clear( );

  int vcBegin = 0, vcEnd = gNumVCs-1;
  if ( f->type == commType::READ_REQ ) {
    vcBegin = gReadReqBeginVC;
    vcEnd = gReadReqEndVC;
  } else if ( f->type == commType::WRITE_REQ ) {
    vcBegin = gWriteReqBeginVC;
    vcEnd = gWriteReqEndVC;
  } else if ( f->type ==  commType::READ_REP ) {
    vcBegin = gReadReplyBeginVC;
    vcEnd = gReadReplyEndVC;
  } else if ( f->type ==  commType::WRITE_REP ) {
    vcBegin = gWriteReplyBeginVC;
    vcEnd = gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  if ( inject ) {
    outputs->addRange( -1, vcBegin, vcEnd - 1 );
    f->dr = 0; // zero dimension reversals
    return;
  }

  int cur = r->GetID( );
  int dest = f->dst;
  
  if ( cur != dest ) {
    if ( ( f->vc != vcEnd ) && 
	 ( f->dr != vcEnd - 1 ) ) {
      
      for ( int n = 0; n < c->gN; ++n ) {
	if ( ( cur % c->gK ) != ( dest % c->gK ) ) { 
	  int min_port;
	  if ( ( cur % c->gK ) < ( dest % c->gK ) ) { 
	    min_port = 2*n; // Right
	  } else {
	    min_port = 2*n + 1; // Left
	  }
	  
	  // Go in a productive direction with high priority
	  outputs->addRange( min_port, vcBegin, vcEnd - 1, 2 );
	  
	  // Go in the non-productive direction with low priority
	  outputs->addRange( min_port ^ 0x1, vcBegin, vcEnd - 1, 1 );
	} else {
	  // Both directions are non-productive
	  outputs->addRange( 2*n, vcBegin, vcEnd - 1, 1 );
	  outputs->addRange( 2*n+1, vcBegin, vcEnd - 1, 1 );
	}
	
	cur  /= c->gK;
	dest /= c->gK;
      }
      
    } else {
      outputs->addRange( dor_next_mesh( cur, dest ),
			 vcEnd, vcEnd, 0 );
    }
    
  } else { // at destination
    outputs->addRange( 2*c->gN, vcBegin, vcEnd ); 
  }
}
*/
//=============================================================

void valiant_mesh( const SimulationContext*c, const Router *r, const tRoutingParameters * p, const Flit *f, int in_channel, OutSet *outputs, bool inject )
{
  int vcBegin = 0, vcEnd = p->gNumVCs-1;
  if ( f->type == commType::READ_REQ  || f->type == commType::READ ) {
    vcBegin = p->gReadReqBeginVC;
    vcEnd = p->gReadReqEndVC;
  } else if ( f->type == commType::WRITE_REQ || f->type == commType::WRITE ) {
    vcBegin = p->gWriteReqBeginVC;
    vcEnd = p->gWriteReqEndVC;
  } else if ( f->type ==  commType::READ_ACK ) {
    vcBegin = p->gReadReplyBeginVC;
    vcEnd = p->gReadReplyEndVC;
  } else if ( f->type ==  commType::WRITE_ACK ) {
    vcBegin = p->gWriteReplyBeginVC;
    vcEnd = p->gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  int out_port;

  if(inject) {

    out_port = -1;

  } else {

    if ( in_channel == 2*c->gN ) {
      f->ph   = 0;  // Phase 0
      f->intm = randomInt( c->gNodes - 1 );
    }

    if ( ( f->ph == 0 ) && ( r->GetID( ) == f->intm ) ) {
      f->ph = 1; // Go to phase 1
    }

    out_port = dor_next_mesh( c, r->GetID( ), (f->ph == 0) ? f->intm : f->dst );

    // at the destination router, we don't need to separate VCs by phase
    if(r->GetID() != f->dst) {

      //each class must have at least 2 vcs assigned or else valiant valiant will deadlock
      int const available_vcs = (vcEnd - vcBegin + 1) / 2;
      assert(available_vcs > 0);

      if(f->ph == 0) {
	vcEnd -= available_vcs;
      } else {
	assert(f->ph == 1);
	vcBegin += available_vcs;
      }
    }

  }

  outputs->clear( );

  outputs->addRange( out_port, vcBegin, vcEnd );
}

//=============================================================

void valiant_torus( const SimulationContext*c, const Router *r, const tRoutingParameters * p, const Flit *f, int in_channel, OutSet *outputs, bool inject )
{
  int vcBegin = 0, vcEnd = p->gNumVCs-1;
  if ( f->type == commType::READ_REQ  || f->type == commType::READ ) {
    vcBegin = p->gReadReqBeginVC;
    vcEnd = p->gReadReqEndVC;
  } else if ( f->type == commType::WRITE_REQ || f->type == commType::WRITE ) {
    vcBegin = p->gWriteReqBeginVC;
    vcEnd = p->gWriteReqEndVC;
  } else if ( f->type ==  commType::READ_ACK ) {
    vcBegin = p->gReadReplyBeginVC;
    vcEnd = p->gReadReplyEndVC;
  } else if ( f->type ==  commType::WRITE_ACK ) {
    vcBegin = p->gWriteReplyBeginVC;
    vcEnd = p->gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  int out_port;

  if(inject) {

    out_port = -1;

  } else {

    int phase;
    if ( in_channel == 2*c->gN ) {
      phase   = 0;  // Phase 0
      f->intm = randomInt( c->gNodes - 1 );
    } else {
      phase = f->ph / 2;
    }

    if ( ( phase == 0 ) && ( r->GetID( ) == f->intm ) ) {
      phase = 1; // Go to phase 1
      in_channel = 2*c->gN; // ensures correct vc selection at the beginning of phase 2
    }
  
    int ring_part;
    dor_next_torus( c, r->GetID( ), (phase == 0) ? f->intm : f->dst, in_channel,
		    &out_port, &ring_part, false );

    f->ph = 2 * phase + ring_part;

    // at the destination router, we don't need to separate VCs by phase, etc.
    if(r->GetID() != f->dst) {

      int const ring_available_vcs = (vcEnd - vcBegin + 1) / 2;
      assert(ring_available_vcs > 0);

      if(ring_part == 0) {
	vcEnd -= ring_available_vcs;
      } else {
	assert(ring_part == 1);
	vcBegin += ring_available_vcs;
      }

      int const ph_available_vcs = ring_available_vcs / 2;
      assert(ph_available_vcs > 0);

      if(phase == 0) {
	vcEnd -= ph_available_vcs;
      } else {
	assert(phase == 1);
	vcBegin += ph_available_vcs;
      }
    }

  }

  outputs->clear( );

  outputs->addRange( out_port, vcBegin, vcEnd );
}

//=============================================================

void valiant_ni_torus( const SimulationContext*c, const Router *r, const tRoutingParameters * p,  const Flit *f, int in_channel, 
		       OutSet *outputs, bool inject )
{
  int vcBegin = 0, vcEnd = p->gNumVCs-1;
  if ( f->type == commType::READ_REQ || f->type == commType::READ ) {
    vcBegin = p->gReadReqBeginVC;
    vcEnd = p->gReadReqEndVC;
  } else if ( f->type == commType::WRITE_REQ || f->type == commType::WRITE ) {
    vcBegin = p->gWriteReqBeginVC;
    vcEnd = p->gWriteReqEndVC;
  } else if ( f->type ==  commType::READ_ACK ) {
    vcBegin = p->gReadReplyBeginVC;
    vcEnd = p->gReadReplyEndVC;
  } else if ( f->type ==  commType::WRITE_ACK ) {
    vcBegin = p->gWriteReplyBeginVC;
    vcEnd = p->gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  // at the destination router, we don't need to separate VCs by destination
  if(inject || (r->GetID() != f->dst)) {

    int const vcs_per_dest = (vcEnd - vcBegin + 1) / c->gNodes;
    assert(vcs_per_dest > 0);

    vcBegin += f->dst * vcs_per_dest;
    vcEnd = vcBegin + vcs_per_dest - 1;

  }

  int out_port;

  if(inject) {

    out_port = -1;

  } else {

    int phase;
    if ( in_channel == 2*c->gN ) {
      phase   = 0;  // Phase 0
      f->intm = randomInt( c->gNodes - 1 );
    } else {
      phase = f->ph / 2;
    }

    if ( ( f->ph == 0 ) && ( r->GetID( ) == f->intm ) ) {
      f->ph = 1; // Go to phase 1
      in_channel = 2*c->gN; // ensures correct vc selection at the beginning of phase 2
    }
  
    int ring_part;
    dor_next_torus( c, r->GetID( ), (f->ph == 0) ? f->intm : f->dst, in_channel,
		    &out_port, &ring_part, false );

    f->ph = 2 * phase + ring_part;

    // at the destination router, we don't need to separate VCs by phase, etc.
    if(r->GetID() != f->dst) {

      int const ring_available_vcs = (vcEnd - vcBegin + 1) / 2;
      assert(ring_available_vcs > 0);

      if(ring_part == 0) {
	vcEnd -= ring_available_vcs;
      } else {
	assert(ring_part == 1);
	vcBegin += ring_available_vcs;
      }

      int const ph_available_vcs = ring_available_vcs / 2;
      assert(ph_available_vcs > 0);

      if(phase == 0) {
	vcEnd -= ph_available_vcs;
      } else {
	assert(phase == 1);
	vcBegin += ph_available_vcs;
      }
    }

    if (f->watch) {
      *(c->gWatchOut) << GetSimTime(c) << " | " << r->getFullName() << " | "
		 << "Adding VC range [" 
		 << vcBegin << "," 
		 << vcEnd << "]"
		 << " at output port " << out_port
		 << " for flit " << f->id
		 << " (input port " << in_channel
		 << ", destination " << f->dst << ")"
		 << "." << endl;
    }

  }
  
  outputs->clear( );

  outputs->addRange( out_port, vcBegin, vcEnd );
}

//=============================================================

void dim_order_torus( const SimulationContext*c, const Router *r, const tRoutingParameters * p,  const Flit *f, int in_channel, 
		       OutSet *outputs, bool inject )
{
  int vcBegin = 0, vcEnd = p->gNumVCs-1;
  if ( f->type == commType::READ_REQ || f->type == commType::READ ) {
    vcBegin = p->gReadReqBeginVC;
    vcEnd = p->gReadReqEndVC;
  } else if ( f->type == commType::WRITE_REQ || f->type == commType::WRITE ) {
    vcBegin = p->gWriteReqBeginVC;
    vcEnd = p->gWriteReqEndVC;
  } else if ( f->type ==  commType::READ_ACK ) {
    vcBegin = p->gReadReplyBeginVC;
    vcEnd = p->gReadReplyEndVC;
  } else if ( f->type ==  commType::WRITE_ACK ) {
    vcBegin = p->gWriteReplyBeginVC;
    vcEnd = p->gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  int out_port;

  if(inject) {

    out_port = -1;

  } else {
    
    int cur  = r->GetID( );
    int dest = f->dst;

    dor_next_torus( c, cur, dest, in_channel,
		    &out_port, &f->ph, false );


    // at the destination router, we don't need to separate VCs by ring partition
    if(cur != dest) {

      int const available_vcs = (vcEnd - vcBegin + 1) / 2;
      assert(available_vcs > 0);

      if ( f->ph == 0 ) {
	vcEnd -= available_vcs;
      } else {
	vcBegin += available_vcs;
      } 
    }

    if ( f->watch ) {
      *(c->gWatchOut) << GetSimTime(c) << " | " << r->getFullName() << " | "
		 << "Adding VC range [" 
		 << vcBegin << "," 
		 << vcEnd << "]"
		 << " at output port " << out_port
		 << " for flit " << f->id
		 << " (input port " << in_channel
		 << ", destination " << f->dst << ")"
		 << "." << endl;
    }

  }
 
  outputs->clear( );

  outputs->addRange( out_port, vcBegin, vcEnd );
}

//=============================================================

void dim_order_ni_torus( const SimulationContext*c, const Router *r, const tRoutingParameters * p,  const Flit *f, int in_channel, 
		       OutSet *outputs, bool inject )
{
  int vcBegin = 0, vcEnd = p->gNumVCs-1;
  if ( f->type == commType::READ_REQ || f->type == commType::READ ) {
    vcBegin = p->gReadReqBeginVC;
    vcEnd = p->gReadReqEndVC;
  } else if ( f->type == commType::WRITE_REQ || f->type == commType::WRITE ) {
    vcBegin = p->gWriteReqBeginVC;
    vcEnd = p->gWriteReqEndVC;
  } else if ( f->type ==  commType::READ_ACK ) {
    vcBegin = p->gReadReplyBeginVC;
    vcEnd = p->gReadReplyEndVC;
  } else if ( f->type ==  commType::WRITE_ACK ) {
    vcBegin = p->gWriteReplyBeginVC;
    vcEnd = p->gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  int out_port;

  if(inject) {

    out_port = -1;

  } else {
    
    int cur  = r->GetID( );
    int dest = f->dst;

    dor_next_torus( c, cur, dest, in_channel,
		    &out_port, NULL, false );

    // at the destination router, we don't need to separate VCs by destination
    if(cur != dest) {

      int const vcs_per_dest = (vcEnd - vcBegin + 1) / c->gNodes;
      assert(vcs_per_dest);

      vcBegin += f->dst * vcs_per_dest;
      vcEnd = vcBegin + vcs_per_dest - 1;

    }

    if ( f->watch ) {
      *(c->gWatchOut) << GetSimTime(c) << " | " << r->getFullName() << " | "
		 << "Adding VC range [" 
		 << vcBegin << "," 
		 << vcEnd << "]"
		 << " at output port " << out_port
		 << " for flit " << f->id
		 << " (input port " << in_channel
		 << ", destination " << f->dst << ")"
		 << "." << endl;
    }

  }
  
  outputs->clear( );

  outputs->addRange( out_port, vcBegin, vcEnd );
}

//=============================================================

void dim_order_bal_torus( const SimulationContext*c, const Router *r, const tRoutingParameters * p,  const Flit *f, int in_channel, 
		       OutSet *outputs, bool inject )
{
  int vcBegin = 0, vcEnd = p->gNumVCs-1;
  if ( f->type == commType::READ_REQ || f->type == commType::READ ) {
    vcBegin = p->gReadReqBeginVC;
    vcEnd = p->gReadReqEndVC;
  } else if ( f->type == commType::WRITE_REQ || f->type == commType::WRITE ) {
    vcBegin = p->gWriteReqBeginVC;
    vcEnd = p->gWriteReqEndVC;
  } else if ( f->type ==  commType::READ_ACK ) {
    vcBegin = p->gReadReplyBeginVC;
    vcEnd = p->gReadReplyEndVC;
  } else if ( f->type ==  commType::WRITE_ACK ) {
    vcBegin = p->gWriteReplyBeginVC;
    vcEnd = p->gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  int out_port;

  if(inject) {

    out_port = -1;

  } else {

    int cur  = r->GetID( );
    int dest = f->dst;

    dor_next_torus( c, cur, dest, in_channel,
		    &out_port, &f->ph, true );

    // at the destination router, we don't need to separate VCs by ring partition
    if(cur != dest) {

      int const available_vcs = (vcEnd - vcBegin + 1) / 2;
      assert(available_vcs > 0);

      if ( f->ph == 0 ) {
	vcEnd -= available_vcs;
      } else {
	assert(f->ph == 1);
	vcBegin += available_vcs;
      } 
    }

    if ( f->watch ) {
      *(c->gWatchOut) << GetSimTime(c) << " | " << r->getFullName() << " | "
		 << "Adding VC range [" 
		 << vcBegin << "," 
		 << vcEnd << "]"
		 << " at output port " << out_port
		 << " for flit " << f->id
		 << " (input port " << in_channel
		 << ", destination " << f->dst << ")"
		 << "." << endl;
    }

  }
  
  outputs->clear( );

  outputs->addRange( out_port, vcBegin, vcEnd );
}

//=============================================================

void min_adapt_torus( const SimulationContext*c, const Router *r, const tRoutingParameters * p,  const Flit *f, int in_channel, OutSet *outputs, bool inject )
{
  int vcBegin = 0, vcEnd = p->gNumVCs-1;
  if ( f->type == commType::READ_REQ || f->type == commType::READ ) {
    vcBegin = p->gReadReqBeginVC;
    vcEnd = p->gReadReqEndVC;
  } else if ( f->type == commType::WRITE_REQ || f->type == commType::WRITE ) {
    vcBegin = p->gWriteReqBeginVC;
    vcEnd = p->gWriteReqEndVC;
  } else if ( f->type ==  commType::READ_ACK ) {
    vcBegin = p->gReadReplyBeginVC;
    vcEnd = p->gReadReplyEndVC;
  } else if ( f->type ==  commType::WRITE_ACK ) {
    vcBegin = p->gWriteReplyBeginVC;
    vcEnd = p->gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  outputs->clear( );

  if(inject) {
    // injection can use all VCs
    outputs->addRange(-1, vcBegin, vcEnd);
    return;
  } else if(r->GetID() == f->dst) {
    // ejection can also use all VCs
    outputs->addRange(2*c->gN, vcBegin, vcEnd);
  }

  int in_vc;
  if ( in_channel == 2*c->gN ) {
    in_vc = vcEnd; // ignore the injection VC
  } else {
    in_vc = f->vc;
  }
  
  int cur = r->GetID( );
  int dest = f->dst;

  int out_port;

  if ( in_vc > ( vcBegin + 1 ) ) { // If not in the escape VCs
    // Minimal adaptive for all other channels
    
    for ( int n = 0; n < c->gN; ++n ) {
      if ( ( cur % c->gK ) != ( dest % c->gK ) ) {
	int dist2 = c->gK - 2 * ( ( ( dest % c->gK ) - ( cur % c->gK ) + c->gK ) % c->gK );
	
	if ( dist2 > 0 ) { /*) || 
			     ( ( dist2 == 0 ) && ( randomInt( 1 ) ) ) ) {*/
	  outputs->addRange( 2*n, vcBegin+3, vcBegin+3, 1 ); // Right
	} else {
	  outputs->addRange( 2*n + 1, vcBegin+3, vcBegin+3, 1 ); // Left
	}
      }

      cur  /= c->gK;
      dest /= c->gK;
    }
    
    // DOR for the escape channel (VCs 0-1), low priority --- 
    // trick the algorithm with the in channel.  want VC assignment
    // as if we had injected at this node
    dor_next_torus( c, r->GetID( ), f->dst, 2*c->gN,
		    &out_port, &f->ph, false );
  } else {
    // DOR for the escape channel (VCs 0-1), low priority 
    dor_next_torus( c, cur, dest, in_channel,
		    &out_port, &f->ph, false );
  }

  if ( f->ph == 0 ) {
    outputs->addRange( out_port, vcBegin, vcBegin, 0 );
  } else  {
    outputs->addRange( out_port, vcBegin+1, vcBegin+1, 0 );
  } 
}

//=============================================================

void dest_tag_fly( const SimulationContext*c, const Router *r, const tRoutingParameters * p, const Flit *f, int in_channel, 
		   OutSet *outputs, bool inject )
{
  int vcBegin = 0, vcEnd = p->gNumVCs-1;
  if ( f->type == commType::READ_REQ || f->type == commType::READ ) {
    vcBegin = p->gReadReqBeginVC;
    vcEnd = p->gReadReqEndVC;
  } else if ( f->type == commType::WRITE_REQ || f->type == commType::WRITE ) {
    vcBegin = p->gWriteReqBeginVC;
    vcEnd = p->gWriteReqEndVC;
  } else if ( f->type ==  commType::READ_ACK ) {
    vcBegin = p->gReadReplyBeginVC;
    vcEnd = p->gReadReplyEndVC;
  } else if ( f->type ==  commType::WRITE_ACK ) {
    vcBegin = p->gWriteReplyBeginVC;
    vcEnd = p->gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  int out_port;

  if(inject) {

    out_port = -1;

  } else {

    int stage = ( r->GetID( ) * c->gK ) / c->gNodes;
    int dest  = f->dst;

    while( stage < ( c->gN - 1 ) ) {
      dest /= c->gK;
      ++stage;
    }

    out_port = dest % c->gK;
  }

  outputs->clear( );

  outputs->addRange( out_port, vcBegin, vcEnd );
}



//=============================================================

void chaos_torus( const SimulationContext*c, const Router *r, const tRoutingParameters* p, const Flit *f, 
		  int in_channel, OutSet *outputs, bool inject )
{
  outputs->clear( );

  if(inject) {
    outputs->addRange(-1, 0, 0);
    return;
  }

  int cur = r->GetID( );
  int dest = f->dst;
  
  if ( cur != dest ) {
    for ( int n = 0; n < c->gN; ++n ) {

      if ( ( cur % c->gK ) != ( dest % c->gK ) ) { 
	int dist2 = c->gK - 2 * ( ( ( dest % c->gK ) - ( cur % c->gK ) + c->gK ) % c->gK );
      
	if ( dist2 >= 0 ) {
	  outputs->addRange( 2*n, 0, 0 ); // Right
	} 
	
	if ( dist2 <= 0 ) {
	  outputs->addRange( 2*n + 1, 0, 0 ); // Left
	}
      }

      cur  /= c->gK;
      dest /= c->gK;
    }
  } else {
    outputs->addRange( 2*c->gN, 0, 0 ); 
  }
}


//=============================================================

void chaos_mesh( const SimulationContext*c, const Router *r, const tRoutingParameters* p, const Flit *f, 
		  int in_channel, OutSet *outputs, bool inject )
{
  outputs->clear( );

  if(inject) {
    outputs->addRange(-1, 0, 0);
    return;
  }

  int cur = r->GetID( );
  int dest = f->dst;
  
  if ( cur != dest ) {
    for ( int n = 0; n < c->gN; ++n ) {
      if ( ( cur % c->gK ) != ( dest % c->gK ) ) { 
	// Add minimal direction in dimension 'n'
	if ( ( cur % c->gK ) < ( dest % c->gK ) ) { // Right
	  outputs->addRange( 2*n, 0, 0 ); 
	} else { // Left
	  outputs->addRange( 2*n + 1, 0, 0 ); 
	}
      }
      cur  /= c->gK;
      dest /= c->gK;
    }
  } else {
    outputs->addRange( 2*c->gN, 0, 0 ); 
  }
}

//=============================================================

tRoutingParameters initializeRoutingMap( const Configuration & config)
{ 

  // Generate a routing parameter to return
  tRoutingParameters param;

  param.gNumVCs = config.getIntField( "num_vcs" );

  //
  // traffic class partitions
  //
  param.gReadReqBeginVC    = config.getIntField("read_request_begin_vc");
  if(param.gReadReqBeginVC < 0) {
    param.gReadReqBeginVC = 0;
  }
  param.gReadReqEndVC      = config.getIntField("read_request_end_vc");
  if(param.gReadReqEndVC < 0) {
    param.gReadReqEndVC = param.gNumVCs / 2 - 1;
  }
  param.gWriteReqBeginVC   = config.getIntField("write_request_begin_vc");
  if(param.gWriteReqBeginVC < 0) {
    param.gWriteReqBeginVC = 0;
  }
  param.gWriteReqEndVC     = config.getIntField("write_request_end_vc");
  if(param.gWriteReqEndVC < 0) {
    param.gWriteReqEndVC = param.gNumVCs / 2 - 1;
  }
  param.gReadReplyBeginVC  = config.getIntField("read_reply_begin_vc");
  if(param.gReadReplyBeginVC < 0) {
    param.gReadReplyBeginVC = param.gNumVCs / 2;
  }
  param.gReadReplyEndVC    = config.getIntField("read_reply_end_vc");
  if(param.gReadReplyEndVC < 0) {
    param.gReadReplyEndVC = param.gNumVCs - 1;
  }
  param.gWriteReplyBeginVC = config.getIntField("write_reply_begin_vc");
  if(param.gWriteReplyBeginVC < 0) {
    param.gWriteReplyBeginVC = param.gNumVCs / 2;
  }
  param.gWriteReplyEndVC   = config.getIntField("write_reply_end_vc");
  if(param.gWriteReplyEndVC < 0) {
    param.gWriteReplyEndVC = param.gNumVCs - 1;
  }

  /* Register routing functions here */

  // ===================================================
  // Balfour-Schultz
  param.gRoutingFunctionMap["nca_fattree"]         = &fattree_nca;
  param.gRoutingFunctionMap["anca_fattree"]        = &fattree_anca;
  param.gRoutingFunctionMap["nca_qtree"]           = &qtree_nca;
  param.gRoutingFunctionMap["nca_tree4"]           = &tree4_nca;
  param.gRoutingFunctionMap["anca_tree4"]          = &tree4_anca;
  param.gRoutingFunctionMap["dor_mesh"]            = &dim_order_mesh;
  param.gRoutingFunctionMap["xy_yx_mesh"]          = &xy_yx_mesh;
  param.gRoutingFunctionMap["adaptive_xy_yx_mesh"]          = &adaptive_xy_yx_mesh;
  // End Balfour-Schultz
  // ===================================================

  param.gRoutingFunctionMap["dim_order_mesh"]  = &dim_order_mesh;
  param.gRoutingFunctionMap["dim_order_ni_mesh"]  = &dim_order_ni_mesh;
  param.gRoutingFunctionMap["dim_order_pni_mesh"]  = &dim_order_pni_mesh;
  param.gRoutingFunctionMap["dim_order_torus"] = &dim_order_torus;
  param.gRoutingFunctionMap["dim_order_ni_torus"] = &dim_order_ni_torus;
  param.gRoutingFunctionMap["dim_order_bal_torus"] = &dim_order_bal_torus;

  param.gRoutingFunctionMap["romm_mesh"]       = &romm_mesh; 
  param.gRoutingFunctionMap["romm_ni_mesh"]    = &romm_ni_mesh;

  param.gRoutingFunctionMap["min_adapt_mesh"]   = &min_adapt_mesh;
  param.gRoutingFunctionMap["min_adapt_torus"]  = &min_adapt_torus;

  param.gRoutingFunctionMap["planar_adapt_mesh"] = &planar_adapt_mesh;

  // FIXME: This is broken.
  //  gRoutingFunctionMap["limited_adapt_mesh"] = &limited_adapt_mesh;

  param.gRoutingFunctionMap["valiant_mesh"]  = &valiant_mesh;
  param.gRoutingFunctionMap["valiant_torus"] = &valiant_torus;
  param.gRoutingFunctionMap["valiant_ni_torus"] = &valiant_ni_torus;

  param.gRoutingFunctionMap["dest_tag_fly"] = &dest_tag_fly;

  param.gRoutingFunctionMap["chaos_mesh"]  = &chaos_mesh;
  param.gRoutingFunctionMap["chaos_torus"] = &chaos_torus;

  return param;
}

