// ANL CPAC 2018
// Steve Rangel
// Joe Hollowed
//
// Note to user and/or developers: comments are overrated

#include <cstdlib>
#include <stddef.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <string.h>
#include <stdexcept>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <fstream>
#include <algorithm>    // std::sort
#include <sstream>
#include <omp.h>
#include <sys/stat.h>

// Generic IO
#include "GenericIO.h"
#include "Partition.h"

#include "MurmurHashNeutral2.cpp" 

// Cosmotools
using namespace std;
using namespace gio;
using namespace cosmotk;

typedef struct lc_halo {
  float posvel_a_m[8];
  int rr[2];
  int64_t id;
  unsigned int destination_rank;
} lc_halo;

typedef struct fof_halo {
  float mass;
  int64_t id;
  unsigned int destination_rank;
} fof_halo;

typedef struct sod_halo {
  float mass;
  float radius;
  float conc;
  float conc_err;
  int64_t id;
  unsigned int destination_rank;
} sod_halo;

bool comp_by_lc_dest(const lc_halo &a, const lc_halo &b) {
  return a.destination_rank < b.destination_rank;
}

bool comp_by_fof_dest(const fof_halo &a, const fof_halo &b) {
  return a.destination_rank < b.destination_rank;
}

bool comp_by_sod_dest(const sod_halo &a, const sod_halo &b) {
  return a.destination_rank < b.destination_rank;
}

bool comp_by_fof_id(const fof_halo &a, const fof_halo &b) {
  return a.id < b.id;
}

bool comp_by_sod_id(const sod_halo &a, const sod_halo &b) {
  return a.id < b.id;
}

struct IO_Buffers {
  vector<float> x;
  vector<float> y;
  vector<float> z;
  vector<float> vx;
  vector<float> vy;
  vector<float> vz;
  vector<float> a;
  vector<float> fof_mass;
  vector<float> sod_mass;
  vector<float> sod_radius;
  vector<float> sod_cdelta;
  vector<float> sod_cdelta_err;
  vector<int> rotation;
  vector<int> replication;
  vector<int64_t> id;
  double box_size[3];
  double origin[3];
};

IO_Buffers IOB;

void clear_IO_buffers() {
  IOB.x.clear();
  IOB.y.clear();
  IOB.z.clear();
  IOB.vx.clear();
  IOB.vy.clear();
  IOB.vz.clear();
  IOB.a.clear();
  IOB.fof_mass.clear();
  IOB.sod_mass.clear();
  IOB.sod_radius.clear();
  IOB.sod_cdelta.clear();
  IOB.sod_cdelta_err.clear();
  IOB.rotation.clear();
  IOB.replication.clear();
  IOB.id.clear();
}

inline unsigned int tag_to_rank(int64_t fof_tag, int n_ranks) {
    return MurmurHashNeutral2((void*)(&fof_tag),sizeof(int64_t),0) % n_ranks;
}

void read_lc_file(string file_name) {
  GenericIO GIO(Partition::getComm(),file_name,GenericIO::FileIOPOSIX);
  GIO.openAndReadHeader(GenericIO::MismatchRedistribute);
  size_t num_elems = GIO.readNumElems();
  GIO.readPhysScale(IOB.box_size);
  GIO.readPhysOrigin(IOB.origin);
  IOB.x.resize(num_elems + GIO.requestedExtraSpace()/sizeof(float));
  IOB.y.resize(num_elems + GIO.requestedExtraSpace()/sizeof(float));
  IOB.z.resize(num_elems + GIO.requestedExtraSpace()/sizeof(float));
  IOB.vx.resize(num_elems + GIO.requestedExtraSpace()/sizeof(float));
  IOB.vy.resize(num_elems + GIO.requestedExtraSpace()/sizeof(float));
  IOB.vz.resize(num_elems + GIO.requestedExtraSpace()/sizeof(float));
  IOB.a.resize(num_elems + GIO.requestedExtraSpace()/sizeof(float));
  IOB.fof_mass.resize(num_elems + GIO.requestedExtraSpace()/sizeof(float));
  IOB.rotation.resize(num_elems + GIO.requestedExtraSpace()/sizeof(int));
  IOB.replication.resize(num_elems + GIO.requestedExtraSpace()/sizeof(int));
  IOB.id.resize(num_elems + GIO.requestedExtraSpace()/sizeof(int64_t));
  GIO.addVariable("x", IOB.x, true);
  GIO.addVariable("y", IOB.y, true);
  GIO.addVariable("z", IOB.z, true);
  GIO.addVariable("vx", IOB.vx, true);
  GIO.addVariable("vy", IOB.vy, true);
  GIO.addVariable("vz", IOB.vz, true);
  GIO.addVariable("a", IOB.a, true);
  GIO.addVariable("mass", IOB.fof_mass, true); // lc masses fill fof_mass field
  GIO.addVariable("rotation", IOB.rotation, true);
  GIO.addVariable("replication", IOB.replication, true);
  GIO.addVariable("id", IOB.id, true);
  GIO.readData();
  IOB.x.resize(num_elems);
  IOB.y.resize(num_elems);
  IOB.z.resize(num_elems);
  IOB.vx.resize(num_elems);
  IOB.vy.resize(num_elems);
  IOB.vz.resize(num_elems);
  IOB.a.resize(num_elems);
  IOB.fof_mass.resize(num_elems);
  IOB.rotation.resize(num_elems);
  IOB.replication.resize(num_elems);
  IOB.id.resize(num_elems);
}

void read_fof_file(string file_name) {
  GenericIO GIO(Partition::getComm(),file_name,GenericIO::FileIOPOSIX);
  GIO.openAndReadHeader(GenericIO::MismatchRedistribute);
  size_t num_elems = GIO.readNumElems();
  IOB.id.resize(num_elems + GIO.requestedExtraSpace()/sizeof(int64_t));
  IOB.fof_mass.resize(num_elems + GIO.requestedExtraSpace()/sizeof(float));
  GIO.addVariable("fof_halo_tag", IOB.id, true);
  GIO.addVariable("fof_halo_mass", IOB.fof_mass, true);
  GIO.readData();
  IOB.id.resize(num_elems);
  IOB.fof_mass.resize(num_elems);
}

void read_sod_file(string file_name) {
  GenericIO GIO(Partition::getComm(),file_name,GenericIO::FileIOPOSIX);
  GIO.openAndReadHeader(GenericIO::MismatchRedistribute);
  size_t num_elems = GIO.readNumElems();
  IOB.id.resize(num_elems + GIO.requestedExtraSpace()/sizeof(int64_t));
  IOB.sod_mass.resize(num_elems + GIO.requestedExtraSpace()/sizeof(float));
  IOB.sod_radius.resize(num_elems + GIO.requestedExtraSpace()/sizeof(float));
  GIO.addVariable("fof_halo_tag", IOB.id, true);
  GIO.addVariable("sod_halo_mass", IOB.sod_mass, true);
  GIO.addVariable("sod_halo_radius", IOB.sod_radius, true);
  GIO.addVariable("sod_halo_cdelta", IOB.sod_cdelta, true);
  GIO.addVariable("sod_halo_cdelta_err", IOB.sod_cdelta_err, true);
  GIO.readData();
  IOB.id.resize(num_elems);
  IOB.sod_mass.resize(num_elems);
  IOB.sod_radius.resize(num_elems);
}

void write_lc_file(string file_name) {
    GenericIO GIO(Partition::getComm(), file_name);
    GIO.setNumElems(IOB.id.size());
    GIO.setPhysOrigin(IOB.origin[0]);
    GIO.setPhysScale(IOB.box_size[0]);
    GIO.addVariable("x",IOB.x);
    GIO.addVariable("y",IOB.y);
    GIO.addVariable("z",IOB.z);
    GIO.addVariable("vx",IOB.vx);
    GIO.addVariable("vy",IOB.vy);
    GIO.addVariable("vz",IOB.vz);
    GIO.addVariable("a",IOB.a);
    GIO.addVariable("fof_mass",IOB.fof_mass);
    GIO.addVariable("sod_mass",IOB.sod_mass);
    GIO.addVariable("sod_radius",IOB.sod_radius);
    GIO.addVariable("sod_cdelta",IOB.sod_mass);
    GIO.addVariable("sod_cdelta_err",IOB.sod_mass);
    GIO.addVariable("rotation",IOB.rotation);
    GIO.addVariable("replication",IOB.replication);
    GIO.addVariable("id",IOB.id);
    GIO.write();
}

// assumes fof_halo_recv is sorted by id
float fof_mass_lookup(int64_t tag, vector<fof_halo> &fof_halo_recv) {
  fof_halo fh;
  fh.id = tag;
  vector<fof_halo>::iterator item = lower_bound(fof_halo_recv.begin(),fof_halo_recv.end(),fh,comp_by_fof_id);
  if (item!=fof_halo_recv.end() && !comp_by_fof_id(fh,*item))
    return (*item).mass;
  return -1.0;
}

// assumes sod_halo_recv is sorted by id
float sod_mass_lookup(int64_t tag, vector<sod_halo> &sod_halo_recv) {
  sod_halo sh;
  sh.id = tag;
  vector<sod_halo>::iterator item = lower_bound(sod_halo_recv.begin(),sod_halo_recv.end(),sh,comp_by_sod_id);
  if (item!=sod_halo_recv.end() && !comp_by_sod_id(sh,*item))
    return (*item).mass;
  return -1.0;
}
float sod_radius_lookup(int64_t tag, vector<sod_halo> &sod_halo_recv) {
  sod_halo sh;
  sh.id = tag;
  vector<sod_halo>::iterator item = lower_bound(sod_halo_recv.begin(),sod_halo_recv.end(),sh,comp_by_sod_id);
  if (item!=sod_halo_recv.end() && !comp_by_sod_id(sh,*item))
    return (*item).radius;
  return -1.0;
}
float sod_conc_lookup(int64_t tag, vector<sod_halo> &sod_halo_recv) {
  sod_halo sh;
  sh.id = tag;
  vector<sod_halo>::iterator item = lower_bound(sod_halo_recv.begin(),sod_halo_recv.end(),sh,comp_by_sod_id);
  if (item!=sod_halo_recv.end() && !comp_by_sod_id(sh,*item))
    return (*item).conc;
  return -1.0;
}
float sod_conc_err_lookup(int64_t tag, vector<sod_halo> &sod_halo_recv) {
  sod_halo sh;
  sh.id = tag;
  vector<sod_halo>::iterator item = lower_bound(sod_halo_recv.begin(),sod_halo_recv.end(),sh,comp_by_sod_id);
  if (item!=sod_halo_recv.end() && !comp_by_sod_id(sh,*item))
    return (*item).conc_err;
  return -1.0;
}

int main( int argc, char** argv ) {
  
  MPI_Init( &argc, &argv );
  Partition::initialize();
  GenericIO::setNaturalDefaultPartition();

  int rank, n_ranks;
  rank = Partition::getMyProc();
  n_ranks = Partition::getNumProc();
  
  if(rank==0)
      cout << "Initialized MPI on " << n_ranks << " ranks" << endl;

  string lc_file     = string(argv[1]);
  string fof_file    = string(argv[2]);
  string sod_file    = string(argv[3]);
  string lc_out_file = string(argv[4]);
  if(rank==0){
      cout << "Lightcone at " << lc_file << endl;
      cout << "FOF catalog at " << fof_file << endl;
      cout << "SOD catalog at " << sod_file << endl;
      cout << "Matched lightcone to be output at " << lc_out_file << endl;
  }

  
  if(rank==0)
      cout << "\nPacking LC send vector" << endl;
  read_lc_file(lc_file);                  // read the file
  vector<lc_halo> lc_halo_send;
  vector<int> lc_halo_send_cnt;
  lc_halo_send_cnt.resize(n_ranks,0);
  for (size_t i=0;i<IOB.id.size();++i) {     // pack a vector of structures
    int64_t tag = IOB.id[i];
    if (tag<0)
      tag = (-1*tag) & 0x0000ffffffffffff;
    unsigned int recv_rank = tag_to_rank(tag, n_ranks);
    lc_halo h = { {IOB.x[i],IOB.y[i],IOB.z[i],IOB.vx[i],IOB.vy[i],IOB.vz[i],IOB.a[i],IOB.fof_mass[i]},\
                  {IOB.rotation[i],IOB.replication[i]},\
                  tag, recv_rank};
    lc_halo_send.push_back(h);
    ++lc_halo_send_cnt[recv_rank];        // count the send items
  }                                       // prepare to send by contiguous destinations
  std::sort(lc_halo_send.begin(),lc_halo_send.end(),comp_by_lc_dest);

  clear_IO_buffers();
  MPI_Barrier(Partition::getComm());

  if(rank==0)
      cout << "Packing FOF send vector" << endl;
  read_fof_file(fof_file);                // same thing for the fof file
  vector<fof_halo> fof_halo_send;
  vector<int> fof_halo_send_cnt;
  fof_halo_send_cnt.resize(n_ranks,0);
  for (size_t i=0;i<IOB.id.size();++i) {
    unsigned int recv_rank = tag_to_rank(IOB.id[i], n_ranks);
    fof_halo h = { IOB.fof_mass[i], IOB.id[i], recv_rank };
    fof_halo_send.push_back(h);
    ++fof_halo_send_cnt[recv_rank];
  }
  sort(fof_halo_send.begin(),fof_halo_send.end(),comp_by_fof_dest);
  
  clear_IO_buffers();
  MPI_Barrier(Partition::getComm());
  
  if(rank==0)
      cout << "Packing SOD send vector" << endl;
  read_sod_file(sod_file);                // same thing for the sod file
  vector<sod_halo> sod_halo_send;
  vector<int> sod_halo_send_cnt;
  sod_halo_send_cnt.resize(n_ranks,0);
  for (size_t i=0;i<IOB.id.size();++i) {
    unsigned int recv_rank = tag_to_rank(IOB.id[i], n_ranks);
    sod_halo h = { IOB.sod_mass[i], IOB.sod_radius[i], IOB.sod_cdelta[i], IOB.sod_cdelta_err[i], IOB.id[i], recv_rank };
    sod_halo_send.push_back(h);
    ++sod_halo_send_cnt[recv_rank];
  }
  sort(sod_halo_send.begin(),sod_halo_send.end(),comp_by_sod_dest);
  MPI_Barrier(Partition::getComm());
  
  if(rank==0)
      cout << "Creating MPI LC, FOF, SOD datatypes" << endl;
  // create the MPI types
  MPI_Datatype lc_halo_type;
  {
    MPI_Datatype type[5] = { MPI_FLOAT, MPI_INT, MPI_INT64_T, MPI_UNSIGNED, MPI_UB };
    int blocklen[5] = {8,2,1,1,1};
    MPI_Aint disp[5] = {  offsetof(lc_halo,posvel_a_m),
                          offsetof(lc_halo,rr),
                          offsetof(lc_halo,id),
                          offsetof(lc_halo,destination_rank),
                          sizeof(lc_halo) };
    MPI_Type_struct(5,blocklen,disp,type,&lc_halo_type);
    MPI_Type_commit(&lc_halo_type);
  }

  MPI_Datatype fof_halo_type;
  {
    MPI_Datatype type[4] = { MPI_FLOAT, MPI_INT64_T, MPI_UNSIGNED, MPI_UB };
    int blocklen[4] = {1,1,1,1};
    MPI_Aint disp[4] = {  offsetof(fof_halo,mass),
                          offsetof(fof_halo,id),
                          offsetof(fof_halo,destination_rank),
                          sizeof(fof_halo) };
    MPI_Type_struct(4,blocklen,disp,type,&fof_halo_type);
    MPI_Type_commit(&fof_halo_type);
  }
  
  MPI_Datatype sod_halo_type;
  {
    MPI_Datatype type[7] ={ MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_INT64_T, MPI_UNSIGNED, MPI_UB };
    int blocklen[7] = {1,1,1,1,1,1,1};
    MPI_Aint disp[7] = {  offsetof(sod_halo,mass),
                          offsetof(sod_halo,radius),
                          offsetof(sod_halo,conc),
                          offsetof(sod_halo,conc_err),
                          offsetof(sod_halo,id),
                          offsetof(sod_halo,destination_rank),
                          sizeof(sod_halo) };
    MPI_Type_struct(7,blocklen,disp,type,&sod_halo_type);
    MPI_Type_commit(&sod_halo_type);
  }
  
  if(rank==0)
      cout << "Getting recv counts" << endl;
  // get the receive counts
  vector<int> lc_halo_recv_cnt;
  lc_halo_recv_cnt.resize(n_ranks,0);
  MPI_Alltoall(&lc_halo_send_cnt[0],1,MPI_INT,&lc_halo_recv_cnt[0],1,MPI_INT,Partition::getComm());
  vector<int> fof_halo_recv_cnt;
  fof_halo_recv_cnt.resize(n_ranks,0);
  MPI_Alltoall(&fof_halo_send_cnt[0],1,MPI_INT,&fof_halo_recv_cnt[0],1,MPI_INT,Partition::getComm());
  vector<int> sod_halo_recv_cnt;
  sod_halo_recv_cnt.resize(n_ranks,0);
  MPI_Alltoall(&sod_halo_send_cnt[0],1,MPI_INT,&sod_halo_recv_cnt[0],1,MPI_INT,Partition::getComm());

  // each rank now knows how many items it will receive from every other rank

  if(rank==0)
      cout << "Calculating offsets" << endl;
  // calculate the offsets
  vector<int> lc_halo_send_off;
  lc_halo_send_off.resize(n_ranks,0);
  vector<int> fof_halo_send_off;
  fof_halo_send_off.resize(n_ranks,0);
  vector<int> sod_halo_send_off;
  sod_halo_send_off.resize(n_ranks,0);
  
  vector<int> lc_halo_recv_off;
  lc_halo_recv_off.resize(n_ranks,0);
  vector<int> fof_halo_recv_off;
  fof_halo_recv_off.resize(n_ranks,0);
  vector<int> sod_halo_recv_off;
  sod_halo_recv_off.resize(n_ranks,0);

  lc_halo_send_off[0] = lc_halo_recv_off[0] = 0;
  fof_halo_send_off[0] = fof_halo_recv_off[0] = 0;
  sod_halo_send_off[0] = sod_halo_recv_off[0] = 0;

  for (int i=1; i<n_ranks; ++i) {
    lc_halo_send_off[i] = lc_halo_send_off[i-1] + lc_halo_send_cnt[i-1];
    lc_halo_recv_off[i] = lc_halo_recv_off[i-1] + lc_halo_recv_cnt[i-1];
    fof_halo_send_off[i] = fof_halo_send_off[i-1] + fof_halo_send_cnt[i-1];
    fof_halo_recv_off[i] = fof_halo_recv_off[i-1] + fof_halo_recv_cnt[i-1];
    sod_halo_send_off[i] = sod_halo_send_off[i-1] + sod_halo_send_cnt[i-1];
    sod_halo_recv_off[i] = sod_halo_recv_off[i-1] + sod_halo_recv_cnt[i-1];
  }

  // compute the receive totals to allocate buffers
  int lc_halo_recv_total = 0;
  int fof_halo_recv_total = 0;
  int sod_halo_recv_total = 0;
  for (int i=0; i<n_ranks; ++i) {
    lc_halo_recv_total += lc_halo_recv_cnt[i];
    fof_halo_recv_total += fof_halo_recv_cnt[i];
    sod_halo_recv_total += sod_halo_recv_cnt[i];
  }

  vector<lc_halo> lc_halo_recv;
  lc_halo_recv.resize(lc_halo_recv_total);
  vector<fof_halo> fof_halo_recv;
  fof_halo_recv.resize(fof_halo_recv_total);
  vector<sod_halo> sod_halo_recv;
  sod_halo_recv.resize(sod_halo_recv_total); 
  MPI_Barrier(Partition::getComm());

  if(rank==0)
      cout << "Redistributing from all to all" << endl;
  // send the actual data
  MPI_Alltoallv(&lc_halo_send[0],&lc_halo_send_cnt[0],&lc_halo_send_off[0],lc_halo_type,\
                   &lc_halo_recv[0],&lc_halo_recv_cnt[0],&lc_halo_recv_off[0],lc_halo_type,Partition::getComm());
  MPI_Alltoallv(&fof_halo_send[0],&fof_halo_send_cnt[0],&fof_halo_send_off[0],fof_halo_type,\
                   &fof_halo_recv[0],&fof_halo_recv_cnt[0],&fof_halo_recv_off[0],fof_halo_type,Partition::getComm());
  MPI_Alltoallv(&sod_halo_send[0],&sod_halo_send_cnt[0],&sod_halo_send_off[0],sod_halo_type,\
                   &sod_halo_recv[0],&sod_halo_recv_cnt[0],&sod_halo_recv_off[0],sod_halo_type,Partition::getComm());

  if(rank==0)
      cout << "Sorting FOF, SOD catalogs for binary search" << endl;
  // sort the fof+sod halos by id for binary search
  std::sort(fof_halo_recv.begin(),fof_halo_recv.end(),comp_by_fof_id);
  std::sort(sod_halo_recv.begin(),sod_halo_recv.end(),comp_by_sod_id);

  if(rank==0)
      cout << "Matching FOF and SOD properties to lightcone" << endl;
  clear_IO_buffers();
  for (int i=0; i<lc_halo_recv_total; ++i) {
    IOB.x.push_back(lc_halo_recv[i].posvel_a_m[0]);
    IOB.y.push_back(lc_halo_recv[i].posvel_a_m[1]);
    IOB.z.push_back(lc_halo_recv[i].posvel_a_m[2]);
    IOB.vx.push_back(lc_halo_recv[i].posvel_a_m[3]);
    IOB.vy.push_back(lc_halo_recv[i].posvel_a_m[4]);
    IOB.vz.push_back(lc_halo_recv[i].posvel_a_m[5]);
    IOB.a.push_back(lc_halo_recv[i].posvel_a_m[6]);
    IOB.fof_mass.push_back(fof_mass_lookup(lc_halo_recv[i].id,fof_halo_recv));
    IOB.sod_mass.push_back(sod_mass_lookup(lc_halo_recv[i].id,sod_halo_recv));
    IOB.sod_radius.push_back(sod_radius_lookup(lc_halo_recv[i].id,sod_halo_recv));
    IOB.sod_cdelta.push_back(sod_conc_lookup(lc_halo_recv[i].id,sod_halo_recv));
    IOB.sod_cdelta_err.push_back(sod_conc_err_lookup(lc_halo_recv[i].id,sod_halo_recv));
    IOB.rotation.push_back(lc_halo_recv[i].rr[0]);
    IOB.replication.push_back(lc_halo_recv[i].rr[1]);
    IOB.id.push_back(lc_halo_recv[i].id);
  }

  if(rank==0)
      cout << "Writing new lightcone catalog" << endl;
  MPI_Barrier(Partition::getComm());
  write_lc_file(lc_out_file);

  Partition::finalize();
  MPI_Finalize();
  return 0;
}
