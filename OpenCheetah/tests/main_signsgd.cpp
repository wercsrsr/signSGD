/*
Authors: Deevashwer Rathee
Copyright:
Copyright (c) 2021 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "BuildingBlocks/aux-protocols.h"
#include "NonLinear/relu-ring.h"
#include <fstream>
#include <iostream>
#include <thread>

using namespace sci;
using namespace std;

#define MAX_THREADS 4

int party, port = 32000;
int num_threads = 1;
string address = "127.0.0.1";

int dim_grad = 5000;
int num_user = 10;
uint64_t threshold = 0;
double thresholdRatio = 0.5;


sci::NetIO *io;
sci::OTPack<sci::NetIO> *otpack;
AuxProtocols *aux;
ReLURingProtocol<sci::NetIO, uint64_t> *relu;


int main(int argc, char **argv) {
  /************* Argument Parsing  ************/
  /********************************************/
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("nt", num_threads, "Number of threads");
  amap.arg("ip", address, "IP Address of server (ALICE)");

  amap.arg("n", num_user, "Number of users");
  amap.arg("d", dim_grad, "Gradient size");
  amap.arg("t", thresholdRatio, "threshold ratio");

  amap.parse(argc, argv);

  assert(num_threads <= MAX_THREADS);

  io = new sci::NetIO(party == 1 ? nullptr : address.c_str(), port);
  otpack = new sci::OTPack<sci::NetIO>(io, party);
  
  int32_t bitlength = ceil(log2(dim_grad + 1)) + 1;
  uint64_t mask = (bitlength == 64 ? -1 : ((1ULL << bitlength) - 1));

  int32_t bitlength_output = bitlength + ceil(log2(num_user));
  uint64_t mask_output = (bitlength_output == 64 ? -1 : ((1ULL << bitlength_output) - 1));

  aux = new AuxProtocols(party, io, otpack);
  relu = new ReLURingProtocol<sci::NetIO, uint64_t>(party, RING, io, bitlength, MILL_PARAM, otpack);
  

  /*********************** Step 0 Generate Test Data **************************/

  PRG128 prg;

  uint8_t *gs = new uint8_t[dim_grad];
  uint8_t *gi = new uint8_t[num_user * dim_grad];

  prg.random_data(gs, dim_grad * sizeof(uint8_t));
  prg.random_data(gi, dim_grad * num_user * sizeof(uint8_t)); // concatenate the gradients of all users

  for (int i = 0; i < dim_grad; i++) {
    gs[i] = (party == sci::ALICE ? gs[i] & 1 : 0);
  }

  for (int i = 0; i < dim_grad * num_user; i++) {
    gi[i] &= 1;
  }

  uint8_t *tmp_gs = new uint8_t[num_user * dim_grad];  // copy the server gradient $num_user$ times
  for (int i = 0; i < num_user; i ++){
    for (int j = 0; j < dim_grad; j++){
      tmp_gs[i * dim_grad + j] = gs[j];
    }
  }

  /******************************* Step 1 XONR *******************************/

  uint64_t total_comm = 0;
  long long total_time = 0;

  uint64_t comm = io->counter;
  auto start = clock_start();

  uint8_t *xonrOut = new uint8_t[num_user * dim_grad];

  aux->signSGD_xnor(num_user * dim_grad, tmp_gs, gi, xonrOut);

  long long t = time_from(start);
  comm = io->counter - comm;
  
  cout << "Step1---XNOR Time\t" << t / (1000.0) << " ms" << endl;
  cout << "Step1---XNOR Comm Sent\t" << comm / (1024.0 * 1024.0) << " MB" << endl;

  total_comm += comm;
  total_time += t;

  /******************************* Step 2 trustScore *******************************/

  comm = io->counter;
  start = clock_start();

  uint64_t *step2Out = new uint64_t[num_user];
  int32_t dPrime = aux->signSGD_bestSplit(dim_grad);
  cout << "bestSplit = " << dPrime << endl;
  
  aux->signSGD_trustScore(num_user, dim_grad, xonrOut, step2Out, dPrime);

  t = time_from(start);
  comm = io->counter - comm;
  
  cout << "Step2---trustScore Time\t" << t / (1000.0) << " ms" << endl;
  cout << "Step2---trustScore Comm Sent\t" << comm / (1024.0 * 1024.0) << " MB" << endl;

  total_comm += comm;
  total_time += t;

  /******************************* Step 3 filterByzantine *******************************/

  comm = io->counter;
  start = clock_start();

  threshold = (uint64_t)(dim_grad * thresholdRatio);

  uint64_t *step3In = new uint64_t[num_user];
  for (int i = 0; i < num_user; i++){
    step3In[i] = (party == sci::ALICE ? (step2Out[i] - threshold) & mask : step2Out[i]);
  }
  
  uint64_t *step3Out = new uint64_t[num_user];
  relu->relu(step3Out, step3In, num_user, nullptr, false);
  
  t = time_from(start);
  comm = io->counter - comm;
  
  cout << "Step3---filterByzantine Time\t" << t / (1000.0) << " ms" << endl;
  cout << "Step3---filterByzantine Comm Sent\t" << comm / (1024.0 * 1024.0) << " MB" << endl;
  total_comm += comm;
  total_time += t;

  /******************************* Step 4 weightAgg *******************************/

  comm = io->counter;
  start = clock_start();
  
  uint64_t *output = new uint64_t[dim_grad];
  cout <<"l2-bitlength = " << bitlength << endl;
  cout <<"l3-bitlength = " << bitlength_output << endl;

  aux->signSGD_weightAgg(gi, step3Out, output, num_user, dim_grad, bitlength);

  t = time_from(start);
  comm = io->counter - comm;
  
  cout << "Step4---weightAgg Time\t" << t / (1000.0) << " ms" << endl;
  cout << "Step4---weightAgg Comm Sent\t" << comm / (1024.0 * 1024.0) << " MB" << endl;

  total_comm += comm;
  total_time += t;

  cout << "-------------------------------------------------------------" << endl;
  cout << "Total Time\t" << total_time / (1000.0) << " ms" << endl;
  cout << "Total Comm Sent\t" << total_comm / (1024.0 * 1024.0) << " MB" << endl;


  /************** Verification ****************/

  if (party == sci::BOB) {
    io->send_data(gi, num_user * dim_grad * sizeof(uint8_t));

    io->send_data(xonrOut, num_user * dim_grad * sizeof(uint8_t));

    io->send_data(step2Out, num_user * sizeof(uint64_t));

    io->send_data(step3Out, num_user * sizeof(uint64_t));
    
    io->send_data(output, dim_grad * sizeof(uint64_t));
  } else { // party == ALICE
    uint8_t *gi1 = new uint8_t[num_user * dim_grad];
    io->recv_data(gi1, num_user * dim_grad * sizeof(uint8_t));

    for (int i = 0; i < num_user * dim_grad; i++) {
      gi[i] = gi[i] ^ gi1[i];
    }

    uint8_t *xonrOut1 = new uint8_t[num_user * dim_grad];
    io->recv_data(xonrOut1, num_user * dim_grad * sizeof(uint8_t));

    for (int i = 0; i < num_user * dim_grad; i++) {
      xonrOut[i] ^= xonrOut1[i];
    }
    
    uint64_t *step2Out1 = new uint64_t[num_user];
    io->recv_data(step2Out1, num_user * sizeof(uint64_t));

    for (int i = 0; i < num_user; i++) {
      step2Out[i] = (step2Out[i] + step2Out1[i]) & mask; 
    }

    uint64_t *step3Out1 = new uint64_t[num_user];
    io->recv_data(step3Out1, num_user * sizeof(uint64_t));

    for (int i = 0; i < num_user; i++) {
      step3Out[i] = (step3Out[i] + step3Out1[i]) & mask; 
    }

    uint64_t *output1 = new uint64_t[dim_grad];
    io->recv_data(output1, dim_grad * sizeof(uint64_t));

    for (int i = 0; i < dim_grad; i++) {
      output[i] = (output[i] + output1[i]) & mask_output; 
    }

    uint8_t *s = new uint8_t[num_user * dim_grad];
    int64_t *hamming = new int64_t[num_user];
    int64_t *weight = new int64_t[num_user];
    int64_t *result = new int64_t[dim_grad];

    for (int i = 0; i < num_user * dim_grad; i++) {
      s[i] = gi[i] ^ tmp_gs[i] ^ (uint8_t)1;

      if (xonrOut[i] != s[i]){
        std::cout << "Step 1 ---- Error: at index " << i << std::endl;
        std::cout << "xonrOut " << xonrOut[i] + 0 << std::endl;
        std::cout << "s " << s[i] + 0 << std::endl;
        return 0;
      }
    }

    for (int i = 0; i < num_user; i++){
      hamming[i] = 0;

      for (int j = 0; j < dim_grad; j++){
        hamming[i] += (int64_t)s[i * dim_grad + j];
      }

      if (hamming[i] != (int64_t)step2Out[i]){
        std::cout << "Step 2 ---- Error: at index " << i << std::endl;
        std::cout << "hamming " << hamming[i] << std::endl;
        std::cout << "step2Out " << (int64_t)step2Out[i] << std::endl;
        return 0;
      }

      weight[i] = (hamming[i] >= threshold ? hamming[i] - threshold : 0);

      if (weight[i] != (int64_t)step3Out[i]){
        std::cout << "Step 3 ---- Error: at index " << i << std::endl;
        std::cout << "weight " << weight[i] << std::endl;
        std::cout << "step3Out " << (int64_t)step3Out[i] << std::endl;
        return 0;
      }
    }

    for (int i = 0; i < dim_grad; i++){
      result[i] = 0;

      for (int j = 0; j < num_user; j++){
        result[i] += weight[j] * (int64_t)gi[j * dim_grad + i];
      }
      
      if (result[i] != (int64_t)output[i]){
        std::cout << "Step 4 ---- Error: at index " << i << std::endl;
        std::cout << "result " << result[i] << std::endl;
        std::cout << "output " << (int64_t)output[i] << std::endl;
        return 0;
      }
    }

    std::cout << "Correct operations !!!!!" << std::endl;
  }

  delete[] gs;
  delete[] gi;

  return 0;
}
