// Numeric compatibility table for the configured reference CPU libm, not model data.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <chrono>
#include <dlfcn.h>
int main(){
 auto start=std::chrono::steady_clock::now();
 using EF=float(*)(float);using ED=double(*)(double);
 EF volatile ef=(EF)dlsym(RTLD_DEFAULT,"expf");ED volatile ed=(ED)dlsym(RTLD_DEFAULT,"exp");
 constexpr uint32_t size=1u<<25,mask=size-1;
 std::vector<uint32_t>keys(size,0),values(size,0);uint64_t count=0;
 for(uint32_t bits=0x33000000u;bits<0x43000000u;bits++){
  uint32_t neg=bits|0x80000000u;float x;memcpy(&x,&neg,4);float a=ef(x),b=float(ed(double(x)));
  uint32_t ia,ib;memcpy(&ia,&a,4);memcpy(&ib,&b,4);
  if(ia!=ib){
   if(++count>size*3/4){fprintf(stderr,"too many corrections\n");return 2;}
   uint32_t at=(bits*2654435761u)&mask;while(keys[at])at=(at+1)&mask;
   keys[at]=bits;values[at]=ia;
  }
 }
 uint64_t header[2]={size,count};fwrite(header,8,2,stdout);fwrite(keys.data(),4,size,stdout);fwrite(values.data(),4,size,stdout);
 double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
 fprintf(stderr,"{\"entries\":%llu,\"capacity\":%u,\"seconds\":%.6f,\"domain_float32_abs_bits_begin\":855638016,\"domain_float32_abs_bits_end\":1124073472}\n",(unsigned long long)count,size,seconds);
 return ferror(stdout)?1:0;
}
