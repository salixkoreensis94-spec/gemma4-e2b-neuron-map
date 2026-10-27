#include "ggml-cpu.h"
// CPU-only S1. GGUF is mapped read-only; no model/source modification.
// Callback/batch/cvec patterns adapted from atlas_probe.cpp and atlas_intervene.cpp.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "nlohmann/json.hpp"
#include <cblas.h>
#include <zlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>
using json=nlohmann::json;
using Clock=std::chrono::steady_clock;
static constexpr int D=1536,L=35,C=512;
static const std::string MODEL=[](){auto*p=std::getenv("NEURON_GGUF");if(!p)throw std::runtime_error("set NEURON_GGUF");return std::string(p);}();
void need(bool v,const std::string&s){if(!v)throw std::runtime_error(s);}
double sec(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
json readj(const std::string&p){std::ifstream f(p);need(bool(f),"open "+p);json j;f>>j;return j;}
void writej(const std::string&p,const json&j){std::ifstream old(p);need(!old.good(),"output exists: "+p);std::ofstream f(p);f<<j.dump(2,' ',false,json::error_handler_t::replace)<<'\n';need(bool(f),"write "+p);}
struct Gzip{gzFile f;Gzip(const std::string&p){std::ifstream old(p);need(!old.good(),"output exists: "+p);f=gzopen(p.c_str(),"wb1");need(f,"gzip open");gzbuffer(f,1<<20);}void put(const std::string&s){need(gzwrite(f,s.data(),unsigned(s.size()))==int(s.size()),"gzip write");}~Gzip(){if(f)gzclose(f);}};
struct GG{
 int fd;size_t size;const char*map;gguf_context*g=nullptr;ggml_context*ctx=nullptr;int V;float eps,cap;std::vector<std::string>tok;std::vector<int>types;
 GG(){fd=open(MODEL.c_str(),O_RDONLY);need(fd>=0,"model open");struct stat st{};need(fstat(fd,&st)==0,"stat");size=st.st_size;map=(const char*)mmap(nullptr,size,PROT_READ,MAP_PRIVATE,fd,0);need(map!=MAP_FAILED,"mmap");gguf_init_params p{true,&ctx};g=gguf_init_from_file(MODEL.c_str(),p);need(g&&ctx,"GGUF metadata");
  int k=gguf_find_key(g,"tokenizer.ggml.tokens");need(k>=0,"vocab");V=int(gguf_get_arr_n(g,k));for(int i=0;i<V;i++)tok.emplace_back(gguf_get_arr_str(g,k,i));
  k=gguf_find_key(g,"tokenizer.ggml.token_type");need(k>=0,"token types");auto*tp=(const int32_t*)gguf_get_arr_data(g,k);types.assign(tp,tp+V);
  eps=number("gemma4.attention.layer_norm_rms_epsilon",-1);cap=number("gemma4.final_logit_softcapping",0);need(eps>0,"RMS epsilon missing");need(number("gemma4.block_count",0)==L&&number("gemma4.embedding_length",0)==D,"model dimensions");
 }
 ~GG(){gguf_free(g);ggml_free(ctx);munmap((void*)map,size);close(fd);}
 float number(const char*key,float def){int k=gguf_find_key(g,key);if(k<0)return def;switch(gguf_get_kv_type(g,k)){case GGUF_TYPE_FLOAT32:return gguf_get_val_f32(g,k);case GGUF_TYPE_UINT32:return gguf_get_val_u32(g,k);case GGUF_TYPE_INT32:return gguf_get_val_i32(g,k);default:throw std::runtime_error("numeric type");}}
 ggml_tensor*tensor(const std::string&s){auto*t=ggml_get_tensor(ctx,s.c_str());need(t,"tensor "+s);return t;}
 const char*data(const std::string&s){int i=gguf_find_tensor(g,s.c_str());need(i>=0,"tensor offset");auto*t=tensor(s);size_t off=gguf_get_data_offset(g)+gguf_get_tensor_offset(g,i);need(off+ggml_nbytes(t)<=size,"tensor extent");return map+off;}
 std::vector<float>dequant(const std::string&s){auto*t=tensor(s);size_t n=ggml_nelements(t);std::vector<float>x(n);const char*p=data(s);auto*tr=ggml_get_type_traits(t->type);size_t rb=ggml_row_size(t->type,t->ne[0]);for(int64_t r=0;r<ggml_nrows(t);r++){if(t->type==GGML_TYPE_F32)std::memcpy(x.data()+r*t->ne[0],p+r*rb,t->ne[0]*4);else{need(tr->to_float,"dequant type");tr->to_float(p+r*rb,x.data()+r*t->ne[0],t->ne[0]);}}return x;}
 int width(int l){auto*t=tensor("blk."+std::to_string(l)+".ffn_down.weight");need(t->ne[1]==D&&t->ne[2]==1&&t->ne[3]==1,"down orientation");return int(t->ne[0]);}
 std::vector<float>columns(int l){int w=width(l);auto a=dequant("blk."+std::to_string(l)+".ffn_down.weight");std::vector<float>b(size_t(w)*D);for(int n=0;n<w;n++)for(int d=0;d<D;d++)b[size_t(n)*D+d]=a[size_t(d)*w+n];return b;}
 std::string outname(){return gguf_find_tensor(g,"output.weight")<0?"token_embd.weight":"output.weight";}
 void inspect(){json meta=json::object();for(int k=0;k<gguf_get_n_kv(g);k++){std::string key=gguf_get_key(g,k);auto t=gguf_get_kv_type(g,k);if(t==GGUF_TYPE_STRING)meta[key]=gguf_get_val_str(g,k);else if(t==GGUF_TYPE_FLOAT32)meta[key]=gguf_get_val_f32(g,k);else if(t==GGUF_TYPE_UINT32)meta[key]=gguf_get_val_u32(g,k);else if(t==GGUF_TYPE_INT32)meta[key]=gguf_get_val_i32(g,k);else if(t==GGUF_TYPE_BOOL)meta[key]=gguf_get_val_bool(g,k);else if(t==GGUF_TYPE_ARRAY)meta[key]={{"array_length",gguf_get_arr_n(g,k)}};}
 json widths=json::array(),down=json::array();int total=0;for(int l=0;l<L;l++){int w=width(l);widths.push_back(w);total+=w;down.push_back({{"layer",l},{"ggml_shape",{w,D}},{"type",ggml_type_name(tensor("blk."+std::to_string(l)+".ffn_down.weight")->type)}});}
 auto*t=tensor(outname());need(t->ne[0]==D&&t->ne[1]==V,"output shape");
 writej("S1/model_metadata.json",{{"model",MODEL},{"file_bytes",size},{"metadata",meta},{"layers",L},{"d_model",D},{"widths",widths},{"neuron_count",total},{"vocab_size",V},{"rms_epsilon",eps},{"final_logit_softcap",cap},{"output_tensor",outname()},{"output_tied",outname()=="token_embd.weight"},{"output_type",ggml_type_name(t->type)},{"down_tensors",down}});
 json v=json::array();for(int i=0;i<V;i++)v.push_back({{"id",i},{"string",tok[i]},{"type",types[i]}});writej("S1/vocabulary.json",v);
 }
};
float norm(const float*v){double s=0;for(int d=0;d<D;d++)s+=double(v[d])*v[d];return float(std::sqrt(s));}
void random_dir(std::mt19937_64&r,float*v,float mag){std::normal_distribution<float>nd;for(int d=0;d<D;d++)v[d]=nd(r);float k=mag/norm(v);for(int d=0;d<D;d++)v[d]*=k;}


int main(){GG gg;json out=json::array();auto ff=gg.dequant("rope_freqs.weight");
 for(bool swa:{false,true}){
  int d=gg.number(swa?"gemma4.attention.key_length_swa":"gemma4.attention.key_length",0);float base=gg.number(swa?"gemma4.rope.freq_base_swa":"gemma4.rope.freq_base",0);
  ggml_init_params ip{64*1024*1024,nullptr,false};auto*c=ggml_init(ip);
  auto*x=ggml_new_tensor_3d(c,GGML_TYPE_F32,d,1,512);auto*p=ggml_new_tensor_1d(c,GGML_TYPE_I32,512);ggml_tensor*f=nullptr;
  if(!swa){f=ggml_new_tensor_1d(c,GGML_TYPE_F32,d/2);memcpy(f->data,ff.data(),(d/2)*4);}
  for(int at=0;at<512;at++){((int32_t*)p->data)[at]=at;for(int i=0;i<d;i++)((float*)x->data)[at*d+i]=i<d/2?1.0f:0.0f;}
  auto*y=ggml_rope_ext(c,x,p,f,d,2,131072,base,1,0,1,32,1);auto*g=ggml_new_graph(c);ggml_build_forward_expand(g,y);need(ggml_graph_compute_with_ctx(c,g,4)==GGML_STATUS_SUCCESS,"compute rope");
  json rows=json::array();auto*v=(float*)y->data;for(int at=0;at<512;at++){json row=json::array();for(int i=0;i<d/2;i++){row.push_back(v[at*d+i]);row.push_back(v[at*d+d/2+i]);}rows.push_back(row);}out.push_back(rows);ggml_free(c);
 }std::cout<<out.dump()<<'\n';}
