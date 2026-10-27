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

struct Capture{
 bool active=false;int watch_layer=-1;std::vector<float>watched;std::vector<std::vector<double>>norms=std::vector<std::vector<double>>(L);std::string error;
 static bool cb(ggml_tensor*t,bool ask,void*data)noexcept{auto&c=*(Capture*)data;try{if(!c.active||std::strncmp(t->name,"l_out-",6))return false;int l=std::stoi(t->name+6);need(l>=0&&l<L,"callback layer");need(t->type==GGML_TYPE_F32&&t->ne[0]==D&&ggml_is_contiguous(t),"callback shape");if(ask)return true;std::vector<float>x(ggml_nelements(t));ggml_backend_tensor_get(t,x.data(),0,x.size()*4);for(int64_t i=0;i<t->ne[1];i++)c.norms[l].push_back(norm(x.data()+i*D));if(l==c.watch_layer)c.watched.insert(c.watched.end(),x.begin(),x.end());return true;}catch(const std::exception&e){c.error=e.what();return false;}}
 void reset(){for(auto&x:norms)x.clear();watched.clear();error.clear();}
};
struct Runtime{
 llama_model*m=nullptr;llama_context*ctx=nullptr;const llama_vocab*vocab;llama_batch b;Capture cap;int V;
 Runtime(){auto mp=llama_model_default_params();mp.n_gpu_layers=0;mp.use_mmap=true;m=llama_model_load_from_file(MODEL.c_str(),mp);need(m,"load model");vocab=llama_model_get_vocab(m);V=llama_vocab_n_tokens(vocab);auto cp=llama_context_default_params();cp.n_ctx=512;cp.n_batch=cp.n_ubatch=64;cp.n_threads=cp.n_threads_batch=4;cp.n_seq_max=1;cp.offload_kqv=cp.op_offload=false;cp.swa_full=true;cp.cb_eval=Capture::cb;cp.cb_eval_user_data=&cap;ctx=llama_init_from_model(m,cp);need(ctx,"context");b=llama_batch_init(64,0,1);}
 ~Runtime(){llama_batch_free(b);llama_free(ctx);llama_model_free(m);}
 std::vector<llama_token>tokens(const std::string&text){int n=llama_tokenize(vocab,text.data(),int(text.size()),nullptr,0,true,true);need(n<0,"token size");std::vector<llama_token>t(-n);need(llama_tokenize(vocab,text.data(),int(text.size()),t.data(),int(t.size()),true,true)==-n,"tokenize");need(t.size()<=512,"ctx overflow");return t;}
 std::string render(const std::string&user){const char*tmpl=llama_model_chat_template(m,nullptr);need(tmpl,"chat template missing");std::string s=tmpl;need(s.find("'<|turn>' + role")!=std::string::npos&&s.find("'<turn|>\\n'")!=std::string::npos&&s.find("'<|turn>model\\n'")!=std::string::npos,"single-user Gemma4 template assumptions");
  // This checkout's legacy llama_chat_apply_template does not support Gemma4.
  // Exact single text user branch of the GGUF Jinja: no tools/system/thinking.
  // BOS is supplied once by tokenize(add_special=true), not embedded twice.
  return "<|turn>user\n"+user+"<turn|>\n<|turn>model\n";}
 std::vector<float>forward(const std::vector<llama_token>&t,bool capture){cap.reset();cap.active=capture;llama_memory_clear(llama_get_memory(ctx),true);for(size_t at=0;at<t.size();at+=64){int cnt=int(std::min(size_t(64),t.size()-at));b.n_tokens=cnt;for(int i=0;i<cnt;i++){b.token[i]=t[at+i];b.pos[i]=int(at+i);b.n_seq_id[i]=1;b.seq_id[i][0]=0;b.logits[i]=capture||at+i+1==t.size();}need(llama_decode(ctx,b)==0,"decode");llama_synchronize(ctx);need(cap.error.empty(),cap.error);}const float*p=llama_get_logits_ith(ctx,-1);need(p,"logits");std::vector<float>z(p,p+V);for(float x:z)need(std::isfinite(x),"nonfinite runtime");if(capture)for(auto&x:cap.norms)need(x.size()==t.size(),"every-token layer coverage");return z;}
 void cvec(const std::vector<float>*v,int layer,float dose){if(!v){need(llama_set_adapter_cvec(ctx,nullptr,0,D,1,34)==0,"disable cvec");return;}need(layer>0&&layer<L,"no cvec layer0");std::vector<float>data(34*D,0);for(int d=0;d<D;d++)data[size_t(layer-1)*D+d]=(*v)[d]*dose;need(llama_set_adapter_cvec(ctx,data.data(),data.size(),D,layer,layer)==0,"set cvec");}
};


int main(int argc,char**argv){try{
 ggml_backend_load_all();llama_backend_init();Runtime rt;
 if(argc>1 && std::string(argv[1])=="tokenize"){
  for(std::string name:{"scenarios","validation_prompts"}){
   auto x=readj("S3/"+name+"_source.json");
   for(auto &p:x)p["token_ids"]=rt.tokens(p["rendered"]);
   writej("S3/"+name+".json",x);
  }
  json pieces=json::array();
  const char*hex="0123456789abcdef";
  for(int i=0;i<rt.V;i++){char buf[4096];int n=llama_token_to_piece(rt.vocab,i,buf,sizeof(buf),0,true);need(n>=0,"piece buffer");std::string h;for(int j=0;j<n;j++){unsigned char b=buf[j];h+=hex[b>>4];h+=hex[b&15];}pieces.push_back(h);}
  writej("S3/inputs/token_piece_hex.json",pieces);
  return 0;
 }
 auto prompts=readj("S3/validation_prompts.json");json results=json::array();
 for(auto&p:prompts){auto start=Clock::now();auto ids=p["token_ids"].get<std::vector<int>>();auto z=rt.forward(ids,false);json out=json::array();std::string reply;
  for(int step=0;step<96;step++){
   int id=int(std::max_element(z.begin(),z.end())-z.begin());out.push_back(id);
   if(llama_vocab_is_eog(rt.vocab,id))break;
   char buf[4096];int n=llama_token_to_piece(rt.vocab,id,buf,sizeof(buf),0,true);need(n>=0,"piece");reply.append(buf,n);
   rt.b.n_tokens=1;rt.b.token[0]=id;rt.b.pos[0]=int(ids.size()+step);rt.b.n_seq_id[0]=1;rt.b.seq_id[0][0]=0;rt.b.logits[0]=true;
   need(llama_decode(rt.ctx,rt.b)==0,"generation decode");llama_synchronize(rt.ctx);
   const float*q=llama_get_logits_ith(rt.ctx,-1);z.assign(q,q+rt.V);
  }
  results.push_back({{"id",p["id"]},{"token_ids",out},{"reply",reply},{"seconds",sec(start)}});
  std::cerr<<"CPU greedy "<<p["id"]<<" tokens="<<out.size()<<" seconds="<<sec(start)<<'\n';
 }
 std::cout<<results.dump(2,' ',false,json::error_handler_t::replace)<<'\n';return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
