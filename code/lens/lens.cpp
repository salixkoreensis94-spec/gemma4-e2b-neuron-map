// CPU-only S1. GGUF is mapped read-only; no model/source modification.
// Callback/batch/cvec patterns adapted from atlas_probe.cpp and atlas_intervene.cpp.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "nlohmann/json.hpp"
#include <cblas.h>
#include <filesystem>
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
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>
using json=nlohmann::json;
using Clock=std::chrono::steady_clock;
static constexpr int D=1536,L=35,C=512;
static std::string MODEL;
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

struct Lens {
    GG &g;
    std::vector<float> E, N;
    std::vector<std::string> names;
    std::vector<std::vector<int>> groups;
    double max_dot_error=0, scale_invariance_error=0;
    uint64_t dot_checks=0, extreme_count=0, near_cap_extremes=0;
    struct Stats { double mean, sd; std::vector<double> means, scores; };

    Lens(GG &gg):g(gg) {
        auto t=Clock::now();
        E=g.dequant(g.outname()); N=g.dequant("output_norm.weight");
        auto group_json=readj("S1/token_groups.json");
        for(auto &x:group_json["groups"]) {
            names.push_back(x["name"]);
            groups.push_back(x["ids"].get<std::vector<int>>());
            need(!groups.back().empty(),"empty curated group");
        }
        std::cerr<<"embedding dequant seconds "<<sec(t)<<'\n';
    }
    std::vector<float> logits(const float *v,int count) {
        std::vector<float> a(size_t(count)*D), z(size_t(count)*g.V);
        for(int i=0;i<count;i++) {
            double ss=0;
            for(int d=0;d<D;d++)ss+=double(v[size_t(i)*D+d])*v[size_t(i)*D+d];
            float inv=float(1/std::sqrt(ss/D+g.eps));
            for(int d=0;d<D;d++)a[size_t(i)*D+d]=v[size_t(i)*D+d]*inv*N[d];
        }
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,count,g.V,D,1,a.data(),D,E.data(),D,0,z.data(),g.V);
        for(int i:{0,count-1})for(int id:{0,1234,g.V-1}) {
            double ref=0;
            for(int d=0;d<D;d++)ref+=double(a[size_t(i)*D+d])*E[size_t(id)*D+d];
            max_dot_error=std::max(max_dot_error,std::abs(ref-z[size_t(i)*g.V+id]));
            ++dot_checks;
        }
        // Deliberately return PRE-softcap logits. No tanh here.
        return z;
    }
    Stats stats(const float *z) {
        double sum=0, sumsq=0;
        for(int id=0;id<g.V;id++) {
            double x=z[id]; if(!std::isfinite(x))throw std::runtime_error("nonfinite pre-softcap lens");
            sum+=x; sumsq+=x*x;
        }
        Stats st; st.mean=sum/g.V;
        st.sd=std::sqrt(sumsq/g.V-st.mean*st.mean);
        need(std::isfinite(st.sd)&&st.sd>0,"invalid full-vocabulary standard deviation");
        for(auto &ids:groups) {
            double s=0; for(int id:ids)s+=z[id];
            double mean=s/ids.size();
            st.means.push_back(mean); st.scores.push_back((mean-st.mean)/st.sd);
        }
        return st;
    }
    void check_stats(const float *z) {
        // Independent centered long-double variance and mean-of-token-z scores.
        auto st=stats(z);
        long double sum=0; for(int id=0;id<g.V;id++)sum+=z[id];
        long double mean=sum/g.V, var=0;
        for(int id=0;id<g.V;id++){long double d=z[id]-mean;var+=d*d;}
        long double sd=std::sqrt(var/g.V);
        need(std::abs(double(mean)-st.mean)<1e-9&&std::abs(double(sd)-st.sd)<1e-8,"independent moments mismatch");
        for(size_t k=0;k<groups.size();k++) {
            long double ref=0;
            for(int id:groups[k])ref+=(z[id]-mean)/sd;
            ref/=groups[k].size();
            need(std::abs(double(ref)-st.scores[k])<1e-9,"mean token z-score mismatch");
        }
        std::vector<float> shifted(g.V);
        for(int id=0;id<g.V;id++)shifted[id]=float(3.7*double(z[id])+11.0);
        auto transformed=stats(shifted.data());
        for(size_t k=0;k<groups.size();k++)scale_invariance_error=std::max(scale_invariance_error,std::abs(st.scores[k]-transformed.scores[k]));
        need(scale_invariance_error<1e-5,"scale/shift invariance check failed");
    }
    json extremes(const float *z,bool top) {
        std::vector<int> ids(g.V); std::iota(ids.begin(),ids.end(),0);
        auto cmp=[&](int a,int b){return z[a]==z[b]?a<b:(top?z[a]>z[b]:z[a]<z[b]);};
        std::nth_element(ids.begin(),ids.begin()+50,ids.end(),cmp);
        std::sort(ids.begin(),ids.begin()+50,cmp);
        json out=json::array();
        for(int i=0;i<50;i++) {
            int id=ids[i]; double raw=z[id];
            double capped=g.cap>0?double(g.cap)*std::tanh(raw/g.cap):raw;
            out.push_back({id,g.tok[id],raw,capped});
            ++extreme_count; near_cap_extremes+=std::abs(capped)>=29.9;
        }
        return out;
    }
    std::string header(bool percentile) {
        std::string s="layer\tindex\tnorm\tpre_mean\tpre_std";
        for(auto &n:names) {
            s+='\t'+n+"_score";
            if(percentile)s+='\t'+n+"_percentile";
            s+='\t'+n+"_pre_mean";
        }
        return s+"\ttop50\tbottom50\n";
    }
    std::string row(int l,int n,float nm,const Stats &st,const std::vector<std::vector<double>> &cs,const float *z) {
        std::ostringstream o; o.precision(17);
        o<<l<<'\t'<<n<<'\t'<<nm<<'\t'<<st.mean<<'\t'<<st.sd;
        for(size_t k=0;k<st.scores.size();k++) {
            double score=st.scores[k]; o<<'\t'<<score;
            if(!cs.empty()) {
                need(cs[k].size()==C,"512 complete controls required for percentiles");
                double less=0,equal=0;
                for(double x:cs[k]){less+=x<score;equal+=x==score;}
                o<<'\t'<<100*(less+.5*equal)/C;
            }
            o<<'\t'<<st.means[k];
        }
        o<<'\t'<<extremes(z,true).dump()<<'\t'<<extremes(z,false).dump()<<'\n';
        return o.str();
    }
    void run(bool pilot) {
        auto start=Clock::now(); const int batch=128;
        Gzip neurons(pilot?"S1/logs/s1b_pilot_neurons.tsv.gz":"S1/neurons.tsv.gz");
        Gzip controls(pilot?"S1/logs/s1b_pilot_controls.tsv.gz":"S1/controls.tsv.gz");
        neurons.put(header(true)); controls.put(header(false));
        std::ifstream cv("S1/control_directions.f32",std::ios::binary);
        need(bool(cv),"saved S1 controls missing");
        json times=json::array();
        for(int l=0;l<(pilot?1:L);l++) {
            auto lt=Clock::now(); auto v=g.columns(l); int w=g.width(l);
            std::vector<float> c(size_t(C)*D);
            cv.read((char*)c.data(),c.size()*sizeof(float)); need(bool(cv),"read exact S1 controls");
            std::vector<std::vector<double>> cs(names.size());
            for(int at=0;at<C;at+=batch) {
                auto z=logits(c.data()+size_t(at)*D,batch);
                if(pilot&&at==0)check_stats(z.data());
                for(int i=0;i<batch;i++) {
                    const float *p=z.data()+size_t(i)*g.V; auto st=stats(p);
                    for(size_t k=0;k<cs.size();k++)cs[k].push_back(st.scores[k]);
                    controls.put(row(l,at+i,norm(c.data()+size_t(at+i)*D),st,{},p));
                }
            }
            int nn=pilot?128:w;
            for(int at=0;at<nn;at+=batch) {
                int cnt=std::min(batch,nn-at); auto z=logits(v.data()+size_t(at)*D,cnt);
                if(at==0)check_stats(z.data());
                for(int i=0;i<cnt;i++) {
                    const float *p=z.data()+size_t(i)*g.V; auto st=stats(p);
                    neurons.put(row(l,at+i,norm(v.data()+size_t(at+i)*D),st,cs,p));
                }
                if(at%2048==0)std::cerr<<"lens layer "<<l<<" neurons "<<at+cnt<<"/"<<nn<<" elapsed "<<sec(start)<<'\n';
            }
            times.push_back({{"layer",l},{"neurons",nn},{"controls",C},{"seconds",sec(lt)}});
            std::cerr<<"layer complete "<<times.back()<<'\n';
        }
        writej(pilot?"S1/logs/s1b_lens_pilot.json":"S1/lens_run.json",{
            {"schema","S1b-pre-softcap-zscore-v1"},{"seconds",sec(start)},{"layers",times},
            {"scalar_dot_max_abs_error_before_softcap",max_dot_error},{"scalar_dot_checks",dot_checks},
            {"scale_and_shift_invariance_max_abs_score_error",scale_invariance_error},
            {"independent_moments_and_mean_token_z_check",true},{"batch",batch},{"controls_per_layer",C},
            {"controls_reused_exactly_from_S1",true},{"primary_score","mean pre-softcap within-direction token z-score, population std"},
            {"percentile","100 * (layer control scores strictly below + 0.5 * ties) / 512"},
            {"extreme_entry_schema",{"token_id","raw_GGUF_string","pre_softcap_logit","post_softcap_reference"}},
            {"extreme_entries",extreme_count},{"extreme_entries_abs_post_at_least_29_9",near_cap_extremes},
            {"float_serialization_significant_digits",17},{"pilot",pilot}});
    }
};
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
double quantile(std::vector<double>v,double q){need(!v.empty(),"quantile empty");std::sort(v.begin(),v.end());double at=q*(v.size()-1);int a=int(at),b=int(std::ceil(at));return v[a]+(v[b]-v[a])*(at-a);}
void calibrate(bool pilot){auto start=Clock::now();Runtime rt;auto prompts=readj("S1/prompts.json");int count=pilot?2:40;std::vector<std::vector<double>>all(L);json rendered=json::array(),perprompt=json::array();Gzip raw(pilot?"S1/logs/residual_pilot_tokens.jsonl.gz":"S1/residual_token_norms.jsonl.gz");double maxdiff=0;int unequal=0,total=0;
 for(int p=0;p<count;p++){auto text=rt.render(prompts[p]["user"]);auto t=rt.tokens(text);auto tm=Clock::now();auto z=rt.forward(t,true);if(p==0){auto base=rt.forward(t,false);for(int i=0;i<rt.V;i++){maxdiff=std::max(maxdiff,std::abs(double(z[i])-base[i]));unequal+=std::memcmp(&z[i],&base[i],4)!=0;}rt.forward(t,true);}
  for(int l=0;l<L;l++)all[l].insert(all[l].end(),rt.cap.norms[l].begin(),rt.cap.norms[l].end());for(size_t i=0;i<t.size();i++){json ns=json::array();for(int l=0;l<L;l++)ns.push_back(rt.cap.norms[l][i]);raw.put(json({{"prompt_id",p},{"position",i},{"token_id",t[i]},{"layer_norms",ns}}).dump()+"\n");}rendered.push_back({{"id",p},{"user",prompts[p]["user"]},{"register",prompts[p]["register"]},{"rendered",text},{"token_ids",t}});perprompt.push_back({{"id",p},{"tokens",t.size()},{"seconds",sec(tm)}});total+=int(t.size());std::cerr<<"calibrate prompt "<<p+1<<'/'<<count<<" tokens "<<t.size()<<" seconds "<<sec(tm)<<'\n';
 }
 json layers=json::array();for(int l=0;l<L;l++)layers.push_back({{"layer",l},{"tokens",all[l].size()},{"median",quantile(all[l],.5)},{"p90",quantile(all[l],.9)}});
 writej(pilot?"S1/logs/residual_pilot.json":"S1/residual_norms.json",{{"prompts",count},{"tokens",total},{"scope","all prompt tokens including chat-template tokens; l_out after out_scale and cvec (disabled)"},{"quantile","linear interpolation at q*(N-1), token-weighted"},{"layers",layers},{"callback_fidelity_max_abs_logit_difference",maxdiff},{"callback_fidelity_bitwise_unequal_logits",unequal},{"seconds",sec(start)},{"per_prompt",perprompt}});writej(pilot?"S1/logs/prompts_pilot_rendered.json":"S1/prompts_rendered.json",rendered);
}
void sanity(bool pilot){auto start=Clock::now();GG g;auto selection=readj("S1/selection.json");auto residual=readj("S1/residual_norms.json");auto prompts=readj("S1/prompts_rendered.json");Runtime rt;std::vector<std::vector<llama_token>>ts;std::vector<std::vector<float>>base;for(int p=0;p<8;p++){ts.push_back(prompts[p]["token_ids"].get<std::vector<llama_token>>());base.push_back(rt.forward(ts.back(),false));}
 json rows=json::array(),checks=json::array();int limit=pilot?2:48;std::vector<float>last;int last_layer=1;
 for(int k=0;k<limit;k++){auto s=selection["directions"][k];int l=s["layer"],n=s["index"];std::vector<float>v(D);if(s["kind"]=="neuron"){auto cols=g.columns(l);std::copy_n(cols.data()+size_t(n)*D,D,v.data());}else{std::ifstream f("S1/control_directions.f32",std::ios::binary);f.seekg((size_t(l)*C+n)*D*4);f.read((char*)v.data(),D*4);need(bool(f),"read control direction");}float nm=norm(v.data());for(float&x:v)x/=nm;double median=residual["layers"][l]["median"];auto top=s["top10"].get<std::vector<int>>();auto random=s["random10"].get<std::vector<int>>();int direction_prompts=pilot?1:8;
  for(int sign:{1,-1}){float dose=float(sign*.5*median);rt.cvec(&v,l,dose);for(int p=0;p<direction_prompts;p++){auto z=rt.forward(ts[p],false);json dt=json::array(),dr=json::array();double mt=0,mr=0;int nt=0,nr=0;for(int id:top){double d=double(z[id])-base[p][id];dt.push_back(d);mt+=d;nt+=sign*d>0;}for(int id:random){double d=double(z[id])-base[p][id];dr.push_back(d);mr+=d;nr+=sign*d>0;}rows.push_back({{"direction",k},{"kind",s["kind"]},{"selection",s["selection"]},{"layer",l},{"index",n},{"prompt",p},{"sign",sign},{"dose_l2",dose},{"top10_delta",dt},{"random10_delta",dr},{"top10_mean_delta",mt/10},{"random10_mean_delta",mr/10},{"top10_sign_consistent",nt},{"random10_sign_consistent",nr},{"signed_top_mean_exceeds_random",sign*(mt-mr)>0}});}}
  // With a single final-layer push, the cvec API's residual delta must equal dose*v.
  if(k==0){rt.cap.watch_layer=l;rt.cvec(nullptr,1,0);rt.forward(ts[0],true);auto normbase=rt.cap.norms;auto rawbase=rt.cap.watched;float dose=float(.5*median);rt.cvec(&v,l,dose);rt.forward(ts[0],true);need(rawbase.size()==rt.cap.watched.size(),"cvec capture coverage");double err=0,before=0;for(size_t i=0;i<rawbase.size();i++)err=std::max(err,std::abs(double(rt.cap.watched[i])-rawbase[i]-v[i%D]*dose));for(int j=0;j<l;j++)for(size_t i=0;i<normbase[j].size();i++)before=std::max(before,std::abs(normbase[j][i]-rt.cap.norms[j][i]));need(err<1e-3&&before==0,"cvec boundary verification failed");checks.push_back({{"direction",k},{"layer",l},{"base_layer_endpoint_norm",normbase[l].back()},{"pushed_layer_endpoint_norm",rt.cap.norms[l].back()},{"max_abs_error_observed_delta_vs_dose_direction",err},{"preceding_layers_max_abs_norm_change",before},{"verified_tokens",ts[0].size()},{"applied_scope","all prefill tokens at exactly one layer"}});rt.cap.watch_layer=-1;}
  last=v;last_layer=l;std::cerr<<"sanity direction "<<k+1<<'/'<<limit<<" elapsed "<<sec(start)<<'\n';
 }
 rt.cvec(&last,last_layer,0);auto zero=rt.forward(ts[0],false);rt.cvec(nullptr,1,0);auto disabled=rt.forward(ts[0],false);double zeroerr=0,disableerr=0;for(int i=0;i<rt.V;i++){zeroerr=std::max(zeroerr,std::abs(double(zero[i])-base[0][i]));disableerr=std::max(disableerr,std::abs(double(disabled[i])-base[0][i]));}
 writej(pilot?"S1/logs/sanity_pilot.json":"S1/sanity.json",{{"directions",limit},{"prompts",pilot?1:8},{"signs",{1,-1}},{"scope","cvec API applied to all prefill tokens at selected layer; cleared KV for every run; logits only, no token sampled"},{"rows",rows},{"checks",checks},{"zero_vector_max_abs_logit_difference",zeroerr},{"disabled_after_push_max_abs_logit_difference",disableerr},{"seconds",sec(start)}});
}

int main(int argc,char **argv) {
    try {
        need(argc==6,"usage: lens MODE --model GGUF --workdir DIRECTORY");
        need(std::string(argv[2])=="--model" && std::string(argv[4])=="--workdir", "flags");
        MODEL=std::filesystem::absolute(argv[3]).string();
        std::filesystem::current_path(argv[5]);
        ggml_backend_load_all(); llama_backend_init();
        std::string cmd=argv[1];
        if(cmd=="lens"||cmd=="lens-pilot"){GG g;Lens lens(g);lens.run(cmd=="lens-pilot");}
        else if(cmd=="sanity")sanity(false);
        else if(cmd=="inspect"){GG g;g.inspect();}
        else if(cmd=="calibrate")calibrate(false);
        else if(cmd=="controls"){
            std::ifstream old("S1/control_directions.f32",std::ios::binary);need(!old.good(),"controls exist");
            GG g;std::ofstream f("S1/control_directions.f32",std::ios::binary);
            for(int l=0;l<L;l++){
                auto v=g.columns(l);int w=g.width(l);std::vector<float>norms(w);
                for(int n=0;n<w;n++)norms[n]=norm(v.data()+size_t(n)*D);
                std::mt19937_64 rng(2026102600ULL+l);std::uniform_int_distribution<int>pick(0,w-1);
                std::vector<float>c(size_t(C)*D);
                for(int n=0;n<C;n++)random_dir(rng,c.data()+size_t(n)*D,norms[pick(rng)]);
                f.write((char*)c.data(),c.size()*sizeof(float));need(bool(f),"write controls");
            }
        }
        else throw std::runtime_error("unknown S1b command");
        llama_backend_free();return 0;
    } catch(const std::exception &e) {std::cerr<<"S1b ERROR "<<e.what()<<'\n';return 1;}
}
