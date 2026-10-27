#include "llama.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
using json=nlohmann::json;
int main(int argc,char**argv){try{
 if(argc!=4)throw std::runtime_error("input output model.gguf required");
 std::ifstream old(argv[2]);if(old.good())throw std::runtime_error("output exists");
 auto p=llama_model_default_params();p.vocab_only=true;p.n_gpu_layers=0;p.use_mmap=true;
 auto*m=llama_model_load_from_file(argv[3],p);
 if(!m)throw std::runtime_error("vocab load");auto*v=llama_model_get_vocab(m);
 std::ifstream f(argv[1]);json rows;f>>rows;
 for(auto&r:rows){std::string s=r["text"];int n=llama_tokenize(v,s.data(),s.size(),nullptr,0,true,true);
  if(n>=0)throw std::runtime_error("token size");std::vector<llama_token>x(-n);
  if(llama_tokenize(v,s.data(),s.size(),x.data(),x.size(),true,true)!=-n)throw std::runtime_error("tokenize");
  r["token_ids"]=x;}
 std::ofstream out(argv[2]);out<<rows.dump(2,' ',false)<<'\n';llama_model_free(m);return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
