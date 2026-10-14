"""Verified Mandia N’zassa inference; weights are distributed in Releases."""
import argparse,pathlib,sys,hashlib,json
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent/'src'))
import torch
from tokenizers import Tokenizer
from mandia_seq2seq import MandiaS2S,S2SConfig
from traduire_texte import predict

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(description='Mandia N’zassa: experimental French to Baoule text')
 p.add_argument('--weights',type=pathlib.Path,required=True);p.add_argument('--manifest',type=pathlib.Path);p.add_argument('--context',default='');p.add_argument('--text',required=True);p.add_argument('--mode',choices=['reply','translate'],default='translate');p.add_argument('--device',default='cpu');p.add_argument('--max-new-tokens',type=int);p.add_argument('--json',action='store_true');a=p.parse_args()
 root=pathlib.Path(__file__).resolve().parent;m=json.loads((a.manifest or root/'weights/manifest.json').read_text())
 if sha(a.weights)!=m['weights']['sha256']:raise ValueError('Unexpected checkpoint SHA-256')
 token=root/'tokenizer.json'
 if sha(token)!=m['tokenizer_sha256']:raise ValueError('Unexpected tokenizer SHA-256')
 c=torch.load(a.weights,map_location='cpu',weights_only=True)
 if c['tokenizer_sha256']!=m['tokenizer_sha256']:raise ValueError('Checkpoint/tokenizer mismatch')
 tok=Tokenizer.from_file(str(token));device=torch.device(a.device);model=MandiaS2S(S2SConfig(**c['config'])).to(device);model.load_state_dict(c['state_dict']);model.eval();del c
 if sum(x.numel() for x in model.parameters())!=m['parameters']:raise ValueError('Parameter count differs from manifest')
 cap=a.max_new_tokens if a.max_new_tokens is not None else m.get('max_new_tokens',96)
 if not 1<=cap<=model.cfg.context:raise ValueError('max-new-tokens must be between1 and the checkpoint context limit')
 if a.context and a.mode!='reply':raise ValueError('Context is only supported for reply mode')
 result=predict(model,tok,a.text,mode=a.mode,context=a.context,max_source=m.get('source_max',160),max_new_tokens=cap)
 print(json.dumps(result,ensure_ascii=False) if a.json else result['text_bci'])
 if not a.json and not result['all_chunks_ended_eos']:print('Incomplete output: generation limit reached.',file=sys.stderr)
if __name__=='__main__':main()
