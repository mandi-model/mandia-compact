"""Load the verified own Mandia inference checkpoint; no remote code execution."""
import argparse,pathlib,sys,hashlib,json
sys.path.insert(0,str(pathlib.Path(__file__).parent/'src'))
import torch
from tokenizers import Tokenizer
from mandia_seq2seq import MandiaS2S,S2SConfig,generate

def main():
 p=argparse.ArgumentParser();p.add_argument('--weights',type=pathlib.Path,required=True);p.add_argument('--text',required=True);p.add_argument('--mode',choices=['reply','translate'],default='reply');p.add_argument('--device',default='cpu');a=p.parse_args()
 root=pathlib.Path(__file__).parent;manifest=json.loads((root/'weights/manifest.json').read_text())
 assert hashlib.sha256(a.weights.read_bytes()).hexdigest()==manifest['weights']['sha256'],'Unexpected checkpoint'
 token=root/'tokenizer.json';assert hashlib.sha256(token.read_bytes()).hexdigest()==manifest['tokenizer_sha256']
 checkpoint=torch.load(a.weights,map_location='cpu',weights_only=True);assert checkpoint['tokenizer_sha256']==manifest['tokenizer_sha256']
 tok=Tokenizer.from_file(str(token));ids=[4 if a.mode=='reply' else 3]+tok.encode(a.text).ids+[2]
 if len(ids)>160:raise ValueError('Entrée trop longue : maximum 160 tokens avec préfixe et EOS')
 device=torch.device(a.device);model=MandiaS2S(S2SConfig(**checkpoint['config'])).to(device);model.load_state_dict(checkpoint['state_dict']);model.eval()
 with torch.inference_mode():result,_=generate(model,ids,1,2,max_tokens=96)
 print(tok.decode(result))
if __name__=='__main__':main()
