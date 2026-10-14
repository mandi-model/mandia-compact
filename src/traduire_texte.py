"""Inference helpers for the private Mandia checkpoint, no silent input loss.
Segmentation extends technical input length, not a quality guarantee.
"""
import re,contextlib
def split_french(text,tokenizer,max_source=512):
 text=text.strip()
 if not text:return []
 def fits(s):return len(tokenizer.encode(s).ids)+2<=max_source
 if fits(text):return [text]
 pieces=re.split(r'(?<=[.!?…])\s+|\n+',text);chunks=[];buffer=''
 for piece in pieces:
  piece=piece.strip()
  if not piece:continue
  if fits(piece):units=[piece]
  else:units=piece.split()
  for unit in units:
   if not fits(unit):raise ValueError('Un élément du texte dépasse à lui seul la taille acceptée ; découpez cet élément.')
   candidate=(buffer+' '+unit).strip()
   if fits(candidate):buffer=candidate
   else:
    chunks.append(buffer);buffer=unit
 if buffer:chunks.append(buffer)
 assert all(fits(c) for c in chunks)
 assert ' '.join(text.split())==' '.join(' '.join(chunks).split()),'Segmentation lost or changed input words'
 return chunks
def predict(model,tokenizer,text,mode='translate',context='',max_source=512,max_new_tokens=None):
 import torch
 from mandia_seq2seq import generate
 if mode not in ['translate','reply']:raise ValueError('Unknown task')
 if not text.strip():raise ValueError('Le texte est vide.')
 prompts=split_french(text,tokenizer,max_source) if mode=='translate' else [f'Contexte : {context}\nMessage : {text}' if context.strip() else text]
 results=[];device=next(model.parameters()).device
 for prompt in prompts:
  source=[3 if mode=='translate' else 4]+tokenizer.encode(prompt).ids+[2]
  if len(source)>max_source:raise ValueError('Le contexte du dialogue est trop long ; raccourcissez-le.')
  precision=torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else contextlib.nullcontext()
  with torch.inference_mode(),precision:ids,_=generate(model,source,1,2,max_new_tokens or model.cfg.context)
  results.append({'source_fr':prompt,'source_tokens':len(source),'baoule':tokenizer.decode(ids),'generated_tokens':len(ids),'ended_eos':2 in ids})
 return {'text_bci':' '.join(x['baoule'] for x in results),'chunks':results,'all_chunks_ended_eos':all(x['ended_eos'] for x in results),'all_input_processed':True,'native_validated':False}
