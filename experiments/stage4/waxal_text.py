"""Extraction de colonnes texte WAXAL, sans télécharger les octets audio.
Le manifeste garde les références des shards pour une future expérience vocale.
"""
import io,json,pathlib,requests,hashlib,time
import pyarrow.parquet as pq
REVISION='5f4d8ca24f2b9d168b2ee545f1febaaff4b40580'

class RemoteRanges(io.RawIOBase):
    def __init__(self,url):
        self.url=url;self.pos=0;self.transferred=0;self.session=requests.Session()
        r=self.session.get(url,headers={'Range':'bytes=0-3'},timeout=90)
        if r.status_code!=206:raise RuntimeError('Le serveur ne permet pas une lecture partielle ; téléchargement audio refusé.')
        self.size=int(r.headers['Content-Range'].split('/')[-1]);self.transferred+=len(r.content)
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        return self.pos
    def read(self,n=-1):
        if n<0:n=self.size-self.pos
        n=min(n,self.size-self.pos)
        if n<=0:return b''
        for retry in range(4):
            r=self.session.get(self.url,headers={'Range':f'bytes={self.pos}-{self.pos+n-1}'},timeout=90)
            if r.status_code==206:break
            if retry==3:r.raise_for_status();raise RuntimeError('Range ignoré.')
            time.sleep(1+retry)
        data=r.content
        if len(data)!=n:raise RuntimeError('Lecture partielle incorrecte.')
        self.transferred+=n;self.pos+=n;return data
    def readinto(self,b):
        data=self.read(len(b));b[:len(data)]=data;return len(data)

def extract(directory):
    directory=pathlib.Path(directory);directory.mkdir(parents=True,exist_ok=True)
    destination=directory/'waxal-baoule-text.jsonl';receipt=directory/'WAXAL-TEXTE.json'
    if destination.exists() and receipt.exists():
        old=json.loads(receipt.read_text())
        if hashlib.sha256(destination.read_bytes()).hexdigest()==old['sha256']:return destination
    meta=requests.get(f'https://huggingface.co/api/datasets/google/WaxalNLP/revision/{REVISION}',timeout=60)
    meta.raise_for_status();meta=meta.json();assert meta['sha']==REVISION
    files=sorted(x['rfilename'] for x in meta['siblings'] if x['rfilename'].startswith('data/TTS/bau/bau-train-') and x['rfilename'].endswith('.parquet'))
    assert files
    rows=[];shards=[]
    for filename in files:
        url=f'https://huggingface.co/datasets/google/WaxalNLP/resolve/{REVISION}/{filename}'
        remote=RemoteRanges(url)
        table=pq.read_table(remote,columns=['id','text','speaker_id','locale'],pre_buffer=False)
        for row in table.to_pylist():
            assert row['locale']=='bau',row
            rows.append(dict(row,source_file=filename,source_dataset='google/WaxalNLP',revision=REVISION,original_split='train',license='CC-BY-4.0',audio_downloaded=False))
        shards.append({'file':filename,'rows':len(table),'remote_bytes':remote.size,'transferred_bytes':remote.transferred})
        print('WAXAL_TEXTE',filename,len(table),remote.transferred,flush=True)
    part=destination.with_suffix('.part');part.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));part.replace(destination)
    receipt.write_text(json.dumps({'source':'https://huggingface.co/datasets/google/WaxalNLP','revision':REVISION,'provider':'University of Ghana, Google WAXAL','config':'bau_tts (Baoule in publisher card)','license':'CC-BY-4.0','rows':len(rows),'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'shards':shards,'audio_downloaded':False,'usage':'Baoulé denoising training; not French/BCI parallel examples','note':'Only official training split; text exact overlaps with all parallel holdouts excluded before training.'},ensure_ascii=False,indent=2))
    return destination
