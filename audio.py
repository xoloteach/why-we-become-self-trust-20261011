import pathlib,json,subprocess,numpy as np,wave
def gaussian_filter1d(a,sigma):
 x=np.arange(-int(sigma*3),int(sigma*3)+1);k=np.exp(-x*x/(2*sigma*sigma));k/=k.sum();return np.convolve(np.pad(a,(len(k)//2,len(k)//2),mode="edge"),k,mode="valid")
R=pathlib.Path(__file__).resolve().parent;SR=44100
jobs=[('longform',658.023855,0)]+[(s['id'],s['vo_end'],2.5) for s in json.load(open(R/'shorts-timeline.json'))]
for name,dur,pad in jobs:
 mp3=R/'assets'/('narration-master.mp3' if name=='longform' else name+'-narration.mp3');raw=subprocess.check_output(['ffmpeg','-loglevel','error','-i',str(mp3),'-f','f32le','-ac','1','-ar',str(SR),'-'])
 voice=np.frombuffer(raw,np.float32).copy();total=dur+pad;n=round(total*SR);v=np.zeros(n,np.float32);v[:min(n,len(voice))]=voice[:n]
 # Original, quiet ambient bed. No external recording or unlicensed music.
 t=np.arange(n,dtype=np.float32)/SR;music=np.zeros(n,np.float32);chords=[(130.81,164.81,196),(110,130.81,164.81),(146.83,174.61,220),(98,130.81,164.81)]
 for start in np.arange(0,total,8):
  a=int(start*SR);b=min(n,int((start+9)*SR));tt=t[a:b]-start;env=np.minimum(1,np.maximum(0,tt/1.1))*np.minimum(1,np.maximum(0,(9-tt)/1.5));freqs=chords[int(start/8)%4]
  for j,f in enumerate(freqs):music[a:b]+=(.0022*np.sin(2*np.pi*f*tt+j*.8)+.0007*np.sin(2*np.pi*f*2*tt))*env
 # Low-rate, smooth ducking: speech stays dominant.
 step=441;envelope=np.sqrt(np.mean(np.pad(v,(0,(-n)%step)).reshape(-1,step)**2,axis=1));smooth=gaussian_filter1d(envelope,9);duck=1-.65*np.clip(smooth/.06,0,1);du=np.interp(np.arange(n)/step,np.arange(len(duck)),duck);music*=du.astype(np.float32)
 music*=np.minimum(1,t/1.5)*np.minimum(1,(total-t)/1.5)
 effects=np.zeros(n,np.float32);scenes=json.load(open(R/('scene_plan.json' if name=='longform' else name+'-scene_plan.json')))
 for s in scenes:
  for cue in s['cues'][:3]:
   a=int(cue*SR);b=min(n,a+int(.075*SR));tt=np.arange(b-a)/SR;effects[a:b]+=.009*np.sin(2*np.pi*640*tt)*np.exp(-tt*62)
  if s['feature'] or s['dark']:
   a=int(s['start']*SR);b=min(n,a+int(.22*SR));tt=np.arange(b-a)/SR;effects[a:b]+=.014*np.sin(2*np.pi*(95-35*tt)*tt)*np.exp(-tt*18)
 mix=v+music+effects;path=R/'assets'/(name+'-mix-pre.wav')
 with wave.open(str(path),'wb') as f:f.setnchannels(1);f.setsampwidth(2);f.setframerate(SR);f.writeframes((np.clip(mix,-1,1)*32767).astype('<i2').tobytes())
 final=R/'assets'/(name+'-mix.wav');subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(path),'-af','loudnorm=I=-16:LRA=11:TP=-1','-ar',str(SR),'-ac','1',str(final)],check=True);path.unlink();print(name,'mixed',total,flush=True)
