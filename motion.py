import os,sys,json,math,time,subprocess,bisect,functools,pathlib
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=pathlib.Path(__file__).resolve().parent;cv2.setNumThreads(1)
W,H,FPS=1920,1080,30
PAPER=(238,244,247);INK=(15,21,22);RUST=(60,98,200);GOLD=(58,179,232);WHITE=(244,247,250)
def ease(u):u=min(1,max(0,u));return 1-(1-u)**3
def spring(u):u=max(0,min(1,u));return (1-(1+7*u)*math.exp(-7*u))/(1-8*math.exp(-7))
@functools.lru_cache(maxsize=800)
def glyph(t,size=34,col=INK,width=800,weight='ExtraBold'):
 font=ImageFont.truetype(str(ROOT/'fonts'/f'Montserrat-{weight}.ttf'),size);d=ImageDraw.Draw(Image.new('RGBA',(1,1)));ls=[]
 for part in t.split('\n'):
  cur=''
  for word in part.split():
   z=(cur+' '+word).strip()
   if d.textlength(z,font=font)>width and cur:ls.append(cur);cur=word
   else:cur=z
  if cur:ls.append(cur)
 if not ls:ls=[' ']
 ht=round(size*1.24);ww=min(width,math.ceil(max(d.textlength(z,font=font) for z in ls)))+8;im=Image.new('RGBA',(ww,len(ls)*ht+12));d=ImageDraw.Draw(im)
 for i,z in enumerate(ls):d.text((3,i*ht-size*.12),z,font=font,fill=col[::-1]+(255,))
 return np.ascontiguousarray(np.asarray(im)[:,:,[2,1,0,3]])
def blit(f,im,x,y,alpha=1):
 if alpha<=0:return
 x,y=int(x),int(y);h,w=im.shape[:2];xa,ya=max(0,x),max(0,y);xb,yb=min(W,x+w),min(H,y+h)
 if xb<=xa or yb<=ya:return
 src=im[ya-y:yb-y,xa-x:xb-x];a=src[:,:,3:4].astype(np.float32)*(alpha/255);dst=f[ya:yb,xa:xb];dst[:]=(dst*(1-a)+src[:,:,:3]*a).astype(np.uint8)
def text(f,t,x,y,size=34,col=INK,width=800,alpha=1,weight='ExtraBold',center=False):
 a=glyph(t,size,col,width,weight);blit(f,a,x-a.shape[1]/2 if center else x,y,alpha);return a.shape[0]
def line(f,a,b,col=RUST,th=4,p=1):
 if p<=0:return
 a=np.array(a,float);b=a+(np.array(b,float)-a)*min(1,p);cv2.line(f,tuple(a.astype(int)),tuple(b.astype(int)),col,th,cv2.LINE_AA)
def path(f,pts,col=RUST,th=5,p=1,arrow=False):
 if p<=0:return
 pts=np.array(pts,float);lens=np.linalg.norm(np.diff(pts,axis=0),axis=1);left=lens.sum()*min(1,p);out=[pts[0]]
 for i,L in enumerate(lens):
  if left>=L:out.append(pts[i+1]);left-=L
  else:out.append(pts[i]+(pts[i+1]-pts[i])*(left/max(L,.001)));break
 out=np.array(out,np.int32);cv2.polylines(f,[out],False,col,th,cv2.LINE_AA)
 if arrow and len(out)>1 and p>.94:
  v=out[-1]-out[-2];v=v/max(np.linalg.norm(v),.001);n=np.array([-v[1],v[0]]);tip=out[-1];cv2.fillConvexPoly(f,np.array([tip,tip-v*20+n*10,tip-v*20-n*10],np.int32),col,cv2.LINE_AA)
def rect(f,x,y,w,h,col):
 x,y,w,h=map(int,(x,y,w,h));r=13;cv2.rectangle(f,(x+r,y),(x+w-r,y+h),col,-1);cv2.rectangle(f,(x,y+r),(x+w,y+h-r),col,-1)
 for cx,cy in [(x+r,y+r),(x+w-r,y+r),(x+r,y+h-r),(x+w-r,y+h-r)]:cv2.circle(f,(cx,cy),r,col,-1,cv2.LINE_AA)

VERTICAL=False;scenes=[];starts=[];VO=TOTAL=0;CH=json.load(open(ROOT/'chapters.json'));bg={}
def configure(target='longform'):
 global W,H,VERTICAL,scenes,starts,VO,TOTAL,bg
 VERTICAL=target!='longform';W,H=(1080,1920) if VERTICAL else (1920,1080)
 scenes=json.load(open(ROOT/(target+'-scene_plan.json' if VERTICAL else 'scene_plan.json')))
 tl=next(x for x in json.load(open(ROOT/'shorts-timeline.json')) if x['id']==target) if VERTICAL else json.load(open(ROOT/'timeline.json'))
 VO=tl['vo_end'];TOTAL=tl['total'];starts=[s['start'] for s in scenes]
 rng=np.random.default_rng(12);grain=rng.normal(0,.5,(H,W,1))
 bg={False:np.clip(np.array(PAPER)[None,None,:]+grain,0,255).astype(np.uint8),True:np.clip(np.array(INK)[None,None,:]+grain*.4,0,255).astype(np.uint8)}
 for dark in [False,True]:cv2.line(bg[dark],(85,181 if VERTICAL else 139),(W-85,181 if VERTICAL else 139),(45,49,50) if dark else (222,228,232),1,cv2.LINE_AA)
 hero.cache_clear()
@functools.lru_cache(maxsize=8)
def hero(art,feature=False):
 a=cv2.imread(str(ROOT/'assets/layers'/(art+'.png')),-1);h,w=a.shape[:2]
 mx,my=(820,500) if VERTICAL else ((560,470) if feature else (825,610))
 scale=min(mx/w,my/h);a=cv2.resize(a,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA)
 gray=cv2.cvtColor(a[:,:,:3],cv2.COLOR_BGR2GRAY);ink=a.copy();ink[:,:,:3]=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR);ink[:,:,3]=(a[:,:,3]*np.clip((230-gray)/90,0,1)).astype(np.uint8)
 # Split only at a genuinely empty vertical mask gap. Never through an object.
 den=(a[:,:,3]/255).mean(0);lo,hi=int(a.shape[1]*.22),int(a.shape[1]*.78);parts=[(a,0)]
 if hi>lo:
  cut=lo+int(np.argmin(den[lo:hi]));gap=den[max(0,cut-3):cut+4].max()
  if gap<.012 and min(cut,a.shape[1]-cut)>45:parts=[(a[:,:cut].copy(),0),(a[:,cut:].copy(),cut)]
 return a,ink,parts

def art_draw(f,s,t):
 a,ink,parts=hero(s['art'],s['feature']);h,w=a.shape[:2];r=t-s['start'];u=(t-s['start'])/max(.1,s['end']-s['start'])
 if VERTICAL:cx,cy=500,745
 elif s['feature']:cx,cy=380,635
 else:cx,cy=(1400,555) if s['layout']=='right' else (510,555)
 x,y=cx-w/2,cy-h/2;rev=ease(r/.68);fill=ease((r-.16)/.66);ent=spring(r/.72);direction=1 if s['layout']=='right' else -1
 for j,(part,off) in enumerate(parts):
  dx=(1-ent)*34*direction+(ease(u)-.5)*(9 if j==0 else -12);dy=(1-spring(max(0,r-j*.07)/.75))*(25 if j==0 else -17)+(ease(u)-.5)*(3 if j==0 else -4)
  cap=max(0,min(part.shape[1],int(w*rev)-off))
  if cap:
   blit(f,ink[:,off:off+cap],x+off+dx,y+dy,1-fill)
   blit(f,part[:,:cap],x+off+dx,y+dy,fill)

def circle(f,c,r,color,th=4,p=1):
 if p<=0:return
 cv2.ellipse(f,tuple(map(int,c)),(int(r),int(r)),-90,0,360*min(1,p),color,th,cv2.LINE_AA)
def cue_labels(s,t):return [(l,c,spring((t-c)/.45)) for l,c in zip(s['labels'],s['cues'])]
def label(f,l,x,y,size,col,width,a=1,center=False):
 if a>0:text(f,l,x,y,size,col,width,alpha=a,center=center)
def diagram(f,s,t,x,y,w,h):
 k=s['kind'];r=t-s['start'];D=s['end']-s['start'];dark=s['dark'];fg=WHITE if dark else INK;muted=(150,157,163) if dark else (114,119,123);soft=(38,43,47) if dark else (219,229,236);els=cue_labels(s,t);n=len(els);p=ease((r-.12)/.7);active=max([i for i,(_,c,a) in enumerate(els) if t>=c] or [-1]);font=32 if VERTICAL else 31
 if k in ('thought','focus'):
  gap=min(105,h/max(n,1))
  if k=='focus':circle(f,(x+43,y+55),30,RUST,4,p)
  for i,(l,c,a) in enumerate(els):
   yy=y+i*gap;off=94 if k=='focus' else 28
   line(f,(x,yy+8),(x,yy+59),RUST if i==active else muted,4,a)
   label(f,l,x+off+(1-a)*18,yy+8,35 if n<3 else font,RUST if i==active else fg,w-off-12,a)
 elif k in ('stack','feed'):
  rh=min(67,(h-20)/max(n,1));gap=12
  for i,(l,c,a) in enumerate(els):
   if a<=0:continue
   yy=y+i*(rh+gap);xx=x+18*(1-a);line(f,(xx,yy+rh-3),(xx+w-15,yy+rh-3),soft,2,a);label(f,f'{i+1:02}',xx,yy+8,25,RUST,70,a);label(f,l,xx+66,yy+5,font,fg,w-90,a);line(f,(xx+66,yy+rh-3),(xx+w-25,yy+rh-3),RUST,4,ease((t-c)/.65))
 elif k in ('split','balance','window','switch'):
  if k=='switch':
   a0=els[0][2];pos=ease((t-els[-1][1])/.8) if n>1 else ease((r-.35)/.8);xx=x+20;yy=y+23
   rect(f,xx,yy,260,82,soft);cv2.circle(f,(int(xx+44+172*pos),int(yy+41)),31,RUST,-1,cv2.LINE_AA)
   for i,(l,c,a) in enumerate(els[:3]):label(f,l,x+330,y+10+i*95,31,fg,w-340,a)
   line(f,(x+20,y+h-30),(x+w-20,y+h-30),RUST,4,p)
  else:
   line(f,(x+w/2,y+10),(x+w/2,y+h-20),muted,2,p)
   for i,(l,c,a) in enumerate(els[:2]):
    cx=x+w*(.25 if i==0 else .75);yy=y+40;label(f,l,cx,yy,30,fg,int(w*.45),a,True)
    if k=='window':
     side=75 if i==0 else 145;cv2.rectangle(f,(int(cx-side/2),int(y+145)),(int(cx+side/2*a),int(y+145+side)),RUST if i==0 else fg,3,cv2.LINE_AA)
    elif k=='balance':
     circle(f,(cx,y+179),40,RUST if i else fg,4,a);line(f,(cx-95,y+260),(cx+95,y+260),RUST if i else muted,5,a)
    else:
     circle(f,(cx,y+180),48,RUST if i else fg,4,a);line(f,(cx-65,y+265),(cx+65,y+265),RUST if i else fg,4,a)
 elif k=='loop':
  # Cue-by-cue causal nodes on a true oval. No arrow to an absent node.
  count=max(2,n);cx=x+w/2;cy=y+h/2;radx=w*.35;rady=h*.32;pts=[]
  for i in range(count):
   theta=-math.pi/2+i*2*math.pi/count;pts.append((cx+radx*math.cos(theta),cy+rady*math.sin(theta)))
  for i,(l,c,a) in enumerate(els):
   if a<=0:continue
   px,py=pts[i];cv2.circle(f,(int(px),int(py)),max(1,int(14*a)),RUST if i==active else fg,-1,cv2.LINE_AA)
   label(f,l,px,py+23,font-4,fg,min(285,int(w*.43)),a,True)
  for i in range(count):
   j=(i+1)%count
   if i<n and j<n and t>=els[j][1]+.2:
    a=np.array(pts[i]);b=np.array(pts[j]);v=(b-a)/max(1,np.linalg.norm(b-a));path(f,[a+v*24,b-v*24],RUST,4,ease((t-max(els[i][1],els[j][1])-.2)/.7),True)
 elif k in ('paths','stairs','timeline','finish'):
  # Marker follows cumulative distance on the exact drawn polyline.
  if k=='stairs':pts=[(x+20,y+225),(x+170,y+225),(x+170,y+175),(x+330,y+175),(x+330,y+125),(x+490,y+125),(x+490,y+75),(x+w-25,y+75)]
  elif k=='paths':pts=[(x+20,y+200),(x+w*.23,y+200),(x+w*.39,y+130),(x+w*.59,y+200),(x+w*.77,y+120),(x+w-20,y+120)]
  else:pts=[(x+20,y+180),(x+w-20,y+180)]
  progress=ease((r-.1)/min(1.8,max(.8,D*.3)));path(f,pts,RUST,5,progress,True)
  lens=np.linalg.norm(np.diff(np.array(pts),axis=0),axis=1);dist=lens.sum()*min(progress,.15+.75*ease(r/max(D,.1)));q=np.array(pts[0],float)
  for i,L in enumerate(lens):
   if dist>=L:dist-=L;q=np.array(pts[i+1],float)
   else:q=np.array(pts[i])+dist/max(L,.001)*(np.array(pts[i+1])-pts[i]);break
  if progress>.15:cv2.circle(f,tuple(q.astype(int)),10,GOLD,-1,cv2.LINE_AA)
  if active>=0:
   l,c,a=els[active];label(f,l,x+20,y+8,34,fg,w-40,a)
  for i,(l,c,a) in enumerate(els[:3]):
   xx=x+25+i*(w-60)/max(1,min(3,n));label(f,l,xx,y+h-42,24,muted,int(w/max(1,min(3,n)))-18,a)
 elif k in ('evidence','bridge'):
  count=5;yy=y+182;gap=12;bw=(w-75-gap*(count-1))/count
  # Five conceptual bricks, not an empirical measurement.
  for i in range(count):
   a=ease((r-.18-i*.15)/.7);xx=x+25+i*(bw+gap);rect(f,xx,yy+(1-a)*17,bw,max(1,28*a),RUST if i==count-1 else fg)
  line(f,(x+16,yy+52),(x+w-15,yy+52),muted,2,p)
  if active>=0:label(f,els[active][0],x+22,y+19,35,fg,w-40,els[active][2])
  label(f,'ILLUSTRATIVE: REPEATED FOLLOW-THROUGH',x+22,y+258,20,muted,w-44,p)
 elif k=='identity':
  a=ease((r-.15)/.9);cx=x+w/2;label(f,'BEHAVIOR',cx,y+18,28,muted,w,p,True);line(f,(cx,y+68),(cx,y+125),RUST,4,a)
  rect(f,cx-w*.39,y+151,w*.78,90,soft)
  if active>=0:label(f,els[active][0],cx,y+172,32,fg,int(w*.72),els[active][2],True)
  label(f,'A LABEL IS NOT A LIFE SENTENCE',cx,y+282,21,RUST,w,p,True)
 elif k=='calendar':
  for i in range(7):
   xx=x+18+i*(w-45)/7;a=ease((r-.1-i*.06)/.6);side=min(69,(w-75)/7);cv2.rectangle(f,(int(xx),int(y+133)),(int(xx+side),int(y+133+side)),muted,2,cv2.LINE_AA)
   if i!=3:line(f,(xx+13,y+169),(xx+25,y+183),RUST,4,a);line(f,(xx+25,y+183),(xx+side-10,y+150),RUST,4,a)
  if active>=0:label(f,els[active][0],x+20,y+20,33,fg,w-40,els[active][2])
  label(f,'ILLUSTRATIVE WEEK',x+20,y+262,20,muted,w-40,p)
 elif k=='clock':
  cx=x+115;cy=y+158;circle(f,(cx,cy),93,RUST,4,p);ang=-math.pi/2+ease(r/1.25)*math.pi*.55;line(f,(cx,cy),(cx+60*math.cos(ang),cy+60*math.sin(ang)),fg,5,p);line(f,(cx,cy),(cx,cy-74),fg,3,p)
  for i,(l,c,a) in enumerate(els[:3]):label(f,l,x+250,y+36+i*83,32,fg,w-270,a)
 elif k=='pages':
  for i in range(5):
   a=ease((r-.1-i*.13)/.6);xx=x+10+i*39;yy=y+70-i*8;cv2.rectangle(f,(int(xx),int(yy)),(int(xx+120),int(yy+161)),RUST if i==4 else muted,3,cv2.LINE_AA)
   for j in range(3):line(f,(xx+18,yy+40+j*27),(xx+95,yy+40+j*27),muted,2,a)
  if active>=0:label(f,els[active][0],x+340,y+92,40,fg,w-360,els[active][2])
 elif k=='iceberg':
  line(f,(x+15,y+115),(x+w-15,y+115),muted,2,p)
  for i in range(4):path(f,[(x+w/2,y+115),(x+w/2+(i-1.5)*55,y+174),(x+w/2+(i-1.5)*100,y+253)],RUST,3,ease((r-i*.12)/.9))
  if n:label(f,els[0][0],x+20,y+18,33,fg,w-40,els[0][2])
  if n>1:label(f,els[-1][0],x+20,y+275,28,RUST,w-40,els[-1][2])
  label(f,'VISIBLE / PRIVATE',x+w-235,y+86,19,muted,230,p)

def scene_frame(i,t):
 s=scenes[i];r=t-s['start'];f=bg[s['dark']].copy();fg=WHITE if s['dark'] else INK
 text(f,'WHY WE BECOME',85,113 if VERTICAL else 62,26,fg,450,weight='Black')
 header='SELF-TRUST' if VERTICAL else f'{s["chapter"]:02} / {CH[s["chapter"]-1].upper()}'
 text(f,header,640 if VERTICAL else 790,117 if VERTICAL else 65,21 if VERTICAL else 23,RUST,330 if VERTICAL else 1000)
 if VERTICAL:tx,ty,maxw=87,230,800;sz=84;limit=242
 else:tx,ty,maxw=(100 if s['feature'] or s['layout']=='right' else 1035),214,(1720 if s['feature'] else 805);sz=88 if s['feature'] else 82;limit=225
 title=s['title'].replace('\n',' ') if s['feature'] else s['title'];im=glyph(title,sz,fg,maxw,'Black')
 while im.shape[0]>limit and sz>54:sz-=2;im=glyph(title,sz,fg,maxw,'Black')
 ent=spring(r/.55);blit(f,im,tx,ty+(1-ent)*19,ent);line(f,(tx,490 if VERTICAL else 440),(tx+145,490 if VERTICAL else 440),RUST,4,ease((r-.16)/.6))
 art_draw(f,s,t)
 if VERTICAL:diagram(f,s,t,87,1060,790,300)
 else:diagram(f,s,t,680 if s['feature'] else tx,505,1120 if s['feature'] else 790,310)
 py=1620 if VERTICAL else 1025;line(f,(85,py),(W-85,py),(51,56,61) if s['dark'] else (216,225,231),2);line(f,(85,py),(85+(W-170)*min(1,t/TOTAL),py),RUST,3)
 return f
@functools.lru_cache(maxsize=2)
def endcard(vertical):
 im=cv2.imread(str(ROOT/'assets/subscribe-end-card.jpeg'));h,w=im.shape[:2]
 scale=min((930 if vertical else 1700)/w,(650 if vertical else 930)/h);return cv2.resize(im,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA)
def ending_frame(t):
 f=bg[False].copy();im=endcard(VERTICAL);h,w=im.shape[:2];y=(630 if VERTICAL else (H-h)//2)
 f[y:y+h,(W-w)//2:(W+w)//2]=im
 if VERTICAL:
  text(f,'WHY WE BECOME',87,220,60,INK,850,weight='Black');text(f,'Think deeper.\nLive better.\nBecome more.',87,1320,52,RUST,820)
 return f

def frame(t):
 if t>=VO:return ending_frame(t)
 i=max(0,bisect.bisect_right(starts,t)-1);s=scenes[i]
 # Brand artwork under the closing spoken channel tag, without adding master duration.
 if not VERTICAL and t>VO-2.5:
  a=ease((t-(VO-2.5))/.55);return cv2.addWeighted(scene_frame(i,t),1-a,ending_frame(t),a,0)
 f=scene_frame(i,t);r=t-s['start']
 if i and r<.22:
  prev=scene_frame(i-1,starts[i]-.02);a=ease(r/.22)
  if i%5==0 and scenes[i]['dark']==scenes[i-1]['dark']:
   b=int((H if VERTICAL else W)*a)
   if VERTICAL:f[b:]=prev[b:]
   else:f[:,b:]=prev[:,b:]
  else:f=cv2.addWeighted(prev,1-a,f,a,0)
 return f

def render(target,start=0,stop=None,out=None):
 configure(target);stop=min(TOTAL,stop if stop is not None else TOTAL);out=out or str(ROOT/'out'/(target+'.mp4'));caption=ROOT/(target+'-captions.ass' if VERTICAL else 'captions.ass');audio=ROOT/'assets'/(target+'-mix.wav' if VERTICAL else 'longform-mix.wav')
 vf=f'setpts=PTS+{start}/TB,ass={caption}:fontsdir={ROOT}/fonts,setpts=PTS-STARTPTS'
 cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r','30','-i','-','-ss',str(start),'-i',str(audio),'-vf',vf,'-c:v','libx264','-preset','ultrafast','-crf','20','-threads','2','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-t',str(stop-start),'-movflags','+faststart',out]
 p=subprocess.Popen(cmd,stdin=subprocess.PIPE);n0,n1=round(start*FPS),round(stop*FPS);t0=time.time()
 try:
  for n in range(n0,n1):
   p.stdin.write(frame(n/FPS).tobytes())
   if (n-n0)%300==0:print(f'{target} {n-n0}/{n1-n0} {time.time()-t0:.1f}s',flush=True)
 finally:p.stdin.close()
 rc=p.wait();print('DONE',out,rc,time.time()-t0,flush=True)
 if rc:raise RuntimeError('FFmpeg failed')
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--target',default='longform');ap.add_argument('--start',type=float,default=0);ap.add_argument('--stop',type=float);ap.add_argument('--out');ap.add_argument('--stills',action='store_true');args=ap.parse_args();configure(args.target)
 if args.stills:
  for i,s in enumerate(scenes):
   tt=min(s['end']-.05,s['start']+max(.7,(s['end']-s['start'])*.72));cv2.imwrite(str(ROOT/'qa'/f'{args.target}-{i:03}.png'),frame(tt))
 else:render(args.target,args.start,args.stop,args.out)
