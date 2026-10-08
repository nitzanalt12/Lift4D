const $=id=>document.getElementById(id);
let prepared,annotations,definitionId,inputImage,token=0,loadedFrame=null;
async function json(url,options){const r=await fetch(url,options);const data=await r.json();if(!r.ok)throw Error(data.error||'Request failed');return data}
function current(){return prepared.frames.find(f=>f.id===$('frame').value)}
function handle(){return prepared.handles.find(h=>h.id===$('handle').value)}
function observation(fid,hid){return annotations.observations.find(r=>r.frame_id===fid&&r.handle_id===hid)}
function ensure(){
 const f=current(),h=handle();let row=observation(f.id,h.id);
 if(!row){row={frame_id:f.id,handle_id:h.id,image_sha256:f.image_sha256,visible:false,identity_verified:false,xy:null};annotations.observations.push(row)}
 return row;
}
function draw(){
 if(!inputImage||loadedFrame!==$('frame').value)return;
 const f=current(),canvas=$('inputCanvas'),ctx=canvas.getContext('2d');ctx.drawImage(inputImage,0,0);
 for(const [i,h] of prepared.handles.entries()){
  const r=observation(f.id,h.id);
  if($('raw').checked){const [x,y]=f.original_handle_pixels[i];ctx.strokeStyle=h.color;ctx.setLineDash([3,3]);ctx.strokeRect(x-7,y-7,14,14);ctx.setLineDash([])}
  if(r?.visible&&r.xy){const [x,y]=r.xy;ctx.fillStyle=h.color;ctx.strokeStyle='#ffffff';ctx.beginPath();ctx.arc(x,y,h.id===handle().id?7:5,0,2*Math.PI);ctx.fill();ctx.stroke();ctx.fillText(h.id,x+10,y-10)}
 }
 $('rows').replaceChildren();
 for(const h of prepared.handles){const r=observation(f.id,h.id),tr=document.createElement('tr');
  [h.label,r?.xy?r.xy.map(x=>x.toFixed(1)).join(', '):'Not available',r?.visible?'Visible':'Hidden / uncertain',r?.visible?String(r.confidence):'—',r?.identity_verified?'Yes':'No'].forEach(text=>{const td=document.createElement('td');td.textContent=text;tr.append(td)});
  tr.style.color=h.color;tr.onclick=()=>{$('handle').value=h.id;selectHandle()};$('rows').append(tr);
 }
 $('status').textContent=`${prepared.sequence} · ${f.id} · original ${canvas.width} × ${canvas.height} pixels · fixed foot patch identities`;
 canvas.dataset.frame=f.id;canvas.dataset.ready='true';
}
function selectHandle(){const r=ensure();$('identity').checked=r.identity_verified===true;$('confidence').value=r.confidence||.8;draw()}
async function frame(){
 const n=++token,f=current();loadedFrame=null;$('inputCanvas').dataset.ready='false';
 const r=await fetch('/api/video-anchor-image?'+new URLSearchParams({definition:definitionId,frame:f.id}));if(!r.ok)throw Error(await r.text());
 const url=URL.createObjectURL(await r.blob());const image=new Image();image.src=url;await image.decode();URL.revokeObjectURL(url);
 if(n!==token)return;
 if(image.naturalWidth!==f.camera.width||image.naturalHeight!==f.camera.height)throw Error('Input resolution differs; no silent resize');
 inputImage=image;loadedFrame=f.id;$('inputCanvas').width=image.naturalWidth;$('inputCanvas').height=image.naturalHeight;selectHandle();
}
async function load(){
 definitionId=$('definition').value;++token;
 const [d,initial]=await Promise.all([json('/api/video-anchor-definition?'+new URLSearchParams({definition:definitionId})),json('/api/video-anchor-initial?'+new URLSearchParams({definition:definitionId}))]);
 prepared=d;annotations=structuredClone(initial.annotations);annotations.observations||=[];
 $('frame').replaceChildren(...d.frames.map(f=>new Option(f.id,f.id)));$('handle').replaceChildren(...d.handles.map(h=>new Option(h.label,h.id)));
 $('handle').value='foot-4';$('draft').textContent=`Loaded ${annotations.review_status||'unreviewed'} labels. Review target positions and fixed identities; no hidden target is interpolated.`;
 $('bindingImage').src='/api/video-anchor-image?'+new URLSearchParams({definition:definitionId,frame:d.frames[0].id,proposal:1});
 $('provenance').textContent=JSON.stringify(d,null,2);await frame();
}
function safe(fn){return ()=>fn().catch(e=>{$('status').textContent='Not available · '+e.message})}
$('inputCanvas').onclick=event=>{
 if(!prepared||loadedFrame!==$('frame').value)return;
 const canvas=$('inputCanvas'),rect=canvas.getBoundingClientRect(),row=ensure();
 row.xy=[(event.clientX-rect.left)*canvas.width/rect.width,(event.clientY-rect.top)*canvas.height/rect.height];
 row.visible=true;row.confidence=Number($('confidence').value);row.identity_verified=$('identity').checked;
 row.position_status='Manual click on original input image';draw();
};
$('identity').onchange=()=>{ensure().identity_verified=$('identity').checked;draw()};
$('confidence').onchange=()=>{ensure().confidence=Number($('confidence').value);draw()};
$('hidden').onclick=()=>{const row=ensure();row.visible=false;row.xy=null;row.identity_verified=false;row.reason='Manually marked hidden or identity uncertain';selectHandle()};
$('raw').onchange=draw;$('handle').onchange=selectHandle;$('frame').onchange=safe(frame);$('definition').onchange=safe(load);
$('save').onclick=safe(async()=>{
 const result=structuredClone(annotations);result.annotator=$('annotator').value.trim();result.review_status=$('review').value;
 result.annotation_source='Manual reviewed/edited original-image clicks in local video-anchor UI; fixed prepared mesh bindings';
 result.parent_annotation_source=annotations.annotation_source;result.definition_sha256=prepared.sha256;
 const saved=await json('/api/video-annotations',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({definition:definitionId,annotations:result})});
 $('savedPath').textContent=saved.absolute_path+'\n'+saved.note;
});
safe(async()=>{const list=await json('/api/video-anchor-definitions');if(!list.definitions.length)throw Error('No prepared handle definitions');$('definition').replaceChildren(...list.definitions.map(d=>new Option(d.sequence+' · '+d.id,d.id)));await load()})();
