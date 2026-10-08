const $=id=>document.getElementById(id);
const names={iou:'Mask IoU',boundary_mean:'Boundary mean · px',boundary_p95:'Boundary p95 · px',psnr:'Object PSNR · dB',lpips:'Object LPIPS'};
let frames=[],metrics=null,detail=null,timer=null,generation=0,drawToken=0,displayedIndex=0;
async function api(path,body){const r=await fetch(path,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});const data=await r.json();if(!r.ok)throw Error(data.error);return data}
function query(extra={}){return new URLSearchParams({run:$('run').value,sequence:$('sequence').value,view:$('view').value,...extra}).toString()}
function choices(el,items,old){el.replaceChildren();for(const item of items){const o=document.createElement('option');o.value=item.id??item;o.textContent=item.label??item;el.append(o)}if([...el.options].some(o=>o.value===old))el.value=old}
function stop(){if(timer)$('scrubber').value=displayedIndex;drawToken++;clearTimeout(timer);timer=null;$('play').textContent='▶ Play'}
function fmt(v){return v===null||v===undefined?'לא זמין · N/A':v==='Infinity'?'∞':Number(v).toFixed(3)}
function resetMetrics(){metrics=null;cards();$('charts').replaceChildren();$('worst').replaceChildren();$('evalStatus').textContent='Metrics require verified input-camera renders, matching IDs/times and exact resolutions.'}
function cards(){ $('cards').replaceChildren();for(const [k,n] of Object.entries(names)){const s=metrics?.summary[k];const d=document.createElement('div');d.className='card';const label=document.createElement('span');label.textContent=n;const b=document.createElement('b');b.textContent=fmt(s?.mean);const small=document.createElement('small');small.textContent=s?`${s.available_frames}/${frames.length} frames${s.perfect_frames?' · perfect excluded from finite mean':''}`:'Not computed';if(s&&k==='boundary_p95')small.textContent+=' · mean of frame p95s';d.append(label,b,small);$('cards').append(d)}}
let catalog=[];
function uniqueEntries(entries,key,label){return [...new Map(entries.map(e=>[e[key],{id:e[key],label:e[label]}])).values()]}
async function refresh(){
 stop();const gen=++generation;resetMetrics();try{
  const data=await api('/api/catalog?auxiliary='+($('auxiliary').checked?'1':'0'));if(gen!==generation)return;
  catalog=data.entries;choices($('sequence'),uniqueEntries(catalog,'animal','animal_label'),$('sequence').value);
  if(!catalog.length){clearSelection('No experiment results found in '+data.root);return}
  await selectAnimal();
 }catch(e){$('status').textContent=e.message}
}
function clearSelection(reason){
 stop();drawToken++;frames=[];detail=null;resetMetrics();$('input').hidden=true;$('render').hidden=true;
 $('renderMissing').hidden=false;$('renderMissing').textContent=reason;$('status').textContent=reason;
 $('frameLabel').textContent='';$('selectionSummary').textContent='';$('demo').hidden=true;
 for(const id of ['experiment','checkpoint','run'])choices($(id),[]);$('view').value='';$('runChoice').hidden=true;
}
async function selectAnimal(){
 stop();generation++;drawToken++;resetMetrics();
 const entries=catalog.filter(e=>e.animal===$('sequence').value);
 choices($('experiment'),uniqueEntries(entries,'experiment','experiment_label'),$('experiment').value);
 await selectExperiment();
}
async function selectExperiment(){
 stop();generation++;drawToken++;resetMetrics();
 const entries=catalog.filter(e=>e.animal===$('sequence').value&&e.experiment===$('experiment').value);
 choices($('checkpoint'),uniqueEntries(entries,'checkpoint','checkpoint_label'),$('checkpoint').value);
 // Default to latest available scientific checkpoint; preserve explicit choice on refresh.
 if(!entries.some(e=>e.checkpoint===lastCheckpoint&&e.experiment===lastExperiment&&e.animal===lastAnimal)){
  const latest=[...entries].sort((a,b)=>(b.iteration??-1)-(a.iteration??-1)||Number(Boolean(b.complete))-Number(Boolean(a.complete)))[0];
  if(latest)$('checkpoint').value=latest.checkpoint;
 }
 await selectCheckpoint();
}
let lastCheckpoint='',lastExperiment='',lastAnimal='';
async function selectCheckpoint(){
 stop();generation++;drawToken++;resetMetrics();
 const entries=catalog.filter(e=>e.animal===$('sequence').value&&e.experiment===$('experiment').value&&e.checkpoint===$('checkpoint').value);
 const executions=[...entries].sort((a,b)=>(b.created_utc||'').localeCompare(a.created_utc||'')||b.run.localeCompare(a.run));
 const old=$('run').value;
 choices($('run'),executions.map(e=>({id:e.run,label:(e.created_utc?new Date(e.created_utc).toLocaleString():'Execution')+' · '+e.run.split('/').at(-1)})),old);
 $('runChoice').hidden=executions.length<=1;
 lastCheckpoint=$('checkpoint').value;lastExperiment=$('experiment').value;lastAnimal=$('sequence').value;
 await loadRun();
}
async function loadRun(){
 stop();drawToken++;const gen=++generation;resetMetrics();$('input').hidden=true;$('render').hidden=true;$('frameLabel').textContent='';$('renderMissing').hidden=false;$('renderMissing').textContent='Loading selected result…';
 const entry=catalog.find(e=>e.animal===$('sequence').value&&e.experiment===$('experiment').value&&e.checkpoint===$('checkpoint').value&&e.run===$('run').value);
 if(!entry){clearSelection('No saved result for this selection');return}
 $('view').value=entry.view;
 $('selectionSummary').textContent=[entry.animal_label,entry.experiment_label,entry.checkpoint_label,entry.demo?'DEMO':entry.kind==='manifest'?'RGB + alpha':entry.kind==='missing'?'Render unavailable':'RGB comparison · alpha unavailable'].join(' / ');
 try{const data=await api('/api/run?'+query());if(gen!==generation)return;detail=data;$('demo').hidden=!data.demo;await loadFrames()}catch(e){$('status').textContent=e.message}
}
async function loadFrames(){stop();$('evaluate').disabled=false;drawToken++;const gen=++generation;resetMetrics();try{const data=await api('/api/frames?'+query());if(gen!==generation)return;frames=data.frames;const suggested=data.manifest?.target_object_id??(data.target_mask_values?.length===1?data.target_mask_values[0]:null);if(suggested!==null)$('object').value=suggested;$('scrubber').max=Math.max(0,frames.length-1);$('scrubber').value=Math.min(Number($('scrubber').value),frames.length-1);$('details').textContent=JSON.stringify({run:detail,view:data.view,export:data.manifest},null,2);await showFrame();if(gen===generation&&data.manifest?.complete&&data.view?.kind==='manifest')$('evaluate').click()}catch(e){frames=[];$('status').textContent=e.message}}
async function showFrame(){
 const index=Number($('scrubber').value),f=frames[index];if(!f)return;const token=++drawToken;
 const overlay=$('overlay').checked;
 const params={frame:f.id,object:$('object').value,threshold:$('threshold').value,blend:$('blend').value};
 async function load(kind){try{const response=await fetch('/api/image?'+query({...params,kind}));if(!response.ok){const x=await response.json();throw Error(x.error)}const url=URL.createObjectURL(await response.blob());const img=new Image();img.src=url;try{await img.decode()}catch(e){URL.revokeObjectURL(url);throw e}return {url}}catch(e){return {error:e.message}}}
 const pictures=await Promise.all([load('input'),load(overlay?'overlay':'render')]);
 if(token!==drawToken){pictures.forEach(p=>p.url&&URL.revokeObjectURL(p.url));return}
 // Commit both decoded images and labels together; slow requests cannot mix frames.
 displayedIndex=index;
 ['input','render'].forEach((id,i)=>{const im=$(id),p=pictures[i];if(im.dataset.url)URL.revokeObjectURL(im.dataset.url);delete im.dataset.url;im.hidden=Boolean(p.error);if(p.url){im.dataset.url=p.url;im.src=p.url}if(id==='render'){$('renderMissing').hidden=!p.error;$('renderMissing').textContent=p.error?'לא זמין · '+p.error:''}});
 $('frameLabel').textContent=`Frame ${f.id} · ${index+1}/${frames.length}${f.time!==null?' · '+f.time.toFixed(3)+' s':' · timestamp unavailable'}`;
 $('renderTitle').textContent=overlay?'Overlay · cyan target / pink render':'Saved render';
 $('status').textContent=f.reason||'Verified input-camera mapping. Evaluation checks exact image dimensions.';
 drawCharts();if(metrics){const row=metrics.frames.find(x=>x.id===f.id);$('evalStatus').textContent=JSON.stringify(row?.reasons??{})+(metrics.lpips_note?' · '+metrics.lpips_note:'')}
}

function drawCharts(){ $('charts').replaceChildren();if(!metrics)return;for(const [key,label] of Object.entries(names)){const d=document.createElement('div');d.className='chart';const title=document.createElement('span');title.textContent=label;d.append(title);const values=metrics.frames.map(r=>typeof r[key]==='number'?r[key]:null);const good=values.filter(v=>v!==null);if(!good.length){const p=document.createElement('p');p.textContent='לא זמין · No finite verified measurements';d.append(p)}else{const lo=Math.min(...good),hi=Math.max(...good),n=Math.max(1,values.length-1);const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 600 130');const x=i=>20+i/n*560,y=v=>110-(v-lo)/(hi-lo||1)*90;let points=[];function segment(){if(!points.length)return;const line=document.createElementNS(svg.namespaceURI,'polyline');line.setAttribute('points',points.join(' '));line.setAttribute('fill','none');line.setAttribute('stroke','#72ddb7');line.setAttribute('stroke-width','2');svg.append(line);points=[]}values.forEach((v,i)=>{if(v===null)segment();else points.push(`${x(i)},${y(v)}`)});segment();const cursor=document.createElementNS(svg.namespaceURI,'line');const cx=x(Number($('scrubber').value));for(const [k,v] of Object.entries({x1:cx,x2:cx,y1:10,y2:115,stroke:'#ffcc80'}))cursor.setAttribute(k,v);svg.append(cursor);svg.onclick=e=>{const r=svg.getBoundingClientRect();$('scrubber').value=Math.round(Math.max(0,Math.min(1,((e.clientX-r.left)/r.width*600-20)/560))*n);showFrame()};d.append(svg);const p=document.createElement('p');p.textContent=`Range ${fmt(lo)}–${fmt(hi)} · x: explicit frame IDs in input inventory · click to seek`;d.append(p)}$('charts').append(d)}}
function worst(){ $('worst').replaceChildren();if(!metrics)return;const k=$('metric').value;const low=['iou','psnr'].includes(k);const ranked=metrics.frames.filter(r=>typeof r[k]==='number').sort((a,b)=>low?a[k]-b[k]:b[k]-a[k]).slice(0,10);for(const r of ranked){const b=document.createElement('button');b.textContent=`${r.id} · ${fmt(r[k])}`;b.onclick=()=>{$('scrubber').value=frames.findIndex(f=>f.id===r.id);showFrame()};$('worst').append(b)}}
$('evaluate').onclick=async()=>{const gen=generation;$('evaluate').disabled=true;$('evalStatus').textContent='Evaluating saved pixels on CPU / reading cache…';try{const result=await api('/api/evaluate',{run:$('run').value,sequence:$('sequence').value,view:$('view').value,object:$('object').value,threshold:$('threshold').value,lpips:$('lpips').checked});if(gen!==generation)return;metrics=result;cards();drawCharts();worst();$('details').textContent=JSON.stringify({run:detail,evaluation:metrics},null,2);await showFrame()}catch(e){$('evalStatus').textContent=e.message}finally{if(gen===generation)$('evaluate').disabled=false}};
$('refresh').onclick=refresh;$('auxiliary').onchange=refresh;$('run').onchange=loadRun;$('sequence').onchange=selectAnimal;$('experiment').onchange=selectExperiment;$('checkpoint').onchange=selectCheckpoint;$('scrubber').oninput=showFrame;$('overlay').onchange=showFrame;$('blend').oninput=showFrame;$('metric').onchange=worst;
for(const id of ['object','threshold','lpips'])$(id).onchange=()=>{generation++;resetMetrics();showFrame()};
$('play').onclick=()=>{if(timer){stop();return}if(!frames.length)return;$('play').textContent='Ⅱ Pause';const fps=Math.max(1,Math.min(60,Number($('fps').value)||8));async function tick(){if(!timer)return;const i=Number($('scrubber').value);if(i>=frames.length-1){stop();return}$('scrubber').value=i+1;await showFrame();if(timer)timer=setTimeout(tick,1000/fps)}timer=setTimeout(tick,1000/fps)};
refresh();
