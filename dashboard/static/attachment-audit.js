/* Only completed saved diagnostic buffers; no training or model calls. */
const $=id=>document.getElementById(id);
let manifest,canonical,labels,faces,auditId,shownNodes=[],generation=0,drawGeneration=0;
let plotQueue=Promise.resolve();
const cache=new Map();
async function json(url){const r=await fetch(url);if(!r.ok)throw Error(await r.text());return r.json()}
async function buffer(desc){
 const identity=auditId,key=identity+'/'+desc.file;
 if(!cache.has(key))cache.set(key,(async()=>{
  const r=await fetch('/api/attachment-asset?'+new URLSearchParams({audit:identity,file:desc.file}));if(!r.ok)throw Error(await r.text());
  const b=await r.arrayBuffer();const T={float32:Float32Array,int32:Int32Array,int8:Int8Array}[desc.dtype];
  const a=new T(b);if(a.length!==desc.shape.reduce((x,y)=>x*y,1))throw Error('Buffer shape mismatch');return a;
 })());
 return cache.get(key);
}
function xyz(values,ids=null){
 const n=ids?ids.length:values.length/3,x=new Float32Array(n),y=new Float32Array(n),z=new Float32Array(n);
 for(let i=0;i<n;i++){const j=(ids?ids[i]:i)*3;x[i]=values[j];y[i]=values[j+1];z[i]=values[j+2]}return {x,y,z};
}
function color(hex,intensity){const c=hex.match(/[0-9a-f]{2}/gi).map(x=>parseInt(x,16));return `rgb(${c.map(x=>Math.round(35+(x-35)*intensity)).join(',')})`}
function updateWeightChoices(){
 const layer=manifest.layers[$('layer').value];$('weights').querySelector('[value="surface"]').disabled=!layer.weights.surface;
 if(!layer.weights[$('weights').value])$('weights').value='original';
}
async function draw(){
 if(!manifest)return;
 const token=++drawGeneration,currentGeneration=generation;
 $('plot').dataset.ready='false';
 updateWeightChoices();
 const layerName=$('layer').value,mode=$('weights').value,layer=manifest.layers[layerName],mapping=layer.weights[mode];
 const pose=$('pose').value,row=manifest.frames.find(x=>x.id===pose),selection=Number($('node').value),colorMode=$('colorMode').value;
 if(!Number.isInteger(selection)||selection<0||selection>=layer.nodes.shape[0])throw Error('Invalid node ID');
 const [indices,weights,cross,nodeXYZ,mesh,motion]=await Promise.all([
  buffer(mapping.indices),buffer(mapping.values),buffer(mapping.cross_mass),buffer(layer.nodes),
  row?buffer(row[mode==='surface'?'surface':'original']):Promise.resolve(canonical),
  row?buffer(row[layerName+'_motion']):Promise.resolve(null)
 ]);
 if(token!==drawGeneration||currentGeneration!==generation)return;
 const nodes=new Float32Array(nodeXYZ);if(motion)for(let i=0;i<nodes.length;i++)nodes[i]+=motion[i];
 const colors=new Array(labels.length),influence=new Float32Array(labels.length),regionMass=new Float64Array(manifest.regions.length),k=mapping.indices.shape[1],crossByNode=new Float64Array(layer.nodes.shape[0]);
 for(let v=0;v<labels.length;v++){
  for(let j=0;j<k;j++){
   const ni=indices[v*k+j],w=weights[v*k+j];if(ni===selection)influence[v]+=w;
   if(labels[v]>=2&&layer.attached_regions[ni]>=2&&labels[v]!==layer.attached_regions[ni])crossByNode[ni]+=w;
  }
  regionMass[labels[v]]+=influence[v];
  colors[v]=colorMode==='cross'?`rgb(${Math.round(35+220*cross[v])},35,35)`:color(manifest.colors[labels[v]],colorMode==='influence'?.12+.88*influence[v]:1);
 }
 const filter=$('nodeFilter').value;
 shownNodes=Array.from({length:layer.nodes.shape[0]},(_,i)=>i).filter(i=>filter==='all'||filter==='legs'&&layer.attached_regions[i]>=2||filter==='disputed'&&layer.disputed_nodes.includes(i));
 if(!shownNodes.includes(selection))shownNodes.push(selection);
 const nodeRegions=$('nodeColors').value==='owner'?layer.gaussian_owner_regions:layer.attached_regions;
 const anchors=await buffer(layer.anchors);if(token!==drawGeneration||currentGeneration!==generation)return;
 const anchor=anchors[selection];
 const meshTrace={type:'mesh3d',...xyz(mesh),i:faces.i,j:faces.j,k:faces.k,vertexcolor:colors,opacity:Number($('meshOpacity').value),flatshading:false,hoverinfo:'skip',name:'Fixed mesh regions',lighting:{ambient:.8,diffuse:.35,specular:0},showlegend:false};
 const pointTrace={type:'scatter3d',...xyz(nodes,shownNodes),mode:'markers',name:'Control nodes',customdata:shownNodes,
  text:shownNodes.map(i=>`Node ${i} · attachment: ${manifest.regions[layer.attached_regions[i]]} · learned receiver: ${manifest.regions[layer.gaussian_owner_regions[i]]}`),hoverinfo:'text',
  marker:{size:shownNodes.map(i=>i===selection?8:4),color:shownNodes.map(i=>i===selection?'#ffffff':manifest.colors[nodeRegions[i]]),opacity:1}};
 const link={type:'scatter3d',x:[nodes[3*selection],mesh[3*anchor]],y:[nodes[3*selection+1],mesh[3*anchor+1]],z:[nodes[3*selection+2],mesh[3*anchor+2]],mode:'lines+markers',line:{color:'#ffffff',width:6},marker:{size:5,color:'#ffffff'},name:'Selected attachment',hoverinfo:'skip'};
 plotQueue=plotQueue.catch(()=>{}).then(()=>{
 if(token!==drawGeneration||currentGeneration!==generation)return;
 return Plotly.react('plot',[meshTrace,pointTrace,link],{paper_bgcolor:'#0e1727',font:{color:'#b3bac6'},margin:{l:0,r:0,b:0,t:10},uirevision:auditId,
  scene:{aspectmode:'data',xaxis:{title:'X'},yaxis:{title:'Y'},zaxis:{title:'Z'},camera:{eye:{x:1.8,y:.4,z:.65},up:{x:0,y:0,z:1}}}},{responsive:true,displaylogo:false});
 });await plotQueue;
 if(token!==drawGeneration||currentGeneration!==generation)return;
 const total=regionMass.reduce((a,b)=>a+b,0),gm=layer.gaussian_region_mass[selection],gt=gm.reduce((a,b)=>a+b,0);
 $('nodeDetails').textContent=`Node ${selection} · attached to ${manifest.regions[layer.attached_regions[selection]]} at canonical vertex ${anchor}.\nGaussian receiver: ${manifest.regions[layer.gaussian_owner_regions[selection]]}; dominant-region purity ${(100*layer.gaussian_owner_confidence[selection]).toFixed(1)}%, support ${gt.toFixed(2)}. Mixed/body/low-support nodes remain unclassified.\n${layer.motion_description}. Mesh coordinates are saved ${pose==='canonical'?'canonical':mode==='surface'?'2.5':'2.1'} coordinates.`;
 $('nodeTable').replaceChildren();
 manifest.regions.forEach((name,r)=>{const tr=document.createElement('tr');[name,total?(100*regionMass[r]/total).toFixed(1)+'%':'Not available · no mesh influence',gt?(100*gm[r]/gt).toFixed(1)+'%':'Not available · no support'].forEach(x=>{const td=document.createElement('td');td.textContent=x;tr.append(td)});$('nodeTable').append(tr)});
 const summary=manifest.summary.layers[layerName],stat=summary[mode];
 $('summary').textContent=`${summary.nodes_attached_to_leg_cores}/${summary.node_count} nodes attach to the four leg cores. ${summary.high_purity_gaussian_leg_owners} nodes have a high-purity original Gaussian leg receiver; ${summary.attachment_owner_disagreements} have a conflicting leg attachment. Mean other-leg weight: ${(100*stat.mean_other_leg_mass).toFixed(2)}%. `+stat.by_leg.map(x=>`${x.region}: ${(100*x.mean_other_leg_mass).toFixed(2)}%`).join(' · ');
 const ranked=Array.from(crossByNode,(_,i)=>i).filter(i=>crossByNode[i]>.01).sort((a,b)=>crossByNode[b]-crossByNode[a]).slice(0,30);
 $('crossNodes').replaceChildren(new Option(ranked.length?'Select node · ranked by cross-leg mass':'None for these weights',''),...ranked.map(i=>new Option(`Node ${i} · ${manifest.regions[layer.attached_regions[i]]}`,i)));
 $('status').textContent=`Saved ${manifest.sequence} · ${pose} · ${layerName} / ${mode} · ${shownNodes.length} nodes displayed. Colors keep canonical vertex identities. No mapping modified.`;
 $('plot').dataset.frame=pose;$('plot').dataset.layer=layerName;$('plot').dataset.weights=mode;$('plot').dataset.ready='true';
}
async function load(){
 const token=++generation;++drawGeneration;$('plot').dataset.ready='false';auditId=$('audit').value;
 const data=await json('/api/attachment-audit?'+new URLSearchParams({audit:auditId}));if(token!==generation)return;manifest=data;
 const [c,f,l]=await Promise.all([buffer(manifest.geometry.canonical),buffer(manifest.geometry.faces),buffer(manifest.geometry.labels)]);if(token!==generation)return;
 canonical=c;labels=l;const triangles=f.length/3;faces={i:new Int32Array(triangles),j:new Int32Array(triangles),k:new Int32Array(triangles)};
 for(let n=0;n<triangles;n++){faces.i[n]=f[3*n];faces.j[n]=f[3*n+1];faces.k[n]=f[3*n+2]}
 $('pose').replaceChildren(new Option('Canonical mesh','canonical'),...manifest.frames.map(row=>new Option('Frame '+row.id,row.id)));
 $('regions').replaceChildren(...manifest.regions.map((name,i)=>{const span=document.createElement('span');span.textContent='● '+name;span.style.color=manifest.colors[i];return span}));
 $('provenance').textContent=JSON.stringify(manifest,null,2);$('node').max=manifest.layers.base.nodes.shape[0]-1;
 const start=manifest.layers.base.attached_regions.findIndex(r=>r===5);$('node').value=Math.max(0,start);await draw();
}
function safe(fn){return ()=>fn().catch(e=>{$('status').textContent='Not available · '+e.message})}
for(const id of ['pose','layer','weights','colorMode','nodeFilter','nodeColors','node','meshOpacity'])$(id).addEventListener('change',safe(draw));
$('crossNodes').onchange=()=>{if($('crossNodes').value!==''){$('node').value=$('crossNodes').value;$('colorMode').value='influence';safe(draw)()}};
$('audit').addEventListener('change',safe(load));
$('resetCamera').onclick=()=>Plotly.relayout('plot',{'scene.camera':{eye:{x:1.8,y:.4,z:.65},up:{x:0,y:0,z:1}}});
safe(async()=>{
 const listing=await json('/api/attachment-audits');if(!listing.audits.length)throw Error('No completed attachment diagnostic export');
 $('audit').replaceChildren(...listing.audits.map(x=>new Option(x.sequence+' · '+x.created_utc,x.id)));$('audit').value=listing.audits.at(-1).id;
 await load();$('plot').on('plotly_click',event=>{const point=event.points[0];if(point.curveNumber===1){$('node').value=point.customdata;safe(draw)()}});
})();
