// Shared pure pairing utilities: metric rows are joined by frame ID, never position.
function pairedMetric(key,frames,a,b,proof){
 const aa=new Map((a?.frames||[]).map(r=>[r.id,r])),bb=new Map((b?.frames||[]).map(r=>[r.id,r]));
 const valid=new Set((proof?.frames||[]).filter(r=>r.paired_verified).map(r=>r.id));
 const rows=frames.filter(f=>valid.has(f.id)).map(f=>[aa.get(f.id)?.[key],bb.get(f.id)?.[key]])
  .filter(([x,y])=>typeof x==='number'&&Number.isFinite(x)&&typeof y==='number'&&Number.isFinite(y));
 if(!rows.length)return {a:null,b:null,delta:null,count:0};
 const av=rows.reduce((s,r)=>s+r[0],0)/rows.length,bv=rows.reduce((s,r)=>s+r[1],0)/rows.length;
 return {a:av,b:bv,delta:bv-av,count:rows.length};
}
if(typeof module!=='undefined')module.exports={pairedMetric};
