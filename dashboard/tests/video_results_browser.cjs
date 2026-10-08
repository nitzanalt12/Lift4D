const assert=require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_EXECUTABLE,args:['--no-sandbox']});
 const page=await browser.newPage({viewport:{width:1600,height:1050}});page.setDefaultTimeout(120000);
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 const base='sam3d_mesh_video_anchors/camel-bends-20261009/camel-2.6-anchors-';
 await page.goto(process.env.DASHBOARD_URL||'http://127.0.0.1:8765');
 await page.waitForFunction(()=>document.querySelector('#sequence').options.length===4);
 await page.selectOption('#sequence','camel');await page.selectOption('#experiment','2.6');
 await page.waitForFunction(()=>[...document.querySelector('#run').options].some(o=>o.value.includes('camel-bends-20261009')));
 await page.selectOption('#run',base+'0');
 await page.waitForFunction(()=>document.querySelector('#selectionSummary').textContent.includes('draft anchors'));
 await page.check('#compare');await page.selectOption('#experimentB','2.6');await page.selectOption('#runB',base+'10');
 await page.waitForFunction(()=>[...document.querySelectorAll('#cards small')].every(x=>x.textContent.includes('3/90')));
 assert.equal(await page.locator('#charts svg').count(),5);
 await page.locator('#scrubber').evaluate(el=>{el.value=20;el.dispatchEvent(new Event('input'))});
 await page.waitForFunction(()=>document.querySelector('#frameLabel').textContent.includes('00020')&&!document.querySelector('#render').hidden&&!document.querySelector('#renderB').hidden);
 const matched=await page.evaluate(async()=>{
  function pixels(im){const c=document.createElement('canvas');c.width=im.naturalWidth;c.height=im.naturalHeight;c.getContext('2d').drawImage(im,0,0);return c.toDataURL()}
  for(const [im,run,view,kind] of [['input','run','view','input'],['render','run','view','render'],['renderB','runB','viewB','render']]){
   const q=new URLSearchParams({run:document.getElementById(run).value,view:document.getElementById(view).value,sequence:'camel',frame:'00020',object:255,kind});
   const url=URL.createObjectURL(await(await fetch('/api/image?'+q)).blob());const image=new Image();image.src=url;await image.decode();
   const same=pixels(image)===pixels(document.getElementById(im));URL.revokeObjectURL(url);if(!same)return false;
  }return true;
 });assert.equal(matched,true);
 await page.screenshot({path:'/tmp/lift4d-video-anchor-results.png',fullPage:true});
 await page.locator('#scrubber').evaluate(el=>{el.value=10;el.dispatchEvent(new Event('input'))});
 await page.waitForFunction(()=>document.querySelector('#frameLabel').textContent.includes('00010')&&document.querySelector('#render').hidden&&document.querySelector('#renderB').hidden);
 assert.equal(await page.locator('#input').isVisible(),true);
 assert.match(await page.locator('#renderMissing').innerText(),/not available/i);
 assert.match(await page.locator('#renderMissingB').innerText(),/not available/i);
 assert.equal(await page.locator('#worst button').count(),3);
 await page.locator('#worst button').first().click();
 await page.waitForFunction(()=>!document.querySelector('#render').hidden&&!document.querySelector('#renderB').hidden);
 assert.deepEqual(errors,[]);console.log('PASS: draft/partial labels, 3/90 paired metrics, exact decoded input/render IDs, missing frames, navigation to available problematic frames');await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
