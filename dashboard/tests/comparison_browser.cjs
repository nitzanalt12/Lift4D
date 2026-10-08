const assert=require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_EXECUTABLE,args:['--no-sandbox']});
 const page=await browser.newPage({viewport:{width:1600,height:1050}});page.setDefaultTimeout(180000);
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto(process.env.DASHBOARD_URL||'http://127.0.0.1:8766');
 await page.waitForFunction(()=>document.querySelector('#sequence').options.length===4);
 await page.selectOption('#sequence','camel');await page.selectOption('#experiment','2.1');
 await page.waitForFunction(()=>document.querySelector('#run').value.endsWith('camel-2.1')&&!document.querySelector('#render').hidden);
 await page.check('#compare');await page.selectOption('#experimentB','2.3');
 await page.waitForFunction(()=>document.querySelector('#viewB').value.endsWith('mesh-001000.json')&&!document.querySelector('#renderB').hidden);
 await page.waitForFunction(()=>[...document.querySelectorAll('#cards small')].every(x=>x.textContent.includes('90/90')));
 assert.equal(await page.locator('#panelB').isVisible(),true);assert.equal(await page.locator('#charts svg').count(),5);
 assert.ok(await page.locator('#charts svg polyline').count()>=10);
 assert.match(await page.locator('#cards').innerText(),/0.745/);assert.match(await page.locator('#cards').innerText(),/0.853/);
 // Deliver an obsolete request late; all three pictures must keep the same ID.
 await page.route('**/api/image?*',async route=>{if(new URL(route.request().url()).searchParams.get('frame')==='00004')await new Promise(r=>setTimeout(r,300));await route.continue()});
 await page.locator('#scrubber').evaluate(el=>{el.value=4;el.dispatchEvent(new Event('input'));el.value=30;el.dispatchEvent(new Event('input'))});
 await page.waitForFunction(()=>document.querySelector('#frameLabel').textContent.includes('00030'));
 await page.waitForTimeout(400);
 const matched=await page.evaluate(async()=>{
  function pixels(im){const c=document.createElement('canvas');c.width=im.naturalWidth;c.height=im.naturalHeight;c.getContext('2d').drawImage(im,0,0);return c.toDataURL()}
  for(const [image,run,view,kind] of [['input','run','view','input'],['render','run','view','render'],['renderB','runB','viewB','render']]){
   const params=new URLSearchParams({run:document.getElementById(run).value,view:document.getElementById(view).value,sequence:'camel',frame:'00030',object:255,kind});
   const url=URL.createObjectURL(await(await fetch('/api/image?'+params)).blob());const expected=new Image();expected.src=url;await expected.decode();
   const same=pixels(expected)===pixels(document.getElementById(image));URL.revokeObjectURL(url);if(!same)return false;
  }return true;
 });assert.equal(matched,true);
 await page.check('#overlay');await page.waitForFunction(()=>document.querySelector('#renderTitleB').textContent.includes('Overlay'));
 await page.click('#refresh');await page.waitForFunction(()=>[...document.querySelectorAll('#cards small')].every(x=>x.textContent.includes('90/90')));
 assert.equal(await page.locator('#experiment').inputValue(),'2.1');assert.equal(await page.locator('#experimentB').inputValue(),'2.3');
 await page.selectOption('#checkpointB','mesh_finetune:250');
 await page.waitForFunction(()=>document.querySelector('#renderMissingB').textContent.includes('Saved render is not available'));
 await page.waitForFunction(()=>[...document.querySelectorAll('#cards small')].every(x=>x.textContent.includes('0/90')));
 assert.equal(await page.locator('#render').isVisible(),true);
 await page.selectOption('#checkpointB','mesh_finetune:1000');
 await page.waitForFunction(()=>[...document.querySelectorAll('#cards small')].every(x=>x.textContent.includes('90/90')));
 await page.screenshot({path:'/tmp/lift4d-comparison-dashboard.png',fullPage:true});
 await page.uncheck('#compare');await page.waitForFunction(()=>!document.querySelector('#render').hidden&&[...document.querySelectorAll('#cards small')].every(x=>x.textContent.includes('90/90')));
 assert.equal(await page.locator('#panelB').isVisible(),false);assert.equal(await page.locator('#experiment').inputValue(),'2.1');
 assert.deepEqual(errors,[]);console.log('PASS: A/B metrics and shared coverage, two curves, three synchronized decoded images, delayed responses, overlay, refresh, missing B checkpoint, return to single mode');await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
