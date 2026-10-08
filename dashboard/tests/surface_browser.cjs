// Integration check on the completed real camel surface-transfer artifacts.
const assert=require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_EXECUTABLE,args:['--no-sandbox']});
 const page=await browser.newPage({viewport:{width:1600,height:1050}});page.setDefaultTimeout(180000);
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto(process.env.DASHBOARD_URL||'http://127.0.0.1:8765');
 await page.waitForFunction(()=>[...document.querySelector('#sequence').options].some(x=>x.value==='camel'));
 await page.selectOption('#sequence','camel');await page.selectOption('#experiment','2.1');
 await page.check('#compare');await page.selectOption('#experimentB','2.5');
 await page.waitForFunction(()=>document.querySelector('#runB').value.includes('camel-2.5-')&&[...document.querySelectorAll('#cards small')].every(x=>x.textContent.includes('90/90'))&&document.querySelectorAll('#cards small').length===5);
 assert.equal(await page.locator('#checkpointB').inputValue(),'surface_transfer:30000');
 await page.locator('#scrubber').evaluate(el=>{el.value=30;el.dispatchEvent(new Event('input'))});
 await page.waitForFunction(()=>document.querySelector('#frameLabel').textContent.includes('00030'));
 await page.waitForFunction(()=>!document.querySelector('#renderB').hidden&&document.querySelector('#renderB').naturalWidth===854);
 // Compare decoded displayed pixels to independently requested frame-ID images.
 await page.waitForFunction(async()=>{
  function pixels(im){const c=document.createElement('canvas');c.width=im.naturalWidth;c.height=im.naturalHeight;c.getContext('2d').drawImage(im,0,0);return c.toDataURL()}
  for(const [image,run,view,kind] of [['input','run','view','input'],['render','run','view','render'],['renderB','runB','viewB','render']]){
   const params=new URLSearchParams({run:document.getElementById(run).value,view:document.getElementById(view).value,sequence:'camel',frame:'00030',object:255,kind});
   const url=URL.createObjectURL(await(await fetch('/api/image?'+params)).blob());const expected=new Image();expected.src=url;await expected.decode();
   const same=pixels(expected)===pixels(document.getElementById(image));URL.revokeObjectURL(url);if(!same)return false;
  }return true;
 });
 assert.equal(await page.locator('#charts svg').count(),5);
 await page.screenshot({path:'/tmp/lift4d-surface-comparison-dashboard.png',fullPage:true});
 await page.check('#overlay');await page.waitForFunction(()=>document.querySelector('#renderTitleB').textContent.includes('Overlay'));
 assert.deepEqual(errors,[]);
 console.log('PASS: real completed 2.5 selection, 90 paired frames, original source checkpoint, five graphs, three decoded images synchronized by frame ID, overlay');
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
