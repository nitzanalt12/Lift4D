const assert=require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_EXECUTABLE,args:['--no-sandbox']});
 const page=await browser.newPage({viewport:{width:1200,height:1000}});page.setDefaultTimeout(120000);
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 const definition=process.env.ANCHOR_DEFINITION;
 await page.goto((process.env.DASHBOARD_URL||'http://127.0.0.1:8766')+'/video-anchors'+(definition?'?definition='+encodeURIComponent(definition):''));
 await page.waitForFunction(()=>document.querySelector('#inputCanvas').dataset.ready==='true');
 assert.match(await page.locator('#draft').innerText(),/assistant-draft/);
 if(definition){
  assert.equal(await page.locator('#definition').inputValue(),definition);
  await page.selectOption('#handle','bend-hind-near');
  assert.match(await page.locator('#caption').innerText(),/surface|bend/);
  assert.equal(await page.locator('#handle option').count(),6);
 }
 await page.selectOption('#handle','foot-5');assert.equal(await page.locator('#identity').isChecked(),false);
 const canvas=page.locator('#inputCanvas'),box=await canvas.boundingBox();const target=[302.5,420.25];
 await page.mouse.click(box.x+box.width*target[0]/854,box.y+box.height*target[1]/480);
 assert.equal(await page.locator('#identity').isChecked(),false);
 await page.click('#save');await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('verified identity'));
 await page.click('#hidden');assert.match(await page.locator('#rows').innerText(),/Hidden \/ uncertain/);
 await page.selectOption('#frame','00020');
 await page.waitForFunction(()=>document.querySelector('#inputCanvas').dataset.ready==='true'&&document.querySelector('#inputCanvas').dataset.frame==='00020');
 await page.selectOption('#handle','foot-4');await page.uncheck('#identity');
 const b=await canvas.boundingBox();await page.mouse.click(b.x+b.width*target[0]/854,b.y+b.height*target[1]/480);
 assert.equal(await page.locator('#identity').isChecked(),false);await page.check('#identity');
 // Intercept accepted save to avoid creating browser-test labels in real runs.
 let saved;
 await page.route('**/api/video-annotations',async route=>{saved=route.request().postDataJSON();await route.fulfill({contentType:'application/json',body:JSON.stringify({absolute_path:'/test-only/labels.json',note:'Test save intercepted; no real labels written'})})});
 await page.click('#save');await page.waitForFunction(()=>document.querySelector('#savedPath').textContent.includes('/test-only/labels.json'));
 const row=saved.annotations.observations.find(x=>x.frame_id==='00020'&&x.handle_id==='foot-4');
 assert.ok(Math.abs(row.xy[0]-target[0])<1);assert.ok(Math.abs(row.xy[1]-target[1])<1);assert.equal(row.identity_verified,true);
 assert.equal(saved.annotations.review_status,'manual-edit-draft');
 await page.reload();await page.waitForFunction(()=>document.querySelector('#inputCanvas').dataset.ready==='true');
 await page.screenshot({path:'/tmp/lift4d-video-anchors.png',fullPage:true});
 assert.deepEqual(errors,[]);console.log('PASS: real draft labels, native-pixel clicks under display scaling, no automatic identity confirmation, rejected uncertain targets, hidden points, frame switching and save payload');await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
