// Optional browser check: PLAYWRIGHT_MODULE=/path/to/playwright package node this-file.
const assert = require('node:assert/strict');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
(async () => {
  const browser = await chromium.launch({headless:true,
    ...(process.env.CHROMIUM_EXECUTABLE ? {executablePath:process.env.CHROMIUM_EXECUTABLE}:{}),
    args:['--no-sandbox']});
  const page = await browser.newPage({viewport:{width:1400,height:1000}});
  page.setDefaultTimeout(240000);
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(process.env.DASHBOARD_URL || 'http://127.0.0.1:8765');
  await page.selectOption('#run','dashboard-demo/DEMO-synthetic');
  await page.waitForFunction(()=>document.querySelector('#render').naturalWidth===160);
  assert.equal(await page.locator('#demo').isVisible(),true);
  await page.click('#evaluate');
  await page.waitForFunction(()=>document.querySelectorAll('#worst button').length>0);
  assert.match(await page.locator('#worst button').first().innerText(),/00007/);
  await page.locator('#worst button').first().click();
  await page.waitForFunction(()=>document.querySelector('#frameLabel').textContent.includes('00007'));
  // Deliberately deliver old frame requests late to exercise synchronized commits.
  await page.route('**/api/image?*',async route=>{
    if(new URL(route.request().url()).searchParams.get('frame')==='00004')
      await new Promise(resolve=>setTimeout(resolve,300));
    await route.continue();
  });
  await page.locator('#scrubber').evaluate(el=>{el.value=4;el.dispatchEvent(new Event('input'));el.value=7;el.dispatchEvent(new Event('input'))});
  await page.waitForTimeout(500);
  assert.match(await page.locator('#frameLabel').innerText(),/00007/);
  const pixelsMatch = await page.evaluate(async()=>{
    const dataURL=img=>{const c=document.createElement('canvas');c.width=img.naturalWidth;c.height=img.naturalHeight;c.getContext('2d').drawImage(img,0,0);return c.toDataURL()};
    for(const id of ['input','render']){
      const q=new URLSearchParams({run:document.querySelector('#run').value,sequence:'synthetic',view:'dashboard_exports/aligned.json',frame:'00007',kind:id,object:1});
      const url=URL.createObjectURL(await(await fetch('/api/image?'+q)).blob());
      const expected=new Image();expected.src=url;await expected.decode();
      const match=dataURL(expected)===dataURL(document.getElementById(id));URL.revokeObjectURL(url);
      if(!match)return false;
    }return true;
  });
  assert.equal(pixelsMatch,true);
  await page.check('#overlay');await page.waitForTimeout(300);
  await page.locator('#blend').evaluate(el=>{el.value=.25;el.dispatchEvent(new Event('input'))});
  await page.waitForTimeout(300);
  await page.click('#play');await page.waitForTimeout(300);await page.click('#play');
  assert.equal(await page.locator('#play').innerText(),'▶ Play');
  const stopped=await page.locator('#frameLabel').innerText();await page.waitForTimeout(300);
  assert.equal(await page.locator('#frameLabel').innerText(),stopped);
  await page.click('#refresh');await page.waitForTimeout(300);
  assert.equal(await page.locator('#run').inputValue(),'dashboard-demo/DEMO-synthetic');
  await page.selectOption('#run','baseline/animals-rhino-20261008T120751Z');
  await page.waitForFunction(()=>document.querySelector('#frameLabel').textContent.includes('/90'));
  await page.waitForFunction(()=>[...document.querySelectorAll('#cards small')].every(x=>x.textContent.includes('90/90')), null, {timeout:240000});
  assert.equal(await page.locator('#demo').isVisible(),false);
  assert.equal(await page.locator('#object').inputValue(),'255');
  assert.equal(await page.locator('#renderMissing').isVisible(),false);
  assert.match(await page.locator('#view').inputValue(),/dashboard_exports/);
  assert.equal(await page.locator('#charts svg').count(),5);
  await page.check('#overlay');await page.waitForTimeout(300);
  assert.match(await page.locator('#renderTitle').innerText(),/Overlay/);
  assert.deepEqual(errors,[]);
  console.log('PASS: demo and verified real final run with five metrics, sync under delayed responses, overlay, playback/pause, refresh, worst-frame navigation');
  await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
