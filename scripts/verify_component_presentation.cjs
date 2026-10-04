#!/usr/bin/env node
'use strict';

// Read-only browser QA of one portable presentation. Evidence and an isolated
// renamed copy are written only to the explicitly supplied scratch directory.
// No experiment, renderer, model, Docker, or network client is invoked.
const fs = require('node:fs/promises');
const path = require('node:path');
const crypto = require('node:crypto');
const {pathToFileURL} = require('node:url');

const OBJECTIVES = ['tracking_mse','tracking_3d','reconstruction_3d','confidence','joint_training'];
const MEDIA_IDS = ['po_rgb','po_tracking','po_recon_tracking','dr_tracking'];
const METRICS = ['tracking_apd_drop_pp','reconstruction_apd_drop_pp','tracking_epe_increase_m','reconstruction_epe_increase_m'];
const LOSS_EXPRESSIONS = [
  'mean ||s_med P1 - G1||^2','mean(c1_eff * e1)','mean(C2 * e2)',
  '-0.2 * (mean log max(c1_eff,1) + mean log C2)',
  'tracking_3d + reconstruction_3d + confidence'
];
const DISPLAY_EXPRESSIONS = [
  'mean ‖smedP₁ − G₁‖₂²','mean(c₁ × e₁)','mean(C₂ × e₂)',
  '−0.2[mean log max(c₁,1) + mean log C₂]',
  'Tracking 기하 + Reconstruction 기하 + Confidence'
];
const sha = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const currentUtc = () => new Date().toISOString();
const check = (condition, message) => {if(!condition) throw new Error(message);};
const finite = value => typeof value === 'number' && Number.isFinite(value);
const close = (actual, expected, message, atol=1e-6) => check(finite(actual)&&finite(expected)&&Math.abs(actual-expected)<=atol, message+`: ${actual} vs ${expected}`);
const normalizeText = text => String(text).replace(/\s+/g,' ').trim();

function argumentsFrom(argv) {
  const allowed = new Set(['presentation','qa-dir','playwright-module','browser-executable']);
  const result = {};
  for(let i=0;i<argv.length;i++) {
    if(argv[i]==='--help'||argv[i]==='-h') return {help:true};
    check(argv[i].startsWith('--')&&allowed.has(argv[i].slice(2)), `Unknown argument ${argv[i]}`);
    const key=argv[i].slice(2);
    check(!(key in result)&&i+1<argv.length&&!argv[i+1].startsWith('--'), `Missing or repeated argument ${argv[i]}`);
    result[key]=argv[++i];
  }
  for(const key of allowed) check(key in result&&path.isAbsolute(result[key]), `--${key} must be an absolute path`);
  return result;
}

function validateData(data) {
  check(data&&typeof data==='object', 'Presentation data JSON is missing');
  const scope=data.scope||{};
  check(scope.kind==='full_campaign'&&scope.clips===8&&scope.num_frames===128&&scope.steps===20&&scope.epsilon_255===4
    &&scope.expected_conditions===48&&scope.all_frames===true, 'Presentation scope is not the completed 48-condition / 8-clip / all128-frame experiment');
  check(JSON.stringify(scope.dataset_counts)===JSON.stringify({po_mini:4,ds_mini:4})
    ||scope.dataset_counts?.po_mini===4&&scope.dataset_counts?.ds_mini===4&&Object.keys(scope.dataset_counts).length===2,
    'Presentation must contain PO4 and DR4');
  check(JSON.stringify(scope.objectives)===JSON.stringify(OBJECTIVES), 'Five attack objectives or their order differ');
  check(data.sources&&typeof data.sources==='object'&&Object.keys(data.sources).length>0, 'Presentation source provenance is absent');
  check(Array.isArray(data.aggregates), 'Presentation aggregates must be an array');
  const aggregates=data.aggregates.filter(row=>row.dataset===undefined||row.dataset==='all');
  check(aggregates.length===5&&new Set(aggregates.map(r=>r.objective)).size===5
    &&OBJECTIVES.every(o=>aggregates.some(r=>r.objective===o)), 'Five unique overall objective aggregates are required');
  for(const row of aggregates) for(const metric of METRICS) check(finite(row[metric]), `Nonfinite aggregate ${row.objective}/${metric}`);
  check(Array.isArray(data.losses)&&data.losses.length===5, 'Five loss definitions are required');
  for(const [i,loss] of data.losses.entries()) check(loss.id===OBJECTIVES[i]&&loss.expression===LOSS_EXPRESSIONS[i],
    'Loss IDs, expression schema, or fixed order differ');
  const contrasts=Array.isArray(data.contrasts)?data.contrasts:data.contrasts?.estimates;
  check(Array.isArray(contrasts), 'Direct contrasts are missing');
  const overall=contrasts.filter(row=>row.dataset===undefined||row.dataset==='all');
  check(overall.length===4&&new Set(overall.map(r=>r.metric)).size===4, 'Four overall paired direct contrasts are required');
  const a=aggregates.find(r=>r.objective==='tracking_3d'), b=aggregates.find(r=>r.objective==='reconstruction_3d');
  for(const row of overall) {
    check(METRICS.includes(row.metric)&&Array.isArray(row.ci95)&&row.ci95.length===2&&row.ci95.every(finite)&&row.ci95[0]<=row.ci95[1], 'Malformed paired contrast CI');
    close(row.estimate,a[row.metric]-b[row.metric], `Direct contrast does not use tracking_3d minus reconstruction_3d for ${row.metric}`);
    check((row.paired_clips??row.common_clips)===8, 'Direct contrast support is not eight clips');
    check(row.unit===(row.metric.includes('apd')?'percentage_points':'metres'), 'Contrast units differ from APD percentage points / EPE meters');
  }
  check(data.media&&typeof data.media==='object'&&JSON.stringify(Object.keys(data.media).sort())===JSON.stringify([...MEDIA_IDS].sort()), 'The four representative media IDs differ');
  for(const id of MEDIA_IDS) {
    const media=data.media[id],probe=media.probe||{};
    check(typeof media.path==='string'&&media.path.length>0&&/^[a-f0-9]{64}$/.test(media.sha256), `Media source path/SHA is absent: ${id}`);
    check(probe.decoded_frames===128&&probe.codec==='h264'&&media.cpu_only===true,
      `Media lacks CPU ffprobe decoded128-frame proof: ${id}`);
    check(/^[a-f0-9]{64}$/.test(media.recipe?.source_sha256||'')
      &&media.recipe.source_sha256===data.sources.selected_assets?.[media.path],
      `Encoded video recipe is not bound to the verified original report asset: ${id}`);
    close(probe.fps,10, `Video is not a 10fps preview: ${id}`);
    close(probe.duration,12.8, `Video duration is not 128/10 seconds: ${id}`,0.05);
    check(Number.isInteger(probe.width)&&probe.width>0&&Number.isInteger(probe.height)&&probe.height>0, `Media dimensions are invalid: ${id}`);
  }
  return {aggregates,contrasts:overall};
}

async function waitImages(page) {
  await page.evaluate(()=>document.fonts.ready);
  await page.waitForFunction(()=>Array.from(document.images).every(img=>img.complete&&img.naturalWidth>0),{},{timeout:30000});
  const images=await page.locator('img').evaluateAll(nodes=>nodes.map(img=>({src:img.getAttribute('src'),width:img.naturalWidth,height:img.naturalHeight})));
  check(images.length>0&&images.every(img=>/^data:image\//.test(img.src||'')&&img.width>0&&img.height>0), 'All presentation images must be loaded inline data:image assets');
  return {count:images.length,all_inline_loaded:true};
}

async function slideState(page,index) {
  const state=await page.evaluate(()=>({currentIndex:window.presentation.currentIndex,count:window.presentation.count,
    active:Array.from(document.querySelectorAll('section.slide.active')).map(e=>e.id),
    visible:Array.from(document.querySelectorAll('section.slide')).filter(e=>{
      const s=getComputedStyle(e),r=e.getBoundingClientRect();return s.display!=='none'&&s.visibility!=='hidden'&&Number(s.opacity)!==0&&r.width>0&&r.height>0;
    }).map(e=>e.id)}));
  check(state.count===12&&state.currentIndex===index&&JSON.stringify(state.active)===JSON.stringify([`slide-${index+1}`])
    &&JSON.stringify(state.visible)===JSON.stringify(state.active), `Slide visibility/API index differs at ${index+1}`);
  return state;
}

async function settleLayout(page) {
  // Wait for real layout/opacity stability across animation frames, including
  // viewport and fullscreen resize handlers, without disabling the product UI.
  await page.evaluate(()=>new Promise((resolve,reject)=>{
    let previous=null,stable=0;
    const started=performance.now();
    const tick=()=>{
      const deck=document.getElementById('deck'),active=document.querySelector('section.slide.active');
      const values=[innerWidth,innerHeight];
      for(const el of [deck,active]){const r=el.getBoundingClientRect(),s=getComputedStyle(el);values.push(r.x,r.y,r.width,r.height,Number(s.opacity));}
      const snapshot=JSON.stringify(values);
      stable=snapshot===previous?stable+1:0;previous=snapshot;
      if(stable>=4)return resolve();
      if(performance.now()-started>5000)return reject(new Error('Presentation layout did not stabilize'));
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }));
}

async function goTo(page,index) {
  await page.evaluate(i=>window.presentation.goTo(i),index);
  await page.waitForFunction(i=>window.presentation.currentIndex===i,index,{timeout:10000});
  await settleLayout(page);
  return slideState(page,index);
}

async function layout(page) {
  await settleLayout(page);
  const result=await page.evaluate(()=>{
    const deck=document.getElementById('deck'),active=document.querySelector('section.slide.active'),r=deck.getBoundingClientRect(),style=getComputedStyle(deck);
    const rect=el=>{const b=el.getBoundingClientRect();return {left:b.left,top:b.top,right:b.right,bottom:b.bottom,width:b.width,height:b.height};};
    const allowed=el=>Boolean(el.closest('[data-overlap-allowed="true"]'))||el.closest('svg')!==null;
    const targets=Array.from(active.querySelectorAll('h1,h2,h3,p,table,video,img,figure,ul,ol,.card,.chart,.metric-card,.panel'))
      .filter(el=>{const s=getComputedStyle(el),b=el.getBoundingClientRect();return s.display!=='none'&&s.visibility!=='hidden'&&Number(s.opacity)!==0&&b.width>0&&b.height>0;});
    const outside=[],overlaps=[];
    for(const el of targets){const b=rect(el);if(!allowed(el)&&(b.left<r.left-2||b.right>r.right+2||b.top<r.top-2||b.bottom>r.bottom+2))
      outside.push({tag:el.tagName,id:el.id,text:(el.textContent||'').trim().slice(0,100),rect:b});}
    for(let i=0;i<targets.length;i++)for(let j=i+1;j<targets.length;j++){
      const a=targets[i],b=targets[j];if(a.contains(b)||b.contains(a)||allowed(a)||allowed(b))continue;
      const ar=rect(a),br=rect(b),w=Math.min(ar.right,br.right)-Math.max(ar.left,br.left),h=Math.min(ar.bottom,br.bottom)-Math.max(ar.top,br.top);
      if(w>2&&h>2&&w*h>64)overlaps.push({first:{tag:a.tagName,id:a.id,text:(a.textContent||'').trim().slice(0,90)},second:{tag:b.tagName,id:b.id,text:(b.textContent||'').trim().slice(0,90)},overlap_width:w,overlap_height:h});
    }
    return {viewport:{width:innerWidth,height:innerHeight},deck:rect(deck),logical_width:parseFloat(style.width),logical_height:parseFloat(style.height),
      slide:active.id,title:(active.querySelector('h1,h2')?.textContent||'').trim(),outside,overlaps,
      body_scroll_width:document.documentElement.scrollWidth,body_scroll_height:document.documentElement.scrollHeight};
  });
  close(result.logical_width,1600,'Deck logical width must remain 1600 CSS pixels');
  close(result.logical_height,900,'Deck logical height must remain 900 CSS pixels');
  check(result.deck.width>0&&result.deck.height>0&&result.deck.left>=-2&&result.deck.top>=-2
    &&result.deck.right<=result.viewport.width+2&&result.deck.bottom<=result.viewport.height+2, `Deck does not fit viewport at ${result.slide}`);
  close(result.deck.width/result.deck.height,1600/900,'Scaled deck aspect ratio differs',0.002);
  check(result.title.length>0, `Slide title is missing: ${result.slide}`);
  check(result.outside.length===0, `Slide content exits its deck: ${result.slide}: ${JSON.stringify(result.outside)}`);
  check(result.overlaps.length===0, `Unmarked content overlap: ${result.slide}: ${JSON.stringify(result.overlaps)}`);
  check(result.body_scroll_width<=result.viewport.width+2&&result.body_scroll_height<=result.viewport.height+2,
    `Presentation page overflows the viewport at ${result.slide}`);
  return result;
}

async function checkBars(page,aggregates) {
  const rows=await page.locator('[data-metric][data-value]').evaluateAll(nodes=>nodes.map(el=>({
    metric:el.dataset.metric,value:Number(el.dataset.value),objective:el.dataset.objective,
    slideIndex:Number(el.closest('section.slide').id.slice(6))-1})));
  check(rows.length===10,'Exactly ten metric bars are required (five objectives × two APD tasks)');
  const pairs=new Set();
  for(const row of rows) {
    check(OBJECTIVES.includes(row.objective)&&METRICS.slice(0,2).includes(row.metric), 'Metric bar objective/task identity is missing');
    const key=row.objective+'/'+row.metric;check(!pairs.has(key),`Duplicate metric bar ${key}`);pairs.add(key);
    check(row.value===aggregates.find(a=>a.objective===row.objective)[row.metric], `Bar does not preserve the exact aggregate value: ${key}`);
  }
  const geometry=[];
  for(const index of [...new Set(rows.map(row=>row.slideIndex))]) {
    await goTo(page,index);
    geometry.push(...await page.locator('section.slide.active [data-metric][data-value]').evaluateAll(nodes=>nodes.map(el=>({
      metric:el.dataset.metric,objective:el.dataset.objective,value:Number(el.dataset.value),width:el.getBoundingClientRect().width,height:el.getBoundingClientRect().height}))));
  }
  for(const metric of METRICS.slice(0,2)) {
    const bars=geometry.filter(r=>r.metric===metric);
    check(bars.length===5&&bars.every(r=>r.height>0&&r.width>=0),'Metric bars are not visible');
    const eligible=bars.filter(r=>Math.abs(r.value)>1e-12);
    check(eligible.every(r=>r.width>0),'A nonzero metric has an invisible bar');
    if(eligible.length>1) {
      const factors=eligible.map(r=>r.width/Math.abs(r.value)),mean=factors.reduce((a,b)=>a+b,0)/factors.length;
      check(factors.every(v=>Math.abs(v-mean)<=Math.max(mean*0.02,0.05)),`Bar lengths do not proportionally encode ${metric}`);
      const sorted=[...eligible].sort((a,b)=>Math.abs(a.value)-Math.abs(b.value));
      check(sorted.every((r,i)=>!i||r.width+1>=sorted[i-1].width),`Bar widths invert magnitude order for ${metric}`);
    }
  }
  return {count:rows.length,exact_aggregate_binding:true,proportional_lengths_verified:true,geometry};
}

async function checkLosses(page,data) {
  const rows=await page.locator('#loss-table tbody tr').evaluateAll(nodes=>nodes.map(row=>({objective:row.dataset.objective,
    id:row.querySelector('code')?.textContent,expression:row.cells[2]?.textContent,text:row.textContent})));
  check(rows.length===5,'Loss table must contain five definitions');
  for(const [i,row] of rows.entries()) {
    const loss=data.losses[i];
    // The table typesets norms, indices and multiplication with Unicode and
    // HTML subscripts. Verify its independent mathematical presentation rather
    // than demanding that ASCII provenance formulas be displayed verbatim.
    const compact=value=>String(value).replace(/\s+/g,'');
    check(row.objective===loss.id&&normalizeText(row.id)===loss.id&&compact(row.expression)===compact(DISPLAY_EXPRESSIONS[i]),
      `Displayed loss ID/expression differs from embedded definition: ${loss.id}`);
  }
  return {rows:5,exact_ids_and_expressions:true};
}

async function checkNavigation(page) {
  const checks=[];
  await goTo(page,0);
  for(const [key,index] of [['ArrowRight',1],['ArrowLeft',0],['End',11],['Home',0]]) {
    await page.keyboard.press(key);await page.waitForFunction(i=>window.presentation.currentIndex===i,index,{timeout:10000});
    checks.push({key,...await slideState(page,index)});
  }
  for(const [button,index] of [['#next',1],['#prev',0]]) {await page.locator(button).click();checks.push({button,...await slideState(page,index)});}
  for(const [key,id,button] of [['n','#notes','#notes-button'],['o','#overview','#overview-button']]) {
    check(!await page.locator(id).isVisible(),`${id} should start hidden`);
    await page.keyboard.press(key);await page.locator(id).waitFor({state:'visible'});
    check(normalizeText(await page.locator(id).textContent()).length>10,`${id} has no usable content`);
    await page.keyboard.press(key);await page.locator(id).waitFor({state:'hidden'});
    await page.locator(button).click();await page.locator(id).waitFor({state:'visible'});
    // The open overlay is modal and deliberately intercepts pointer events.
    // Its visible close control and Escape must close it; do not force a click
    // through the overlay onto the background toolbar.
    await page.locator(id+' .close').click();await page.locator(id).waitFor({state:'hidden'});
    await page.locator(button).click();await page.locator(id).waitFor({state:'visible'});
    await page.keyboard.press('Escape');await page.locator(id).waitFor({state:'hidden'});
    checks.push({check:id+'_keyboard_toggle_button_open_close_and_escape',passed:true});
  }
  await page.keyboard.press('o');await page.locator('#overview').waitFor({state:'visible'});
  const items=page.locator('#overview [data-slide-index]');check(await items.count()===12,'Overview must expose twelve selectable slide indices');
  await page.locator('#overview [data-slide-index="5"]').click();await page.locator('#overview').waitFor({state:'hidden'});await slideState(page,5);
  checks.push({check:'overview_select_slide',slide_index:5,passed:true});
  const fullscreenApplicable=await page.evaluate(()=>Boolean(document.fullscreenEnabled&&document.documentElement.requestFullscreen));
  if(fullscreenApplicable) {
    await page.keyboard.press('f');await page.waitForFunction(()=>document.fullscreenElement!==null,{},{timeout:10000});
    await page.evaluate(()=>document.exitFullscreen());await page.waitForFunction(()=>document.fullscreenElement===null,{},{timeout:10000});
    await page.locator('#fullscreen').click();await page.waitForFunction(()=>document.fullscreenElement!==null,{},{timeout:10000});
    await page.evaluate(()=>document.exitFullscreen());await page.waitForFunction(()=>document.fullscreenElement===null,{},{timeout:10000});
  }
  checks.push({check:'fullscreen_keyboard_and_button',applicable:fullscreenApplicable,passed:fullscreenApplicable?true:null});
  await goTo(page,0);
  return checks;
}

async function mediaProof(page,id,reference) {
  const locator=page.locator(`video[data-media="${id}"]`);
  check(await locator.count()===1,`Video identity is missing/duplicate: ${id}`);
  const index=await locator.evaluate(el=>Number(el.closest('section.slide').id.slice(6))-1);
  await goTo(page,index);
  await page.evaluate(mediaId=>window.presentation.loadVideo(mediaId),id);
  await locator.evaluate(el=>new Promise((resolve,reject)=>{
    if(el.readyState>=1)return resolve();const timer=setTimeout(()=>reject(new Error('video metadata timeout')),15000);
    el.addEventListener('loadedmetadata',()=>{clearTimeout(timer);resolve();},{once:true});
    el.addEventListener('error',()=>{clearTimeout(timer);reject(new Error('video decode error'));},{once:true});
  }));
  const metadata=await locator.evaluate(el=>({src:el.currentSrc,controls:el.controls,autoplay:el.autoplay,duration:el.duration,
    width:el.videoWidth,height:el.videoHeight,paused:el.paused,error:el.error?.message||null}));
  check(metadata.controls&&!metadata.autoplay&&!metadata.error&&metadata.src.startsWith('blob:'),`Video must use controlled local Blob playback: ${id}`);
  close(metadata.duration,reference.probe.duration,`Browser duration differs from frame-count proof: ${id}`,0.05);
  check(metadata.width===reference.probe.width&&metadata.height===reference.probe.height,`Browser video dimensions differ from source proof: ${id}`);
  await locator.evaluate(async el=>{el.currentTime=0;await el.play();});
  await page.waitForFunction(mediaId=>document.querySelector(`video[data-media="${mediaId}"]`).currentTime>=0.3,id,{timeout:15000});
  const advanced=await locator.evaluate(el=>({current_time:el.currentTime,paused:el.paused,quality:el.getVideoPlaybackQuality?.()||null}));
  check(advanced.current_time>=0.3&&!advanced.paused,`Video did not actually advance: ${id}`);
  const lastTime=(reference.probe.decoded_frames-1)/reference.probe.fps;
  const last=await locator.evaluate(async (el,lastTime)=>{
    el.pause();el.__lastMediaTime=null;
    const frame=new Promise((resolve,reject)=>{
      const timer=setTimeout(()=>reject(new Error('last decoded frame was not presented')),15000);
      const onFrame=(_now,metadata)=>{if(metadata.mediaTime>=lastTime-0.02){clearTimeout(timer);el.__lastMediaTime=metadata.mediaTime;resolve(metadata.mediaTime);}else el.requestVideoFrameCallback(onFrame);};
      if(typeof el.requestVideoFrameCallback!=='function'){clearTimeout(timer);return reject(new Error('requestVideoFrameCallback unavailable'));}
      el.requestVideoFrameCallback(onFrame);
    });
    const seek=new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(new Error('last-frame seek timeout')),15000);
      el.addEventListener('seeked',()=>{clearTimeout(timer);resolve();},{once:true});});
    el.currentTime=lastTime;await seek;await el.play();await frame;
    await new Promise((resolve,reject)=>{if(el.ended)return resolve();const timer=setTimeout(()=>reject(new Error('video did not play to end')),15000);
      el.addEventListener('ended',()=>{clearTimeout(timer);resolve();},{once:true});});
    return {target_last_frame_time:lastTime,last_presented_media_time:el.__lastMediaTime,current_time:el.currentTime,ended:el.ended,
      quality:el.getVideoPlaybackQuality?{total_video_frames:el.getVideoPlaybackQuality().totalVideoFrames,dropped_video_frames:el.getVideoPlaybackQuality().droppedVideoFrames}:null};
  },lastTime);
  check(last.ended&&last.last_presented_media_time>=lastTime-0.02,`Last frame/end playback failed: ${id}`);
  await locator.evaluate(async el=>{el.currentTime=0.2;await el.play();});
  await page.waitForFunction(mediaId=>!document.querySelector(`video[data-media="${mediaId}"]`).paused,id,{timeout:10000});
  await goTo(page,index===11?0:11);
  check(await locator.evaluate(el=>el.paused),`Leaving a slide did not pause its video: ${id}`);
  return {id,source_sha256:reference.sha256,decoded_frame_proof:128,preview_fps:10,metadata,actual_playback:advanced,
    last_frame_playback:last,leaving_slide_pauses:true};
}

async function optionalPrint(page,directory) {
  const evidence={optional:true,status:'not_generated'};
  try {
    await page.emulateMedia({media:'print'});
    const visible=await page.locator('section.slide').evaluateAll(nodes=>nodes.filter(el=>{
      const s=getComputedStyle(el);return s.display!=='none'&&s.visibility!=='hidden'&&Number(s.opacity)!==0;
    }).length);
    if(visible!==12){evidence.reason='Print CSS does not expose all twelve slides';return evidence;}
    const target=path.join(directory,'presentation_print.pdf');
    const bytes=await page.pdf({path:target,printBackground:true,preferCSSPageSize:true});
    const pages=(bytes.toString('latin1').match(/\/Type\s*\/Page\b/g)||[]).length;
    check(pages===12,`Print PDF has ${pages} pages instead of twelve`);
    Object.assign(evidence,{status:'passed',path:path.basename(target),pages,sha256:sha(bytes),bytes:bytes.length});
  } catch(error) {evidence.reason=String(error.message||error);}
  finally {await page.emulateMedia({media:'screen'});}
  return evidence;
}

async function verifyLocation(browser,htmlFile,directory,{screenshots=false,print=false}={}) {
  await fs.mkdir(directory,{recursive:true});
  const context=await browser.newContext({viewport:{width:1600,height:1000},deviceScaleFactor:1,offline:true,acceptDownloads:false});
  const page=await context.newPage(),expectedURL=pathToFileURL(htmlFile).href;
  const events={external_requests:[],file_asset_requests:[],console_errors:[],page_errors:[]};
  context.on('request',request=>{const url=request.url();if(/^https?:/i.test(url))events.external_requests.push(url);
    if(url.startsWith('file:')&&url.split('#')[0]!==expectedURL)events.file_asset_requests.push(url);});
  page.on('console',msg=>{if(msg.type()==='error')events.console_errors.push(msg.text());});
  page.on('pageerror',error=>events.page_errors.push(String(error.message||error)));
  await context.route('**/*',route=>/^https?:/i.test(route.request().url())?route.abort('blockedbyclient'):route.continue());
  const result={status:'running',file_path:htmlFile,file_url:expectedURL,started_at_utc:currentUtc(),events};
  try {
    await page.goto(expectedURL,{waitUntil:'load',timeout:60000});
    await page.waitForFunction(()=>window.presentation&&typeof window.presentation.goTo==='function'
      &&typeof window.presentation.loadVideo==='function',{},{timeout:30000});
    const data=await page.locator('#presentation-data').evaluate(el=>JSON.parse(el.textContent));
    const validated=validateData(data);
    const ids=await page.locator('section.slide').evaluateAll(nodes=>nodes.map(el=>el.id));
    check(JSON.stringify(ids)===JSON.stringify(Array.from({length:12},(_,i)=>`slide-${i+1}`)),'Twelve numbered slides in order are required');
    result.scope=data.scope;result.sources=data.sources;result.images=await waitImages(page);
    const videoStates=await page.locator('video').evaluateAll(nodes=>nodes.map(el=>({id:el.dataset.media,src:el.getAttribute('src'),controls:el.controls,
      autoplay:el.autoplay,slide:Number(el.closest('section.slide').id.slice(6))-1})));
    check(videoStates.length===4&&videoStates.every(v=>MEDIA_IDS.includes(v.id)&&v.controls&&!v.autoplay), 'Exactly four controlled, non-autoplay videos are required');
    check(videoStates.every(v=>v.slide===0||!v.src),'Inactive video sources were eagerly loaded before visiting their slide');
    result.embedded_media=[];
    for(const id of MEDIA_IDS) {
      const encoded=(await page.locator(`#media-${id}`).textContent()).replace(/\s+/g,'');
      check(/^[A-Za-z0-9+/]+={0,2}$/.test(encoded),'Media script is not embedded base64');
      const bytes=Buffer.from(encoded,'base64');
      check(bytes.length>0&&bytes.length===data.media[id].bytes&&sha(bytes)===data.media[id].sha256,`Inline video bytes do not match their encoded size/SHA: ${id}`);
      result.embedded_media.push({id,bytes:bytes.length,sha256:sha(bytes)});
    }
    result.bars=await checkBars(page,validated.aggregates);result.loss_table=await checkLosses(page,data);
    result.navigation=await checkNavigation(page);result.desktop_slides=[];
    for(let i=0;i<12;i++) {
      await goTo(page,i);await waitImages(page);const state=await layout(page);
      if(screenshots){const filename=`slide${String(i+1).padStart(3,'0')}.png`;await page.screenshot({path:path.join(directory,filename),animations:'disabled'});state.screenshot=filename;}
      result.desktop_slides.push(state);
    }
    result.viewport_checks=[];
    for(const viewport of [{width:1280,height:800},{width:390,height:844},{width:844,height:390}]) {
      await page.setViewportSize(viewport);
      const checks=[];
      for(let i=0;i<12;i++){await goTo(page,i);checks.push(await layout(page));}
      result.viewport_checks.push({viewport,slides:checks});
      if(screenshots&&viewport.width===390){for(const i of [0,11]){await goTo(page,i);await page.screenshot({path:path.join(directory,`mobile-slide-${i+1}.png`),animations:'disabled'});}}
    }
    await page.setViewportSize({width:1600,height:1000});result.videos=[];
    for(const id of MEDIA_IDS) result.videos.push(await mediaProof(page,id,data.media[id]));
    await goTo(page,0);
    if(print)result.print=await optionalPrint(page,directory);
    check(events.external_requests.length===0&&events.file_asset_requests.length===0&&events.console_errors.length===0&&events.page_errors.length===0,
      `Offline/console failure: ${JSON.stringify(events)}`);
    Object.assign(result,{status:'passed',completed_at_utc:currentUtc(),all_slides_visible_and_fitted:true,
      actual_four_video_playback:true,all_last_frame_playback:true,leaving_slide_pauses_videos:true});
    return result;
  } catch(error) {result.status='failed';result.error=String(error.stack||error);throw Object.assign(error,{location_evidence:result});}
  finally {await context.close();}
}

async function main() {
  const args=argumentsFrom(process.argv.slice(2));
  if(args.help){process.stdout.write('Usage: node verify_component_presentation.cjs --presentation ABSHTML --qa-dir ABSSCRATCH --playwright-module MODULE --browser-executable EDGE\n');return 0;}
  const source=await fs.realpath(args.presentation),qaDir=path.resolve(args['qa-dir']);
  check((await fs.stat(source)).isFile()&&/\.html?$/i.test(source),'Presentation must be one existing HTML file');
  const relative=path.relative(source,qaDir);check(relative!==''&&!qaDir.startsWith(source+path.sep),'QA scratch must not overwrite the presentation');
  await fs.mkdir(qaDir,{recursive:true});const qaPath=path.join(qaDir,'presentation_qa.json');
  try {await fs.access(qaPath);throw new Error('Existing QA evidence must be preserved; choose a fresh --qa-dir');}catch(error){if(error.code!=='ENOENT')throw error;}
  const started=currentUtc(),inputBytes=await fs.readFile(source),inputSha=sha(inputBytes);
  const output={schema_version:1,status:'running',started_at_utc:started,presentation:source,presentation_sha256:inputSha,
    verifier_sha256:sha(await fs.readFile(__filename)),command:process.argv,cpu_only:true,new_model_inference:false,experiment_modified:false,errors:[]};
  let browser;
  try {
    const playwright=require(args['playwright-module']);
    browser=await playwright.chromium.launch({headless:true,executablePath:args['browser-executable'],
      args:['--disable-gpu','--disable-background-networking','--disable-extensions','--no-first-run']});
    output.browser_version=browser.version();
    output.original=await verifyLocation(browser,source,path.join(qaDir,'original'),{screenshots:true,print:true});
    const copyDir=await fs.mkdtemp(path.join(qaDir,'단독 이동 복사 ')),copy=path.join(copyDir,'공격 실험 요약 발표.html');
    await fs.copyFile(source,copy,fs.constants.COPYFILE_EXCL);
    check(JSON.stringify(await fs.readdir(copyDir))===JSON.stringify([path.basename(copy)]),'Portable test folder must contain only the renamed HTML');
    output.relocated_html=copy;output.relocated_sha256=sha(await fs.readFile(copy));check(output.relocated_sha256===inputSha,'Renamed presentation bytes differ');
    output.relocated=await verifyLocation(browser,copy,path.join(qaDir,'relocated'),{screenshots:false,print:false});
    check(sha(await fs.readFile(source))===inputSha&&sha(await fs.readFile(copy))===inputSha,'Presentation bytes changed during QA');
    Object.assign(output,{status:'passed',completed_at_utc:currentUtc(),isolated_single_file_copy_verified:true,external_network_requests:0,
      external_file_asset_requests:0,console_errors:0,actual_four_video_playback:true,all_last_frame_playback:true});
  } catch(error) {output.status='failed';output.errors.push(String(error.stack||error));if(error.location_evidence)output.failed_location=error.location_evidence;}
  finally {if(browser)await browser.close();}
  output.elapsed_seconds=(Date.parse(currentUtc())-Date.parse(started))/1000;
  await fs.writeFile(qaPath,JSON.stringify(output,null,2)+'\n',{encoding:'utf8',flag:'wx'});
  process.stdout.write(JSON.stringify({status:output.status,qa:qaPath,presentation_sha256:inputSha,errors:output.errors})+'\n');
  return output.status==='passed'?0:1;
}

if(require.main===module) main().then(code=>{process.exitCode=code;}).catch(error=>{process.stderr.write(String(error.stack||error)+'\n');process.exitCode=1;});
module.exports={argumentsFrom,validateData,validateLocation:verifyLocation};
