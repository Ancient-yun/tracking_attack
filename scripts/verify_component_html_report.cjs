/* Offline browser verification for the portable component-study HTML report.
 * Usage: node verify_component_html_report.cjs --report-dir PATH
 *   --playwright-module PATH --browser-executable PATH [--synthetic]
 * No model/experiment is run. A synthetic smoke can never mark a real report complete.
 */
'use strict';
const fs = require('node:fs');
const fsp = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const { pathToFileURL, fileURLToPath } = require('node:url');

const OBJECTIVES = ['tracking_mse','tracking_3d','reconstruction_3d','confidence','joint_training'];
const METRICS = ['tracking_apd_drop_pp','reconstruction_apd_drop_pp','tracking_epe_increase_m','reconstruction_epe_increase_m'];
const METRIC_LABELS = {tracking_apd_drop_pp:'Tracking APD 감소 (%p)',reconstruction_apd_drop_pp:'Recon APD 감소 (%p)',
  tracking_epe_increase_m:'Tracking EPE 증가 (m)',reconstruction_epe_increase_m:'Recon EPE 증가 (m)'};
const GROUP_LABELS = {all:'전체 8클립',po_mini:'Point Odyssey 4클립',ds_mini:'Dynamic Replica 4클립'};
const fmt=(v,n=2)=>typeof v==='number'&&Number.isFinite(v)?v.toFixed(n):'N/A';
const sign=(v,n=2)=>typeof v==='number'&&Number.isFinite(v)?(v>0?'+':'')+v.toFixed(n):'N/A';
const pairText=(row,m,n=2)=>fmt(row[m+'_clean'],n)+' → '+fmt(row[m+'_attacked'],n);
const ciText=(ci,n=2)=>ci?'['+sign(ci[0],n)+', '+sign(ci[1],n)+']':'N/A';
const NOW = () => new Date().toISOString();
function check(condition, message) { if (!condition) throw new Error(message); }
function argumentsOf(argv) {
  const result = { synthetic: false };
  for (let i=0; i<argv.length; ++i) {
    const value=argv[i];
    if(value==='--synthetic') { result.synthetic=true; continue; }
    check(['--report-dir','--playwright-module','--browser-executable'].includes(value),`Unknown option ${value}`);
    check(i+1<argv.length,`Missing value for ${value}`);
    result[value.slice(2).replace(/-/g,'_')]=argv[++i];
  }
  for (const field of ['report_dir','playwright_module','browser_executable']) check(result[field],`Required --${field.replace(/_/g,'-')}`);
  return result;
}
async function readJson(file) { return JSON.parse((await fsp.readFile(file,'utf8')).replace(/^\uFEFF/,'')); }
async function writeJson(file,value) {
  await fsp.mkdir(path.dirname(file),{recursive:true});
  const temporary=file+'.writing';
  await fsp.writeFile(temporary,JSON.stringify(value,null,2)+'\n','utf8');
  await fsp.rename(temporary,file);
}
async function hashFile(file) {
  const digest=crypto.createHash('sha256');
  for await (const chunk of fs.createReadStream(file)) digest.update(chunk);
  return digest.digest('hex');
}
function localPath(root,relative) {
  check(typeof relative==='string' && !path.isAbsolute(relative),`Unsafe output path: ${relative}`);
  const resolved=path.resolve(root,relative), relation=path.relative(root,resolved);
  check(relation && !relation.startsWith('..'+path.sep) && relation!=='..' && !path.isAbsolute(relation),`Output path escapes report: ${relative}`);
  return resolved;
}
async function fileEvidence(file) { const stat=await fsp.stat(file); return {sha256:await hashFile(file),bytes:stat.size}; }
async function verifyOutputs(root,outputs) {
  check(outputs && typeof outputs==='object', 'Output hash inventory is required');
  check(outputs['index.html'] && outputs['data/report_data.json'], 'Index and report data must have hash evidence');
  for (const [relative, recorded] of Object.entries(outputs)) {
    const actual=await fileEvidence(localPath(root,relative));
    check(actual.sha256===recorded.sha256 && actual.bytes===recorded.bytes,`Published output changed: ${relative}`);
  }
}
async function listFiles(root,relative='') {
  let results=[];
  for (const entry of await fsp.readdir(path.join(root,relative),{withFileTypes:true})) {
    const name=path.join(relative,entry.name);
    check(!entry.isSymbolicLink(),`Report must not depend on symlinks: ${name}`);
    if(entry.isDirectory()) results.push(...await listFiles(root,name));
    else if(entry.isFile() && !name.endsWith('.writing') && name!=='report_manifest.json') results.push(name.replace(/\\/g,'/'));
  }
  return results;
}
function expectedCells(row) {
  const pair=(m,n=2)=>fmt(row[m+'_clean'],n)+' → '+fmt(row[m+'_attacked'],n);
  return [pair(METRICS[0]),sign(row[METRICS[0]])+' / '+fmt(row.tracking_apd_relative_drop_pct)+'%',
          pair(METRICS[1]),sign(row[METRICS[1]])+' / '+fmt(row.reconstruction_apd_relative_drop_pct)+'%',
          pair(METRICS[2],3),sign(row[METRICS[2]],3)+' / '+fmt(row.tracking_epe_ratio)+'배',
          pair(METRICS[3],3),sign(row[METRICS[3]],3)+' / '+fmt(row.reconstruction_epe_ratio)+'배'];
}
async function numericRows(page,expectedRows) {
  const rows=await page.locator('#details-table tbody tr').evaluateAll(nodes=>nodes.map(tr=>{
    const cells=Array.from(tr.cells), first=cells[0];
    return {dataset:tr.dataset.dataset || first.querySelector('small').textContent.trim(),
      sequence:tr.dataset.sequence || first.firstChild.textContent.trim(),
      objective:tr.dataset.objective || cells[1].textContent.trim(),
      cells:cells.slice(2,10).map(cell=>cell.textContent.trim()), selected:cells[10].textContent.trim()};
  }));
  check(rows.length===expectedRows.length,`Detail support ${rows.length}, expected ${expectedRows.length}`);
  const expected=new Map(expectedRows.map(row=>[[row.dataset,row.sequence,row.objective].join('/'),row]));
  const seen=new Set();
  for(const row of rows) {
    const key=[row.dataset,row.sequence,row.objective].join('/'), value=expected.get(key);
    check(value && !seen.has(key),`Unexpected/duplicate detail key ${key}`); seen.add(key);
    assert.deepStrictEqual(row.cells,expectedCells(value),`Numeric cell mismatch for ${key}`);
    check(row.selected===(value.selected_state.restart===-1?'clean':String(value.selected_state.step)),`Selected iterate differs for ${key}`);
  }
  return rows.length;
}
async function summaryTables(page,payload,dataset) {
  const impacts=payload.impact.filter(row=>row.dataset===dataset);
  const rows=await page.locator('#impact-table tbody tr').evaluateAll(nodes=>nodes.map(tr=>Array.from(tr.cells).map(cell=>({
    main:cell.firstChild?.textContent.trim()||'',small:cell.querySelector('small')?.textContent.trim()||null}))));
  check(rows.length===5&&impacts.length===5,'Impact must include all five objectives');
  for(let i=0;i<impacts.length;i++) {
    const r=impacts[i], expected=[{main:r.objective,small:String(r.paired_clips||r.expected_clips)+' paired clips'},
      {main:pairText(r,METRICS[0]),small:null},{main:sign(r[METRICS[0]]),small:ciText(r.ci95?.[METRICS[0]])},
      {main:fmt(r.tracking_apd_relative_drop_pct),small:null},{main:pairText(r,METRICS[1]),small:null},
      {main:sign(r[METRICS[1]]),small:ciText(r.ci95?.[METRICS[1]])},{main:fmt(r.reconstruction_apd_relative_drop_pct),small:null}];
    for(const m of METRICS.slice(2)) {
      const ratio=m===METRICS[2]?r.tracking_epe_ratio:r.reconstruction_epe_ratio;
      expected.push({main:pairText(r,m,3),small:'증가 '+sign(r[m],3)+'m · '+fmt(ratio)+'배 · CI '+ciText(r.ci95?.[m],3)});
    }
    assert.deepStrictEqual(rows[i],expected,`Impact values/CI/ratios differ: ${dataset}/${r.objective}`);
  }
  const contrasts=payload.contrasts.estimates.filter(row=>row.dataset===dataset);
  const actual=await page.locator('#contrast-table tbody tr').evaluateAll(nodes=>nodes.map(tr=>Array.from(tr.cells).map(cell=>cell.textContent.trim())));
  check(actual.length===4&&contrasts.length===4,'Direct contrast must include four metrics');
  for(let i=0;i<contrasts.length;i++) {
    const r=contrasts[i],n=r.metric.includes('epe')?3:2;
    const interpretation=r.ci95[0]<=0&&r.ci95[1]>=0?'차이 불확실 (0 포함)':r.estimate>0?'Tracking 목적 공격의 상대 손상 큼':'Reconstruction 목적 공격의 상대 손상 큼';
    assert.deepStrictEqual(actual[i],[METRIC_LABELS[r.metric],sign(r.estimate,n),ciText(r.ci95,n),String(r.paired_clips||r.common_clips),interpretation],
      `Direct contrast point/CI/support differs: ${dataset}/${r.metric}`);
  }
  const points=payload.scatter.points.filter(row=>dataset==='all'||row.dataset===dataset);
  const excluded=payload.scatter.excluded.filter(row=>dataset==='all'||row.dataset===dataset);
  const connections=payload.scatter.connections.filter(row=>dataset==='all'||row.dataset===dataset);
  const support=await page.locator('#scatter-support').textContent();
  check(support.includes(GROUP_LABELS[dataset])&&support.includes('유효 점 '+points.length)&&support.includes('N/A 제외 '+excluded.length),
    `Scatter support/exclusion text differs: ${dataset}`);
  if(support.includes('연결선')) check(support.includes('연결선 '+connections.length),`Scatter connection count differs: ${dataset}`);
  const excludedRows=await page.locator('#scatter-excluded tbody tr').evaluateAll(nodes=>nodes.map(tr=>Array.from(tr.cells).map(cell=>cell.textContent.trim())));
  check(excludedRows.length===excluded.length,`Scatter N/A exclusion table support differs: ${dataset}`);
  for(let i=0;i<excluded.length;i++) {
    const row=excluded[i];
    assert.deepStrictEqual(excludedRows[i].slice(0,3),[row.dataset+' / '+row.sequence,row.objective,fmt(row.x)+' / '+fmt(row.y)],
      `Scatter N/A table identity/value differs: ${dataset}`);
    check(excludedRows[i][3].includes(row.reason),'Scatter denominator exclusion reason was hidden');
  }
  for(const connection of connections) {
    const a=points.find(row=>row.dataset===connection.dataset&&row.sequence===connection.sequence&&row.objective==='tracking_3d');
    const b=points.find(row=>row.dataset===connection.dataset&&row.sequence===connection.sequence&&row.objective==='reconstruction_3d');
    check(a&&b,'Scatter connection lacks the same clip\'s two geometry attacks');
    assert.deepStrictEqual(connection.from,[a.x,a.y],'Scatter tracking->reconstruction source differs');
    assert.deepStrictEqual(connection.to,[b.x,b.y],'Scatter tracking->reconstruction destination differs');
  }
  return {impact_rows:5,contrast_rows:4,scatter_points:points.length,scatter_excluded:excluded.length,scatter_connections:connections.length};
}
async function readyImages(page) {
  await page.waitForFunction(()=>Array.from(document.images).filter(img=>img.getAttribute('src')).every(img=>img.complete && img.naturalWidth>0),{},{timeout:30000});
}
async function readyVideos(page) {
  await page.waitForFunction(()=>Array.from(document.querySelectorAll('video')).every(video=>video.readyState>=1 && Number.isFinite(video.duration) && !video.error),{},{timeout:30000});
  return page.locator('video').evaluateAll(videos=>videos.map(video=>({id:video.id,duration:video.duration,
    paused:video.paused,autoplay:video.autoplay,controls:video.controls,width:video.videoWidth,height:video.videoHeight,src:video.currentSrc})));
}
async function checkLocalReferences(page,root) {
  const urls=await page.evaluate(()=>Array.from(document.querySelectorAll('[src],[href],[poster]')).flatMap(node=>
    ['src','href','poster'].filter(attribute=>node.hasAttribute(attribute)).map(attribute=>node.getAttribute(attribute))).filter(Boolean));
  let checked=0;
  for(const value of urls) {
    if(value.startsWith('#') || value.startsWith('data:') || value.startsWith('mailto:')) continue;
    const resolved=new URL(value,pathToFileURL(path.join(root,'index.html')).href);
    check(resolved.protocol==='file:',`Report references a network asset: ${value}`);
    const absolute=fileURLToPath(resolved), relative=path.relative(root,absolute);
    check(relative && !relative.startsWith('..'+path.sep) && relative!=='..' && !path.isAbsolute(relative),`Report asset escapes portable folder: ${value}`);
    check((await fsp.stat(absolute)).isFile(),`Missing local reference ${value}`); checked++;
  }
  return checked;
}
async function allOption(page,selector) {
  const values=await page.locator(selector).evaluate(node=>Array.from(node.options).map(option=>option.value));
  const value=values.includes('all')?'all':values.includes('')?'':values[0];
  await page.selectOption(selector,value);
}
async function exerciseReport(browser,root,payload,qaDir,portable=false) {
  const context=await browser.newContext({offline:true,viewport:{width:1440,height:1100},deviceScaleFactor:1});
  const page=await context.newPage();
  const errors=[],external=[],steps=[];
  page.on('pageerror',error=>errors.push(String(error)));
  page.on('console',message=>{if(message.type()==='error') errors.push(message.text());});
  page.on('request',request=>{if(/^https?:/i.test(request.url())) external.push(request.url());});
  await page.route(/^https?:\/\//,route=>route.abort('internetdisconnected'));
  try {
    await page.goto(pathToFileURL(path.join(root,'index.html')).href,{waitUntil:'load',timeout:30000});
    await page.waitForFunction(()=>window.reportReady===true,{},{timeout:30000});
    const inline=await page.locator('#report-payload').evaluate(node=>JSON.parse(node.textContent));
    assert.deepStrictEqual(inline,payload,'Embedded browser data differs from hash-verified report_data.json');
    await readyImages(page);await readyVideos(page);
    const references=await checkLocalReferences(page,root);
    await numericRows(page,payload.conditions);
    await summaryTables(page,payload,'all');
    const initialVideos=await readyVideos(page);
    check(initialVideos.length===2 && initialVideos.every(video=>video.paused && !video.autoplay && video.controls),'Videos must have controls and no automatic playback');
    steps.push({check:'file_url_ready_local_assets_no_autoplay',passed:true,local_references:references});
    const failureCount=Number(payload.validation.recorded_failed_attempts);
    if(failureCount>0) check((await page.locator('#warnings').textContent()).includes('과거 실패 시도 기록: '+JSON.stringify(payload.validation.recorded_failed_attempts)),
      'Recorded historical failures were omitted from the visible warning area');
    steps.push({check:'recorded_failure_warning_preserved',passed:true,recorded_failed_attempts:failureCount});
    for(const dataset of ['po_mini','ds_mini','all']) {
      await page.locator(`#dataset-buttons button[data-value="${dataset}"]`).click();
      await readyImages(page);
      const expected=payload.conditions.filter(row=>dataset==='all'||row.dataset===dataset);
      await numericRows(page,expected);
      const summary=await summaryTables(page,payload,dataset);
      steps.push({check:'dataset_filter_numeric_rows_and_summaries',dataset,count:expected.length,passed:true,...summary});
    }
    await allOption(page,'#detail-objective');await allOption(page,'#detail-clip');
    await page.selectOption('#detail-objective','tracking_3d');
    await numericRows(page,payload.conditions.filter(row=>row.objective==='tracking_3d'));
    const first=payload.clips[0], firstId=first.dataset+'/'+first.sequence;
    await page.selectOption('#detail-clip',firstId);
    await numericRows(page,payload.conditions.filter(row=>row.objective==='tracking_3d'&&row.dataset===first.dataset&&row.sequence===first.sequence));
    await allOption(page,'#detail-objective');await allOption(page,'#detail-clip');
    await page.fill('#detail-search',first.sequence);
    await numericRows(page,payload.conditions.filter(row=>(row.dataset+' '+row.sequence+' '+row.objective).toLowerCase().includes(first.sequence.toLowerCase())));
    await page.fill('#detail-search','');
    await page.selectOption('#detail-sort','tracking_apd_drop_pp');
    await numericRows(page,payload.conditions);
    const firstSorted=await page.locator('#details-table tbody tr').first().locator('td').nth(3).textContent();
    const highest=Math.max(...payload.conditions.map(row=>row.tracking_apd_drop_pp));
    check(firstSorted.trim().startsWith((highest>0?'+':'')+highest.toFixed(2)),'Descending metric sort is incorrect');
    await page.selectOption('#detail-sort','manifest');
    await numericRows(page,payload.conditions);
    steps.push({check:'objective_clip_search_sort_restore',passed:true});
    const caseList=portable?[payload.conditions[0]]:payload.conditions;
    for(const row of caseList) {
      await page.selectOption('#clip-select',row.dataset+'/'+row.sequence);
      await page.selectOption('#objective-select',row.objective);
      await readyImages(page);const videos=await readyVideos(page);
      check(videos.length===2 && videos.every(video=>video.paused && !video.autoplay),'Case change started automatic playback');
      const media=payload.media.find(value=>value.dataset===row.dataset&&value.sequence===row.sequence&&value.objective===row.objective);
      for(const [id,name] of [['rgb-video','rgb_comparison.mp4'],['tracking-video','tracking_comparison.mp4']]) {
        const video=videos.find(value=>value.id===id), record=media.assets[name];
        check(record.probe.decoded_frames===128 && record.probe.success===true,'Video lacks verified decoded128 frame proof');
        check(Math.abs(video.duration-record.probe.duration)<.1 && video.width===record.probe.width && video.height===record.probe.height,
          `Browser video metadata differs from ffprobe: ${row.objective}/${row.sequence}/${id}`);
      }
    }
    steps.push({check:'case_objective_media_metadata',passed:true,cases:caseList.length});
    for(const frame of [0,64,127]) {
      await page.locator(`#frame-buttons button[data-frame="${frame}"]`).click();
      await readyImages(page);
      check((await page.locator('#triptych-image').getAttribute('src')).endsWith(`geometry_attack_triptych_frame_${String(frame).padStart(3,'0')}.png`),'Frame selector changed to wrong still');
    }
    await page.locator('#impact-image').click();
    check(await page.locator('#image-dialog').isVisible(),'Image expansion dialog did not open');
    await readyImages(page);await page.locator('#dialog-close').click();
    check(!(await page.locator('#image-dialog').isVisible()),'Image dialog did not close');
    steps.push({check:'source_frames_0_64_127_and_image_dialog',passed:true});
    for(const id of ['rgb-video','tracking-video']) {
      await page.locator('#'+id).evaluate(async video=>{video.muted=true;video.currentTime=.15;await video.play();});
      await page.waitForFunction(id=>{const video=document.getElementById(id);return !video.paused && video.currentTime>.35;},id,{timeout:15000});
      const played=await page.locator('#'+id).evaluate(video=>{const value={current_time:video.currentTime,duration:video.duration};video.pause();return value;});
      steps.push({check:'actual_video_playback',video:id,passed:true,...played});
    }
    await page.locator('#dataset-buttons button[data-value="all"]').click();
    await page.selectOption('#clip-select',firstId);await page.selectOption('#objective-select','tracking_3d');
    await page.locator('#frame-buttons button[data-frame="64"]').click();
    await readyImages(page);await readyVideos(page);
    await page.evaluate(()=>window.scrollTo(0,0));
    if(!portable) {
      await fsp.mkdir(path.join(qaDir,'screenshots'),{recursive:true});
      await page.screenshot({path:path.join(qaDir,'screenshots','desktop.png'),fullPage:true});
      await page.screenshot({path:path.join(qaDir,'screenshots','desktop_first_view.png')});
    }
    for(const width of [1440,390]) {
      await page.setViewportSize({width,height:1100});await readyImages(page);
      const dimensions=await page.evaluate(()=>({width:document.documentElement.clientWidth,scroll_width:document.body.scrollWidth}));
      check(dimensions.scroll_width<=dimensions.width+2,`Body overflows at viewport ${width}: ${dimensions.scroll_width}`);
      if(width===390&&!portable) {
        await page.screenshot({path:path.join(qaDir,'screenshots','mobile.png'),fullPage:true});
        await page.screenshot({path:path.join(qaDir,'screenshots','mobile_first_view.png')});
      }
      steps.push({check:'responsive_body_overflow',passed:true,viewport:width,...dimensions});
    }
    await page.setViewportSize({width:1122,height:794});await page.emulateMedia({media:'print'});
    const printInfo=await page.evaluate(()=>({overflow:Array.from(document.querySelectorAll('.table-wrap')).map(node=>({client:node.clientWidth,scroll:node.scrollWidth})),
      visible_videos:Array.from(document.querySelectorAll('video')).filter(node=>getComputedStyle(node).display!=='none'&&node.getBoundingClientRect().height>0).length,
      body_width:document.body.scrollWidth,viewport:document.documentElement.clientWidth}));
    check(printInfo.body_width<=printInfo.viewport+2 && printInfo.overflow.every(value=>value.scroll<=value.client+2),'Print tables overflow printable layout');
    check(printInfo.visible_videos===0,'Print layout must replace video players with static evidence');
    if(!portable) {
      await page.pdf({path:path.join(qaDir,'report_print.pdf'),format:'A4',landscape:true,printBackground:true,preferCSSPageSize:true});
      await page.screenshot({path:path.join(qaDir,'screenshots','print_layout.png'),fullPage:true});
    }
    steps.push({check:'print_static_tables',passed:true,...printInfo});
    check(errors.length===0,'Browser errors: '+errors.join(' | '));
    check(external.length===0,'External network requests: '+external.join(' | '));
    return {passed:true,portable_copy:portable,steps,console_errors:errors,external_network_requests:external,local_references_checked:references};
  } finally { await context.close(); }
}

async function main() {
  const args=argumentsOf(process.argv.slice(2));
  const root=path.resolve(args.report_dir), qaDir=path.join(root,'qa'), manifestFile=path.join(root,'report_manifest.json');
  let manifest, browser, temporary;
  const qa={schema_version:1,status:'running',synthetic_only:args.synthetic,started_at_utc:NOW(),report_dir:root,
    cpu_only:true,new_model_inference:false,command:process.argv,errors:[]};
  try {
    manifest=await readJson(manifestFile);
    check(['built_pending_browser_qa','complete'].includes(manifest.status),'Report must be freshly built or previously verified complete');
    check(args.synthetic || manifest.browser_qa?.synthetic_only!==true,'Synthetic QA evidence cannot be promoted to a real report');
    const payload=await readJson(path.join(root,'data','report_data.json'));
    check(Boolean(payload.synthetic_only)===args.synthetic && Boolean(manifest.synthetic_only)===args.synthetic,
      '--synthetic must match the report builder\'s explicit synthetic_only flag');
    check(payload.run_signature===manifest.run_signature && payload.run_id===manifest.run_id,'Payload and manifest run identity differ');
    check(payload.conditions.length===40 && payload.clips.length===8 && payload.scope.num_frames===128 && payload.scope.expected_conditions===48
      && payload.scope.clips===8 && payload.scope.all_frames===true,'Report is not the complete 8clip/128frame/48condition study');
    assert.deepStrictEqual(payload.objectives,OBJECTIVES,'Objective order differs');
    check(payload.validation.completed_conditions===48 && payload.validation.raw_frame_verified_clips===8 && payload.validation.errors.length===0
      && payload.audit.passed===true && payload.audit.full_campaign_completed===true && payload.audit.all_frames_completed===true,'Full numerical audit completion is required');
    await verifyOutputs(root,manifest.outputs);
    const coreOutputs=Object.fromEntries(Object.entries(manifest.outputs).filter(([relative])=>!relative.startsWith('qa/')));
    qa.index_sha256=await hashFile(path.join(root,'index.html'));
    qa.report_data_sha256=await hashFile(path.join(root,'data','report_data.json'));
    qa.run_signature=manifest.run_signature;qa.outputs_verified_before=Object.keys(manifest.outputs).length;
    const {chromium}=require(path.resolve(args.playwright_module));
    browser=await chromium.launch({executablePath:path.resolve(args.browser_executable),headless:true,
      args:['--disable-gpu','--disable-background-networking','--disable-extensions','--no-first-run']});
    qa.browser_version=browser.version();qa.browser_executable=path.resolve(args.browser_executable);
    qa.original_folder=await exerciseReport(browser,root,payload,qaDir,false);
    temporary=await fsp.mkdtemp(path.join(os.tmpdir(),'codex-component-report-qa-'));
    await fsp.cp(root,temporary,{recursive:true});
    qa.portable_folder=await exerciseReport(browser,temporary,payload,qaDir,true);
    await verifyOutputs(root,coreOutputs); // QA evidence is regenerated; numerical/report assets stay immutable.
    check(qa.index_sha256===await hashFile(path.join(root,'index.html')),'Index changed during browser QA');
    qa.qa_source_sha256=await hashFile(__filename);
    qa.completed_at_utc=NOW();qa.status='passed';
    await writeJson(path.join(qaDir,'browser_qa.json'),qa);
    manifest.browser_qa={status:'passed',synthetic_only:args.synthetic,index_sha256:qa.index_sha256,
      report_data_sha256:qa.report_data_sha256,qa_source_sha256:qa.qa_source_sha256,
      report:'qa/browser_qa.json',sha256:await hashFile(path.join(qaDir,'browser_qa.json')),completed_at_utc:qa.completed_at_utc,
      actual_video_playback:true,portable_copy_verified:true,external_network_requests:0};
    for(const relative of await listFiles(root)) manifest.outputs[relative]=await fileEvidence(localPath(root,relative));
    manifest.status=args.synthetic?'synthetic_qa_passed':'complete';
    manifest.completed_at_utc=qa.completed_at_utc;
    await writeJson(manifestFile,manifest);
    await verifyOutputs(root,manifest.outputs);
    console.log(JSON.stringify({status:manifest.status,synthetic_only:args.synthetic,report_dir:root,qa:'qa/browser_qa.json',index_sha256:qa.index_sha256}));
  } catch(error) {
    qa.status='failed';qa.completed_at_utc=NOW();qa.errors.push(String(error.stack||error));
    await writeJson(path.join(qaDir,'browser_qa.json'),qa);
    if(manifest) {
      manifest.status='failed';manifest.browser_qa={status:'failed',synthetic_only:args.synthetic,report:'qa/browser_qa.json',error:String(error.message||error)};
      manifest.errors=[...(manifest.errors||[]),{stage:'browser_qa',error:String(error.message||error),at_utc:NOW()}];
      await writeJson(manifestFile,manifest);
    }
    console.error(error.stack||String(error));process.exitCode=1;
  } finally {
    if(browser) await browser.close();
    if(temporary) {
      const resolved=await fsp.realpath(temporary), tempRoot=await fsp.realpath(os.tmpdir()), relative=path.relative(tempRoot,resolved);
      check(relative && !relative.startsWith('..'+path.sep) && !path.isAbsolute(relative)
        && path.basename(resolved).startsWith('codex-component-report-qa-'),'Refusing cleanup outside the named QA temp directory');
      await fsp.rm(resolved,{recursive:true,force:true});
    }
  }
}
main().catch(error=>{console.error(error.stack||String(error));process.exitCode=1;});
