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
async function interpretationParagraphs(page,selector,expected,label) {
  check(Array.isArray(expected),`Missing interpretation evidence: ${label}`);
  const rows=await page.locator(selector+' p').evaluateAll(nodes=>nodes.map(node=>({text:node.textContent.trim(),
    dataset:node.dataset.dataset||null,objective:node.dataset.objective||null})));
  assert.deepStrictEqual(rows.map(row=>row.text),expected,`Visible interpretation differs from verified scalar narrative: ${label}`);
  return rows;
}
async function numericKeyRows(page,selector,expected,label) {
  const actual=await page.locator(selector+' tbody tr').evaluateAll(nodes=>nodes.map(row=>({
    dataset:row.dataset.dataset||null,sequence:row.dataset.sequence||null,objective:row.dataset.objective||null,
    metric:row.dataset.metric||null,state:row.dataset.state||null,stage:row.dataset.stage||null,
    cells:Array.from(row.querySelectorAll('[data-key]')).map(cell=>({key:cell.dataset.key,text:cell.textContent.trim(),
      digits:cell.dataset.digits??null,signed:cell.dataset.signed??null,raw:cell.dataset.raw??null}))})));
  check(actual.length===expected.length,`${label} row support differs: ${actual.length}/${expected.length}`);
  for(let i=0;i<expected.length;i++) {
    const reference=expected[i],row=actual[i];
    for(const identity of ['dataset','sequence','objective','metric','state','stage']) if(reference[identity]!==undefined)
      check(row[identity]===reference[identity],`${label} row ${i} ${identity} differs`);
    const cells=new Map(row.cells.map(cell=>[cell.key,cell]));
    check(cells.size===row.cells.length,`${label} has duplicate numeric keys`);
    for(const field of reference.fields) {
      const cell=cells.get(field.key);check(cell,`${label} omits visible numeric ${field.key}`);
      const text=field.text!==undefined?field.text:field.signed?sign(field.value,field.digits??3):fmt(field.value,field.digits??3);
      check(cell.text===text,`${label} actual numeric text differs at ${i}/${field.key}: ${cell.text}/${text}`);
      if(cell.digits!==null&&field.digits!==undefined) check(Number(cell.digits)===field.digits,`${label} display precision differs: ${field.key}`);
      if(cell.signed!==null&&field.signed!==undefined) check(cell.signed===String(field.signed),`${label} sign display policy differs: ${field.key}`);
      if(field.raw!==undefined) check(cell.raw===String(field.raw),`${label} actual raw status differs: ${field.key}`);
    }
  }
  return actual.length;
}
async function interpretationGroup(page,payload,dataset) {
  const value=payload.interpretation;
  check(value?.schema_version===1&&value.run_signature===payload.run_signature,'Interpretation must bind the completed scalar run');
  const impacts=await interpretationParagraphs(page,'#impact-interpretation',value.impact_by_group[dataset],dataset+'/impact');
  check(impacts.length===5&&impacts.every((row,i)=>row.dataset===dataset&&row.objective===OBJECTIVES[i]),'Interpretation impact scope/objective differs');
  const comparisons=await interpretationParagraphs(page,'#dataset-comparison',value.dataset_comparison,'PO/DR comparison');
  check(comparisons.length===5&&comparisons.every((row,i)=>row.objective===OBJECTIVES[i]),'Dataset comparison lost an objective');
  const contrasts=await interpretationParagraphs(page,'#contrast-interpretation',value.contrasts_by_group[dataset],dataset+'/contrast');
  check(contrasts.every(row=>row.dataset===dataset),'Contrast narrative did not follow the dataset filter');
  const movement=value.movement_by_group[dataset];
  const moveParagraphs=await interpretationParagraphs(page,'#movement-interpretation',movement.summary,dataset+'/movement');
  check(moveParagraphs.every(row=>row.dataset===dataset),'Movement narrative did not follow the dataset filter');
  const movementRows=await numericKeyRows(page,'#movement-table',movement.clips.map(row=>({dataset:row.dataset,sequence:row.sequence,
    fields:[{key:'from_x',value:row.from[0],digits:2,signed:true},{key:'from_y',value:row.from[1],digits:2,signed:true},
      {key:'to_x',value:row.to[0],digits:2,signed:true},{key:'to_y',value:row.to[1],digits:2,signed:true},
      {key:'delta_x_pct',value:row.delta_x_pct,digits:2,signed:true},{key:'delta_y_pct',value:row.delta_y_pct,digits:2,signed:true}]})),dataset+'/movement coordinates');
  const objective=await page.locator('#detail-objective').inputValue();
  const rows=value.consistency_by_group[dataset].filter(row=>objective==='all'||row.objective===objective);
  const consistencyRows=await numericKeyRows(page,'#consistency-table',rows.map(row=>({dataset:row.dataset,objective:row.objective,metric:row.metric,
    fields:[...['worsened','unchanged','improved'].map(key=>({key:'counters.'+key,value:row.counters[key],digits:0})),
      ...['median','min','max'].map(key=>({key,value:row[key],digits:row.metric.includes('epe')?3:2,signed:true}))]})),dataset+'/clip consistency');
  const paired=dataset==='all'?8:4;
  check(rows.every(row=>row.paired_clips===paired&&row.counters.worsened+row.counters.unchanged+row.counters.improved===paired),
    'Clip consistency counts must preserve paired population');
  const scope=await page.locator('#consistency-scope').textContent();
  check(scope.includes(GROUP_LABELS[dataset])&&scope.includes((objective==='all'?'다섯 목적':objective))
    &&scope.includes(rows.length+'개 지표 행')&&scope.includes('paired 클립 '+paired),'Visible clip consistency scope differs');
  const extrema=await page.locator('#consistency-table tbody tr').evaluateAll(nodes=>nodes.map(row=>Array.from(row.cells).slice(-2).map(cell=>cell.textContent.trim())));
  assert.deepStrictEqual(extrema,rows.map(row=>[row.min_clips.join(', '),row.max_clips.join(', ')]),'Visible minimum/maximum clip identities differ');
  return {impact_paragraphs:impacts.length,dataset_comparison_paragraphs:comparisons.length,
    contrast_paragraphs:contrasts.length,movement_rows:movementRows,consistency_rows:consistencyRows};
}
async function linkedCpuAndLaunchEvidence(page,root,payload) {
  const cpu=payload.cpu_postprocessing,configuration=payload.source_configuration;
  check(cpu?.status==='snapshot_before_browser_qa'&&cpu.not_gpu_experiment_time===true
    &&typeof cpu.builder_elapsed_until_html_content_snapshot_seconds==='number'
    &&Number.isFinite(cpu.builder_elapsed_until_html_content_snapshot_seconds)&&cpu.builder_elapsed_until_html_content_snapshot_seconds>=0,
    'CPU timing snapshot must retain its measured partial scope');
  check(cpu.final_cpu_execution_evidence_url==='data/postprocessing_execution.json','CPU timing link must be portable and explicit');
  check((await page.locator('#cpu-evidence-link').getAttribute('href'))===cpu.final_cpu_execution_evidence_url,'Visible CPU timing evidence link differs');
  check((await page.locator('#launch-link').getAttribute('href'))==='data/launch.json','Visible launch download must be local');
  const launchFile=localPath(root,'data/launch.json'),launch=await readJson(launchFile);
  check(await hashFile(launchFile)===configuration.launch_json_sha256,'Copied original launch byte SHA differs');
  assert.deepStrictEqual(launch,configuration.launch_metadata,'Copied launch metadata differs from bound source evidence');
  check(launch.RunName===payload.run_id&&launch.Clips===8&&launch.ExpectedConditions===48&&launch.FramesPerClip===128&&launch.AllFrames===true,
    'Original launch evidence must describe the complete 8clip/128frame campaign');
  const files=configuration.configuration_files;
  const expectedFiles=[['configs/loss_components_8clips_allframes.json','ConfigSHA256'],
    ['docker/manifests/loss_components_8clips_allframes.json','ManifestSHA256']];
  const configurationRows=await page.locator('#configuration-table tbody tr').allTextContents();
  check(configurationRows.length===2&&Object.keys(files).length===2,'Both original configuration files need visible byte-hash proof');
  for(const [i,[relative,key]] of expectedFiles.entries()) {
    const evidence=files[relative];
    check(evidence&&evidence.sha256===launch[key]&&evidence.matched_original_launch===true&&evidence.hash_kind==='literal file bytes',
      `Original launch does not bind ${relative}`);
    check(configurationRows[i].includes(relative)&&configurationRows[i].includes(evidence.sha256)
      &&configurationRows[i].includes(String(evidence.bytes)),`Visible original config proof differs: ${relative}`);
  }
  const execution=await readJson(localPath(root,cpu.final_cpu_execution_evidence_url));
  if(execution.status==='snapshot_before_browser_qa') assert.deepStrictEqual(execution,cpu,'Linked pre-QA CPU snapshot differs from HTML data');
  else check(execution.status==='cpu_stages_complete'&&execution.run_signature===payload.run_signature&&execution.run_id===payload.run_id
    &&execution.cpu_only===true&&execution.new_model_inference===false&&execution.numerical_experiment_modified===false,
    'Completed CPU evidence must retain the same run and CPU-only execution scope');
  check(Array.isArray(cpu.stages)&&cpu.stages.every(stage=>stage.status==='complete'&&stage.cpu_only===true
    &&typeof stage.elapsed_seconds==='number'&&Number.isFinite(stage.elapsed_seconds)&&stage.elapsed_seconds>=0),
    'HTML CPU snapshot must include successfully measured CPU stages only');
  await numericKeyRows(page,'#cpu-timing-table',[...cpu.stages.map(stage=>({stage:stage.stage,
    fields:[{key:'elapsed_seconds',value:stage.elapsed_seconds,digits:3}]})),
    {stage:'builder_partial',fields:[{key:'elapsed_seconds',value:cpu.builder_elapsed_until_html_content_snapshot_seconds,digits:3}]}],
    'completed CPU stages and partial builder duration');
  return {launch_sha256:configuration.launch_json_sha256,configuration_files:expectedFiles.length,
    cpu_evidence_status:execution.status,builder_partial_seconds:cpu.builder_elapsed_until_html_content_snapshot_seconds,cpu_timing_rows:cpu.stages.length+1};
}
async function interpretationCase(page,payload,row,dataset) {
  const value=payload.interpretation,reference=value.cases.find(r=>r.dataset===row.dataset&&r.sequence===row.sequence&&r.objective===row.objective);
  check(reference,'Selected case interpretation is missing');
  await interpretationParagraphs(page,'#case-interpretation',reference.summary,[row.dataset,row.sequence,row.objective].join('/'));
  const diagnostics=value.diagnostics_by_group[dataset];
  const diagnosticParagraphs=await interpretationParagraphs(page,'#diagnostic-interpretation',
    [diagnostics[OBJECTIVES.indexOf(row.objective)],diagnostics.at(-1)],dataset+'/'+row.objective+'/native diagnostics');
  check(diagnosticParagraphs.every(r=>r.dataset===dataset)&&diagnosticParagraphs[0].objective===row.objective,
    'Native diagnostic interpretation must follow both case objective and group');
  const timeline=reference.tracking_timeline_summary,norm=reference.normalization_summary;
  check(timeline.frame_count===128&&timeline.all_evaluation_query_times===true
    &&timeline.evaluation_query_times===128*timeline.query_count
    &&timeline.increased_frames+timeline.equal_frames+timeline.improved_frames===128,
    'Selected temporal interpretation omitted full query/time support');
  await numericKeyRows(page,'#temporal-segment-table',timeline.fixed_segments.map(segment=>({fields:[
    ...['source_frame_start','source_frame_end','frame_count'].map(key=>({key,value:segment[key],digits:0})),
    ...['clean_epe_mean_m','attack_epe_mean_m'].map(key=>({key,value:segment[key],digits:3})),
    {key:'delta_epe_mean_m',value:segment.delta_epe_mean_m,digits:3,signed:true}]})),row.objective+'/fixed 32-frame segments');
  await numericKeyRows(page,'#normalization-summary-table',['clean','attack','fixed_gt'].map(state=>({state,
    fields:['mean','median','min','max'].map(key=>({key,value:norm[state][key],digits:3}))})),row.objective+'/raw normalization magnitudes');
  await numericKeyRows(page,'#normalization-ratio-table',[{fields:[{key:'ratio_of_means',value:norm.ratio_of_means,digits:3},
    ...['mean','median','min','max'].map(key=>({key,value:norm.framewise_ratio[key],digits:3})),
    ...['valid_frames','excluded_frames'].map(key=>({key,value:norm.framewise_ratio[key],digits:0}))]}],row.objective+'/normalization ratios and denominator support');
  const nativeRows=await page.locator('#diagnostic-table tbody tr').evaluateAll(nodes=>nodes.map(tr=>Array.from(tr.cells).map(cell=>cell.textContent.trim())));
  const fields=['tracking_l21','reconstruction_l21','track_raw_conf_mean','track_conf_mean','reconstruction_conf_mean'];
  assert.deepStrictEqual(nativeRows,[['clean',...fields.map(key=>fmt(row.clean_diagnostics[key],3)),'기준 입력'],
    [row.objective,...fields.map(key=>fmt(row.loss_terms.diagnostics[key],3)),row.selected_state.restart+'/'+row.selected_state.step]],
    'Selected native/confidence diagnostic values differ from original result scalars');
  return {normalization_rows:3,ratio_rows:1,temporal_segments:4,temporal_frames:128,evaluation_queries:timeline.query_count};
}
async function interpretationBudget(page,payload) {
  const budget=payload.interpretation.input_budget;
  await interpretationParagraphs(page,'#input-budget',budget.summary,'all40 epsilon/selected-state budget');
  await numericKeyRows(page,'#budget-table',[{fields:[
    ...['max_linf','epsilon_linf','validation_atol'].map(key=>({key,value:budget[key],digits:8})),
    {key:'max_linf_255',value:budget.max_linf_255,digits:6},
    ...['within_budget','attacks'].map(key=>({key,value:budget[key],digits:0})),
    {key:'within_budget_passed',raw:budget.within_budget===budget.attacks&&budget.attacks===40,
      text:budget.within_budget===budget.attacks&&budget.attacks===40?'통과':'실패'}]}],'input L-infinity budget');
  const states=[['clean','selected_clean_count'],['initialization','selected_initial_count'],
    ['early','selected_early_count'],['final','selected_final_count']];
  await numericKeyRows(page,'#selected-state-table',states.map(([state,key])=>({state,fields:[{key:'count',value:budget[key],digits:0}]})),
    'selected clean/random/early/final state support');
  check(states.reduce((sum,[,key])=>sum+budget[key],0)===40&&budget.within_budget===40&&budget.attacks===40,
    'Input-budget and selected-state counts must cover all 40 independent attacks');
  return {within_budget:budget.within_budget,attacks:budget.attacks,max_linf:budget.max_linf};
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
    const interpretation=await interpretationGroup(page,payload,'all');
    const budget=await interpretationBudget(page,payload);
    const linkedEvidence=await linkedCpuAndLaunchEvidence(page,root,payload);
    steps.push({check:'interpretation_budget_launch_and_cpu_metadata',passed:true,...interpretation,...budget,...linkedEvidence});
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
      const interpretation=await interpretationGroup(page,payload,dataset);
      const selectedId=await page.locator('#clip-select').inputValue(),objective=await page.locator('#objective-select').inputValue();
      const selectedRow=payload.conditions.find(row=>row.dataset+'/'+row.sequence===selectedId&&row.objective===objective);
      await interpretationCase(page,payload,selectedRow,dataset);
      steps.push({check:'dataset_filter_numeric_rows_and_summaries',dataset,count:expected.length,passed:true,...summary,...interpretation});
    }
    await allOption(page,'#detail-objective');await allOption(page,'#detail-clip');
    await page.selectOption('#detail-objective','tracking_3d');
    await numericRows(page,payload.conditions.filter(row=>row.objective==='tracking_3d'));
    check((await interpretationGroup(page,payload,'all')).consistency_rows===4,'Clip consistency table did not follow detail objective filter');
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
      await interpretationCase(page,payload,row,'all');
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
