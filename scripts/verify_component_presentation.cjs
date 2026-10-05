#!/usr/bin/env node
'use strict';

// Read-only browser QA of one portable presentation. Evidence and an isolated
// renamed copy are written only to the explicitly supplied scratch directory.
// No experiment, renderer, model, Docker, or network client is invoked.
const fs = require('node:fs/promises');
const path = require('node:path');
const crypto = require('node:crypto');
const {pathToFileURL} = require('node:url');

const ORIGINAL_OBJECTIVES = ['tracking_mse','tracking_3d','reconstruction_3d','confidence','joint_training'];
const OBJECTIVES = ['tracking_3d','reconstruction_3d','confidence','joint_training'];
const CLIPS = [
  {prefix:'po01',dataset:'po_mini',sequence:'cab_e_3rd_13'},
  {prefix:'dr01',dataset:'ds_mini',sequence:'9c43b3-3_obj_source_left_3'},
  {prefix:'po02',dataset:'po_mini',sequence:'dancingroom1_3rd_7'},
  {prefix:'dr02',dataset:'ds_mini',sequence:'e09bca-3_obj_source_left_2'},
  {prefix:'po03',dataset:'po_mini',sequence:'dancingroom1_3rd2_4'},
  {prefix:'dr03',dataset:'ds_mini',sequence:'fec654-3_obj_source_left_4'},
  {prefix:'po04',dataset:'po_mini',sequence:'seminar_g110_0315_ego2_3'},
  {prefix:'dr04',dataset:'ds_mini',sequence:'423875-3_obj_source_left_0'}
];
const TASKS = ['tracking','reconstruction'];
const SLIDE_COUNT=24;
const CASES=CLIPS.flatMap(row=>OBJECTIVES.map(objective=>({...row,objective})));
const GROUPS=CLIPS.flatMap((row,index)=>TASKS.map((task,taskIndex)=>({...row,id:`${row.prefix}_${task}`,task,slide_index:7+index*2+taskIndex,
  media_ids:OBJECTIVES.map(objective=>`${row.prefix}_${objective}_${task}`)})));
const MEDIA_IDS = CASES.flatMap(row=>TASKS.map(task=>`${row.prefix}_${row.objective}_${task}`));
const METRICS = ['tracking_apd_drop_pp','reconstruction_apd_drop_pp','tracking_epe_increase_m','reconstruction_epe_increase_m'];
const BAR_METRICS = ['tracking_apd_drop_pp_attacked','reconstruction_apd_drop_pp_attacked'];
const LOSS_EXPRESSIONS = [
  'mean(c1_eff * e1)','mean(C2 * e2)',
  '-0.2 * (mean log max(c1_eff,1) + mean log C2)',
  'tracking_3d + reconstruction_3d + confidence'
];
const DISPLAY_EXPRESSIONS = [
  'mean(c₁ × e₁)','mean(C₂ × e₂)',
  '−0.2[mean log max(c₁,1) + mean log C₂]',
  'Tracking 기하 + Reconstruction 기하 + Confidence'
];
const sha = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const currentUtc = () => new Date().toISOString();
const check = (condition, message) => {if(!condition) throw new Error(message);};
const finite = value => typeof value === 'number' && Number.isFinite(value);
const close = (actual, expected, message, atol=1e-6) => check(finite(actual)&&finite(expected)&&Math.abs(actual-expected)<=atol, message+`: ${actual} vs ${expected}`);
const metricClose = (actual,expected,message) => close(actual,expected,message,1e-6+1e-6*Math.abs(expected));
const normalizeText = text => String(text).replace(/\s+/g,' ').trim();
const identity = row => `${row.dataset}/${row.sequence}/${row.objective}`;
const hashString = value => typeof value==='string'&&/^[a-f0-9]{64}$/.test(value);
const exactFrames = value => Array.isArray(value)&&value.length===128&&value.every((frame,index)=>frame===index);
const canonical = value => Array.isArray(value)?value.map(canonical):value&&typeof value==='object'?
  Object.fromEntries(Object.keys(value).sort().map(key=>[key,canonical(value[key])])):value;
const same = (first,second) => JSON.stringify(canonical(first))===JSON.stringify(canonical(second));
const safeRelative = value => typeof value==='string'&&value.length>0&&!path.isAbsolute(value)
  &&!/[\\]/.test(value)&&!value.split('/').some(part=>part==='..'||part==='.'||part==='')&&!/^[a-z]+:/i.test(value);

function argumentsFrom(argv) {
  const required = ['presentation','qa-dir','playwright-module','browser-executable','report-dir','visualization-dir'];
  const allowed = new Set(required);
  const result = {};
  for(let i=0;i<argv.length;i++) {
    if(argv[i]==='--help'||argv[i]==='-h') return {help:true};
    check(argv[i].startsWith('--')&&allowed.has(argv[i].slice(2)), `Unknown argument ${argv[i]}`);
    const key=argv[i].slice(2);
    check(!(key in result)&&i+1<argv.length&&!argv[i+1].startsWith('--'), `Missing or repeated argument ${argv[i]}`);
    result[key]=argv[++i];
  }
  for(const key of required) check(key in result&&path.isAbsolute(result[key]), `--${key} must be an absolute path`);
  return result;
}

function validateData(data) {
  check(data&&typeof data==='object', 'Presentation data JSON is missing');
  check(data.presentation_revision===4&&data.visualization_revision===4&&data.rgb_absolute_difference_included===false,
    'Presentation must use task result pairs, with no absolute RGB difference video');
  const scope=data.scope||{};
  check(scope.kind==='focused_campaign'&&scope.clips===8&&scope.num_frames===128&&scope.steps===20&&scope.epsilon_255===4
    &&scope.expected_conditions===40&&scope.focused_expected_conditions===40&&scope.full_campaign_expected_conditions===48
    &&scope.clean_conditions===8&&scope.attack_conditions===32&&scope.all_frames===true,
    'Public presentation scope must be clean8 + four attacks×8 clips =40 conditions, with all128 frames');
  const sourceScope=data.source_scope||{};
  check(sourceScope.kind==='full_campaign'&&sourceScope.expected_conditions===48&&sourceScope.clips===8&&sourceScope.num_frames===128
    &&sourceScope.all_frames===true&&same(sourceScope.objectives,ORIGINAL_OBJECTIVES),
    'Original 48-condition / five-objective provenance must remain separate from public four-objective focus');
  check(JSON.stringify(scope.dataset_counts)===JSON.stringify({po_mini:4,ds_mini:4})
    ||scope.dataset_counts?.po_mini===4&&scope.dataset_counts?.ds_mini===4&&Object.keys(scope.dataset_counts).length===2,
    'Presentation must contain PO4 and DR4');
  check(same(scope.objectives,OBJECTIVES)&&same(data.objectives,OBJECTIVES), 'Four public attack objectives or their order differ');
  check(Array.isArray(data.clips)&&same(data.clips.map(({prefix,dataset,sequence})=>({prefix,dataset,sequence})),CLIPS),
    'All eight exact clips in original interleaved PO/DR order are required');
  for(const [index,clip] of data.clips.entries())check(clip.id===CLIPS[index].prefix&&clip.global_clip_index===index+1
    &&clip.source_manifest_index===index&&clip.clip_index===Math.floor(index/2)+1&&clip.display_group===(index%2?'DR':'PO')
    &&exactFrames(clip.source_frame_indices)&&hashString(clip.source_sequence_npz_sha256),'Canonical clip descriptor or all-frame/source provenance differs');
  check(data.sources&&typeof data.sources==='object'&&Object.keys(data.sources).length>0, 'Presentation source provenance is absent');
  check(Array.isArray(data.aggregates), 'Presentation aggregates must be an array');
  const aggregates=data.aggregates.filter(row=>row.dataset===undefined||row.dataset==='all');
  check(aggregates.length===4&&new Set(aggregates.map(r=>r.objective)).size===4
    &&same(aggregates.map(r=>r.objective),OBJECTIVES), 'Four unique ordered overall objective aggregates are required');
  for(const row of aggregates) for(const metric of METRICS) check(finite(row[metric]), `Nonfinite aggregate ${row.objective}/${metric}`);
  for(const row of aggregates) for(const metric of BAR_METRICS) {
    check(finite(row[metric])&&row[metric]>=0&&row[metric]<=100, `Post-attack APD must be a finite percentage in [0,100]: ${row.objective}/${metric}`);
    const change=metric.slice(0,-'_attacked'.length),clean=row[change+'_clean'];
    check(finite(clean)&&clean>=0&&clean<=100,`Clean APD reference is absent or invalid: ${row.objective}/${metric}`);
    close(row[change],clean-row[metric],`Aggregate APD clean/attacked/change mapping differs: ${row.objective}/${metric}`);
  }
  check(Array.isArray(data.losses)&&data.losses.length===4, 'Four public loss definitions are required');
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
  check(Array.isArray(data.cases)&&data.cases.length===32&&new Set(data.cases.map(identity)).size===32
    &&CASES.every(row=>data.cases.some(item=>identity(item)===identity(row))), 'Exactly32 cases: eight same clips × four objectives are required');
  check(Array.isArray(data.comparison_groups)&&same(data.comparison_groups.map(row=>Object.fromEntries(Object.keys(GROUPS[0]).map(key=>[key,row[key]]))),GROUPS),
    'Sixteen same-clip/same-task ordered four-objective groups and slide indices7..22 are required');
  check(data.scatter&&data.scatter.points?.length===32&&same(data.scatter.excluded,[])&&data.scatter.connections?.length===8
    &&new Set(data.scatter.points.map(identity)).size===32&&CASES.every(row=>data.scatter.points.some(point=>identity(point)===identity(row))),
    'Scatter must use32 points, eight clip connections and no excluded/missing cases');
  for(const point of data.scatter.points)check(OBJECTIVES.includes(point.objective)&&finite(point.x)&&finite(point.y),'Scatter includes an absent objective or invalid coordinate');
  for(const row of data.cases) for(const metric of METRICS) {
    const clean=row[metric+'_clean'],attacked=row[metric+'_attacked'];
    check(finite(row[metric])&&finite(clean)&&finite(attacked),'Case metric must be a finite original scalar: '+identity(row)+'/'+metric);
    close(row[metric],metric.includes('apd')?clean-attacked:attacked-clean,'Case metric sign or clean/attacked mapping differs: '+identity(row)+'/'+metric);
  }
  check(data.visualization?.schema_version===2&&data.visualization?.status==='complete'&&data.visualization.media&&typeof data.visualization.media==='object',
    'Completed task visualization manifest is required');
  const visual=data.visualization,preservation=visual.original_report_preservation;
  check(same(visual.objectives,OBJECTIVES)&&same(visual.clip_order,data.clips)&&same(visual.rendered_clip_order,data.clips)
    &&visual.plain_video===true&&same(visual.video_size_wh,{tracking:[1024,288],reconstruction:[1600,600]}),
    'Renderer must include all eight clips/four objectives and the fixed plain video sizes');
  for(const key of ['text_overlay','tracking_query_ids_drawn','heatmaps_in_video','charts_in_video','legends_in_video',
    'numbers_overlay','axis_labels_in_video','ticklabels_in_video'])check(visual[key]===false,'Plain renderer contains forbidden video annotation: '+key);
  check(visual.scope?.clips===8&&visual.scope.full_campaign_clips===8&&visual.scope.preview_only===false
    &&visual.scope.expected_full_media===64&&visual.scope.cases===32&&visual.scope.comparison_groups===16
    &&visual.scope.original_completed_conditions===48&&visual.scope.focus_conditions_with_clean===40
    &&visual.scope.focus_attack_conditions===32&&visual.scope.rendered_focus_conditions_with_clean===40
    &&same(visual.scope.excluded_presentation_objectives,['tracking_mse']),
    'Renderer focus/source/public coverage is incomplete or ambiguous');
  check(visual.cpu_only===true&&visual.new_model_inference===false&&visual.GPU_used===false
    &&visual.frame_count===128&&visual.fps===10&&exactFrames(visual.source_frame_indices)
    &&visual.rgb_difference_panels===false&&visual.preview_fps_is_capture_fps===false&&visual.run_id===data.run_id
    &&hashString(visual.run_signature)&&hashString(visual.analysis_sha256)&&same(visual.errors,[]),
    'Task renderer must preserve the completed run and all128 frames with CPU-only, no-inference provenance');
  check(preservation?.passed===true&&Number.isInteger(preservation.files)&&preservation.files>0
    &&hashString(preservation.before_inventory_sha256)&&preservation.before_inventory_sha256===preservation.after_inventory_sha256
    &&preservation.key_files?.['report_manifest.json']?.sha256===data.sources.report_manifest_sha256
    &&preservation.key_files?.['data/report_data.json']?.sha256===data.sources.report_data_sha256,
    'Original detailed report preservation evidence or numerical source SHA differs');
  const codeNames=['render_component_task_results.py','component_projection.py','component_report_data.py',
    'render_component_comparisons.py','render_tracking_comparisons.py','render_pgd_comparisons.py','tracking_projection.py'];
  check(same(Object.keys(visual.source_code_sha256||{}).sort(),[...codeNames].sort())
    &&Object.values(visual.source_code_sha256).every(hashString),'Task-renderer and immutable adapter code SHA inventory differs');
  check(visual.source_freeze_checks?.passed===true&&visual.source_freeze_checks.renderer_and_imports_unchanged===true
    &&visual.source_freeze_checks.verified_scalar_inputs_unchanged===true&&visual.source_freeze_checks.hashed_source_files_metadata_unchanged===true
    &&Number.isInteger(visual.source_freeze_checks.hashed_source_files_count)&&visual.source_freeze_checks.hashed_source_files_count>0
    &&visual.source_freeze_checks.source_evidence_bundle_sha256===sha(Buffer.from(JSON.stringify(canonical(
      Object.fromEntries(Object.entries(visual.media).map(([id,row])=>[id,row.source_evidence])))))),
    'Renderer completed source freeze evidence is missing or stale');
  check(hashString(data.sources.visualization_manifest_sha256)&&data.sources.generated_assets,
    'Visualization manifest provenance and generated asset hashes are required');
  check(data.media&&typeof data.media==='object'&&same(Object.keys(data.media).sort(),[...MEDIA_IDS].sort())
    &&same(Object.keys(data.visualization.media).sort(),[...MEDIA_IDS].sort()), 'All64 exact clip/attack/task media IDs are required');
  for(const id of MEDIA_IDS) {
    const media=data.media[id],probe=media.probe||{};
    const expected=CASES.find(row=>id.startsWith(row.prefix+'_'+row.objective+'_'));
    const task=id.slice((expected.prefix+'_'+expected.objective+'_').length),raw=media.source_render;
    check(raw?.plain_video===true,'Task source movie is not a plain visualization: '+id);
    for(const key of ['text_overlay','tracking_query_ids_drawn','heatmaps_in_video','charts_in_video','legends_in_video',
      'numbers_overlay','axis_labels_in_video','ticklabels_in_video'])check(raw[key]===false,'Source video contains a forbidden annotation: '+id+'/'+key);
    check(media.id===id&&identity(media)===identity(expected)&&media.task===task&&TASKS.includes(task), `Media identity differs: ${id}`);
    check(raw&&same(raw,data.visualization.media[id])&&raw.id===id&&identity(raw)===identity(expected)&&raw.task===task,
      `Source renderer record is missing, replaced, or assigned to another case/task: ${id}`);
    check(safeRelative(raw.path)&&/\.mp4$/i.test(raw.path)&&media.path==='task_visualization/'+raw.path
      &&hashString(media.sha256)&&Number.isInteger(media.bytes)&&media.bytes>0, `Media source path/size/SHA is absent or unsafe: ${id}`);
    check(probe.decoded_frames===128&&probe.codec==='h264'&&media.cpu_only===true,
      `Media lacks CPU ffprobe decoded128-frame proof: ${id}`);
    check(hashString(raw.sha256)&&media.recipe?.source_sha256===raw.sha256
      &&raw.sha256===data.sources.generated_assets[media.path],
      `Encoded video recipe is not bound to the fresh source-rendered task video: ${id}`);
    check(exactFrames(raw.source_frame_indices)&&raw.probe?.decoded_frames===128,
      `Renderer did not preserve every source frame 0 through127: ${id}`);
    check(raw.probe.success===true&&raw.probe.method==='ffprobe_count_frames'&&raw.probe.source_sha256===raw.sha256,
      `Raw task renderer ffprobe is not bound to its own movie bytes: ${id}`);
    close(raw.probe.fps,10,`Source task video fps differs: ${id}`);
    close(raw.probe.duration,12.8,`Source task video duration differs: ${id}`,0.05);
    check(raw.source_evidence&&typeof raw.source_evidence==='object'&&Object.keys(raw.source_evidence).length>0,
      `Task video lacks immutable original source evidence: ${id}`);
    const evidence=raw.source_evidence;
    check(evidence.sequence?.source_npz?.sha256===data.clips.find(clip=>clip.prefix===expected.prefix).source_sequence_npz_sha256,
      'Movie source sequence differs from its canonical clip descriptor: '+id);
    check(same(Object.keys(evidence).sort(),['attack','clean','sequence']),`Task source evidence groups differ: ${id}`);
    check(raw.source_evidence_sha256===sha(Buffer.from(JSON.stringify(canonical(evidence)))),'Task source evidence bundle hash differs: '+id);
    const expectedEvidence={sequence:['source_npz','sequence.json','targets.npz','reconstruction_gt.npy','reconstruction_valid.npy'],
      clean:['result.json','tracks.npz','rgb_float32.npy','delta_float32.npy','reconstruction_float32.npy','reconstruction_confidence_float32.npy','components.npz','history.json'],
      attack:['result.json','tracks.npz','rgb_float32.npy','delta_float32.npy','reconstruction_float32.npy','reconstruction_confidence_float32.npy','components.npz','history.json']};
    for(const [group,names] of Object.entries(expectedEvidence)) {
      check(same(Object.keys(evidence[group]||{}).sort(),[...names].sort()),`Immutable saved source inventory differs: ${id}/${group}`);
      for(const name of names) {
        const leaf=evidence[group][name],filename=name==='source_npz'?expected.sequence+'.npz':name;
        const normalized=typeof leaf.path==='string'?leaf.path.replace(/\\/g,'/'):'';
        const suffix=group==='sequence'?(name==='source_npz'?`/data/worldtrack_release/${expected.dataset}/${filename}`:
          `/sequences/${expected.dataset}/${expected.sequence}/${filename}`):
          `/conditions/${group==='clean'?'clean':expected.objective}/${expected.dataset}/${expected.sequence}/${filename}`;
        check(hashString(leaf.sha256)&&path.isAbsolute(leaf.path||'')&&normalized.endsWith(suffix)
          &&(leaf.bytes===undefined||Number.isInteger(leaf.bytes)&&leaf.bytes>0),`Original array/scalar source SHA identity differs: ${id}/${group}/${name}`);
      }
    }
    const caseRow=data.cases.find(row=>identity(row)===identity(expected));
    check(same(raw.selected_state,caseRow.selected_state),`Movie is not from the original selected attack iterate: ${id}`);
    const metricKeys=task==='tracking'?{apd:'apd3d_all',epe:'epe_all_m'}:{apd:'apd3d',epe:'epe_m'};
    for(const state of ['clean','attack']) {
      const suffix=state==='clean'?'clean':'attacked',recorded=raw.case_metrics?.[state],reproduced=raw.metric_reproduction?.[state];
      check(recorded&&reproduced,`Saved and reproduced task metrics are absent: ${id}/${state}`);
      close(recorded[metricKeys.apd],caseRow[task+'_apd_drop_pp_'+suffix],`Movie's saved APD differs from the plotted original result: ${id}/${state}`);
      close(recorded[metricKeys.epe],caseRow[task+'_epe_increase_m_'+suffix],`Movie's saved EPE differs from the plotted original result: ${id}/${state}`);
      if(task==='tracking') {
        check(reproduced.tracking_metrics_reproduced===true&&reproduced.recorded_model_replay_passed===true,`Tracking reproduction/replay evidence did not pass: ${id}/${state}`);
        metricClose(reproduced.tracking_apd_reproduced_percent,recorded[metricKeys.apd],`Reproduced Tracking APD differs: ${id}/${state}`);
        metricClose(reproduced.tracking_epe_reproduced_m,recorded[metricKeys.epe],`Reproduced Tracking EPE differs: ${id}/${state}`);
      } else {
        metricClose(reproduced.apd3d,recorded[metricKeys.apd],`Reproduced Reconstruction APD differs: ${id}/${state}`);
        metricClose(reproduced.epe_m,recorded[metricKeys.epe],`Reproduced Reconstruction EPE differs: ${id}/${state}`);
        check(reproduced.passed===true&&reproduced.valid_pixels===recorded.valid_pixels,`Reconstruction GT support changed: ${id}/${state}`);
      }
    }
    check(raw.posters&&same(Object.keys(raw.posters).sort(),['0','127','64']), `Three fixed renderer poster frames are required: ${id}`);
    for(const frame of ['0','64','127']) {
      const poster=raw.posters[frame];
      check(poster&&safeRelative(poster.path)&&/\.png$/i.test(poster.path)&&hashString(poster.sha256)
        &&Number.isInteger(poster.bytes)&&poster.bytes>0, `Poster source proof differs: ${id}/${frame}`);
    }
    close(probe.fps,10, `Video is not a 10fps preview: ${id}`);
    close(probe.duration,12.8, `Video duration is not 128/10 seconds: ${id}`,0.05);
    check(Number.isInteger(probe.width)&&probe.width>0&&Number.isInteger(probe.height)&&probe.height>0, `Media dimensions are invalid: ${id}`);
    check(probe.width===raw.probe.width&&probe.height===raw.probe.height&&media.recipe.resize===false,
      `Presentation transcode changed the source task-video resolution: ${id}`);
    const size=task==='tracking'?[1024,288]:[1600,600];
    check(probe.width===size[0]&&probe.height===size[1],`Plain video task resolution differs: ${id}`);
    if(task==='reconstruction') {
      const meta=raw.common_display_metadata;
      check(meta.raster_stride===8&&meta.selection_uses_model_errors===false&&meta.selection_uses_confidence===false
        &&meta.point_coordinates_clamped===false&&meta.global_inset_clipping===false&&meta.clipping===false
        &&meta.all_selected_points_retained_in_global_inset===true&&meta.main_detail_crops_to_GT_bounds===true
        &&same(meta.comparison_objectives,OBJECTIVES),'Reconstruction display must preserve GT-selected support, shared bounds and disclosed detail crop');
      for(const condition of ['GT','clean','attack'])check(Array.isArray(raw.display_outside_count?.[condition])
        &&raw.display_outside_count[condition].length===128&&raw.display_outside_count[condition].every(n=>Number.isInteger(n)&&n>=0),
        'Reconstruction outside counts must retain all128 frame positions: '+id+'/'+condition);
    }
  }
  for(const row of CASES) {
    const tracking=data.media[`${row.prefix}_${row.objective}_tracking`].source_render;
    const reconstruction=data.media[`${row.prefix}_${row.objective}_reconstruction`].source_render;
    check(same(tracking.source_evidence,reconstruction.source_evidence)&&same(tracking.selected_state,reconstruction.selected_state)
      &&same(tracking.clean_selected_state,reconstruction.clean_selected_state),
      'The task pair must show the same saved clean input and selected attack: '+identity(row));
  }
  for(const clip of CLIPS) for(const task of TASKS) for(const objective of OBJECTIVES.slice(1)) {
    const prefix=clip.prefix,a=data.media[`${prefix}_${OBJECTIVES[0]}_${task}`].source_render,b=data.media[`${prefix}_${objective}_${task}`].source_render;
    check(same(a.source_evidence.sequence,b.source_evidence.sequence)&&same(a.source_evidence.clean,b.source_evidence.clean)
      &&same(a.case_metrics.clean,b.case_metrics.clean)&&same(a.common_display_metadata,b.common_display_metadata),
      'All four objective examples must share GT, clean sources and common display limits: '+prefix+'/'+task);
  }
  return {aggregates,contrasts:overall};
}

async function waitImages(page) {
  await page.evaluate(()=>document.fonts.ready);
  await page.waitForFunction(()=>Array.from(document.images).every(img=>img.complete&&img.naturalWidth>0),{},{timeout:30000});
  const images=await page.locator('img').evaluateAll(nodes=>nodes.map(img=>({inline_png:/^data:image\/png;base64,/.test(img.getAttribute('src')||''),width:img.naturalWidth,height:img.naturalHeight})));
  check(images.length>0&&images.every(img=>img.inline_png&&img.width>0&&img.height>0), 'All presentation images must be loaded inline PNG assets');
  return {count:images.length,all_inline_loaded:true};
}

async function slideState(page,index) {
  const state=await page.evaluate(()=>({currentIndex:window.presentation.currentIndex,count:window.presentation.count,
    active:Array.from(document.querySelectorAll('section.slide.active')).map(e=>e.id),
    visible:Array.from(document.querySelectorAll('section.slide')).filter(e=>{
      const s=getComputedStyle(e),r=e.getBoundingClientRect();return s.display!=='none'&&s.visibility!=='hidden'&&Number(s.opacity)!==0&&r.width>0&&r.height>0;
    }).map(e=>e.id)}));
  check(state.count===SLIDE_COUNT&&state.currentIndex===index&&JSON.stringify(state.active)===JSON.stringify([`slide-${index+1}`])
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
    const targets=Array.from(active.querySelectorAll('h1,h2,h3,p,table,video,img,figure,ul,ol,.card,.chart,.metric-card,.panel,.case-values,.pair-controls,.pair-controls button,.pair-controls label,.pair-controls input,.pair-controls output'))
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

function validateBarGeometry(geometry,aggregates) {
  check(geometry.length===8,'Exactly eight post-attack APD bars are required');
  const pairs=new Set(),tracks=[];
  for(const row of geometry) {
    check(OBJECTIVES.includes(row.objective)&&BAR_METRICS.includes(row.metric),'Bar must use the actual attacked APD field, not a decline field');
    const key=row.objective+'/'+row.metric;check(!pairs.has(key),`Duplicate metric bar ${key}`);pairs.add(key);
    const expected=aggregates.find(a=>a.objective===row.objective)?.[row.metric];
    check(row.value===expected&&finite(row.value)&&row.value>=0&&row.value<=100,`Bar does not preserve exact post-attack aggregate APD: ${key}`);
    check(row.text===expected.toFixed(2)+'%',`Post-attack APD label must display the original value with percent units: ${key}`);
    check(row.slideIndex===4&&row.height>0&&row.trackWidth>0&&row.width>=0,'Post-attack APD bar is not visible on slide 5');
    close(row.width/row.trackWidth,row.value/100,`Bar does not use the common 0-100 percent scale: ${key}`,0.00015);
    close(row.stylePercent,row.value,`Bar inline width does not equal the original actual APD percent: ${key}`,1e-10);
    tracks.push(row.trackWidth);
  }
  const mean=tracks.reduce((a,b)=>a+b,0)/tracks.length;
  check(tracks.every(value=>Math.abs(value-mean)<=0.1),'Tracking and Reconstruction must have equal full-scale track widths');
  return {count:8,quantity:'post_attack_apd',unit:'percent',scale:[0,100],higher_is_better:true,
    exact_aggregate_binding:true,displayed_percent_labels_verified:true,proportional_lengths_verified:true,
    shared_task_scale_verified:true,geometry};
}

async function checkBars(page,aggregates) {
  const rows=await page.locator('[data-metric][data-value]').evaluateAll(nodes=>nodes.map(el=>({
    metric:el.dataset.metric,value:Number(el.dataset.value),objective:el.dataset.objective,
    slideIndex:Number(el.closest('section.slide').id.slice(6))-1})));
  check(rows.length===8,'Exactly eight metric bars are required (four objectives × two APD tasks)');
  const pairs=new Set();
  for(const row of rows) {
    check(OBJECTIVES.includes(row.objective)&&BAR_METRICS.includes(row.metric), 'Metric bar must use the attacked APD field');
    const key=row.objective+'/'+row.metric;check(!pairs.has(key),`Duplicate metric bar ${key}`);pairs.add(key);
    check(row.value===aggregates.find(a=>a.objective===row.objective)[row.metric], `Bar does not preserve the exact aggregate value: ${key}`);
  }
  await goTo(page,4);
  const geometry=await page.locator('#slide-5 [data-metric][data-value]').evaluateAll(nodes=>nodes.map(el=>({
    metric:el.dataset.metric,objective:el.dataset.objective,value:Number(el.dataset.value),
    slideIndex:Number(el.closest('section.slide').id.slice(6))-1,
    width:el.getBoundingClientRect().width,height:el.getBoundingClientRect().height,
    trackWidth:el.closest('.bar-track')?.getBoundingClientRect().width,
    stylePercent:Number(el.getAttribute('style')?.match(/(?:^|;)\s*width:\s*([\d.eE+-]+)%\s*(?:;|$)/)?.[1]),
    text:el.closest('.bar-cell')?.querySelector('.bar-value')?.textContent.trim()})));
  const result=validateBarGeometry(geometry,aggregates);
  const text=normalizeText(await page.locator('#slide-5').textContent());
  check(text.includes('공격 후 Tracking APD (%)')&&text.includes('공격 후 Reconstruction APD (%)'),
    'Slide 5 task headers must label actual post-attack APD percent');
  check(/0\s*[~–−-]\s*100%/.test(text)&&text.includes('높을수록')&&text.includes('좋'),
    'Slide 5 must explain the common 0-100 percent scale and higher-is-better interpretation');
  check(!text.includes('%p')&&!text.includes('APD 감소'), 'Slide 5 still labels actual APD as a percentage-point decline');
  const trackingClean=aggregates[0].tracking_apd_drop_pp_clean,reconClean=aggregates[0].reconstruction_apd_drop_pp_clean;
  check(aggregates.every(row=>row.tracking_apd_drop_pp_clean===trackingClean&&row.reconstruction_apd_drop_pp_clean===reconClean),
    'Slide 5 cannot use one shared clean reference if objective clean means differ');
  check(text.includes(trackingClean.toFixed(2)+'%')&&text.includes(reconClean.toFixed(2)+'%'),'Clean APD reference labels differ from original aggregate means');
  result.labels={task_headers_verified:true,higher_is_better:true,clean_tracking_apd_percent:trackingClean,
    clean_reconstruction_apd_percent:reconClean,percentage_point_decline_label_removed:true};
  return result;
}

async function checkLosses(page,data) {
  const rows=await page.locator('#loss-table tbody tr').evaluateAll(nodes=>nodes.map(row=>({objective:row.dataset.objective,
    id:row.querySelector('code')?.textContent,expression:row.cells[2]?.textContent,text:row.textContent})));
  check(rows.length===4,'Loss table must contain four public definitions');
  for(const [i,row] of rows.entries()) {
    const loss=data.losses[i];
    // The table typesets norms, indices and multiplication with Unicode and
    // HTML subscripts. Verify its independent mathematical presentation rather
    // than demanding that ASCII provenance formulas be displayed verbatim.
    const compact=value=>String(value).replace(/\s+/g,'');
    check(row.objective===loss.id&&normalizeText(row.id)===loss.id&&compact(row.expression)===compact(DISPLAY_EXPRESSIONS[i]),
      `Displayed loss ID/expression differs from embedded definition: ${loss.id}`);
  }
  return {rows:4,exact_ids_and_expressions:true};
}

async function checkCaseDOM(page,data) {
  const sections=await page.locator('section.slide[data-comparison]').evaluateAll(nodes=>nodes.map(el=>({
    id:el.id,group:el.dataset.comparison,case:el.dataset.case,dataset:el.dataset.dataset,sequence:el.dataset.sequence,task:el.dataset.task,
    media:Array.from(el.querySelectorAll('video')).map(video=>({id:video.dataset.media,task:video.dataset.task,objective:video.dataset.objective})),
    panels:Array.from(el.querySelectorAll('figure[data-objective]')).map(figure=>({objective:figure.dataset.objective,
      heading:figure.querySelector('h3')?.textContent,caption:figure.querySelector('.video-caption')?.textContent})),
    foot:el.querySelector('.foot')?.textContent,notes:el.dataset.notes,
    play:el.querySelectorAll('[data-play-pair]').length,pause:el.querySelectorAll('[data-pause-pair]').length,
    seek:Array.from(el.querySelectorAll('input[data-frame-seek]')).map(input=>({type:input.type,min:input.min,max:input.max,step:input.step}))
  })));
  check(sections.length===16,'Sixteen same-clip/task comparison slides must be present');
  for(const row of GROUPS) {
    const section=sections.find(item=>item.group===row.id);
    check(section?.id===`slide-${row.slide_index+1}`&&section.case===row.id&&section.dataset===row.dataset
      &&section.sequence===row.sequence&&section.task===row.task&&section.play===1&&section.pause===1&&section.seek.length===1,
      'Comparison slide identity or quartet playback controls differ: '+row.id);
    check(section.seek[0].type==='range'&&section.seek[0].min==='0'&&section.seek[0].max==='127'&&section.seek[0].step==='1',
      'Comparison slider must select actual source frames 0 through127: '+row.id);
    check(same(section.media,OBJECTIVES.map((objective,index)=>({id:row.media_ids[index],task:row.task,objective}))),
      'Each quartet must show the same clip/task and all four ordered attack objectives: '+row.id);
    check(section.panels.length===4&&same(section.panels.map(panel=>panel.objective),OBJECTIVES),'Four ordered objective panels are missing: '+row.id);
    for(const panel of section.panels) {
      const original=data.cases.find(item=>item.dataset===row.dataset&&item.sequence===row.sequence&&item.objective===panel.objective);
      check(panel.heading?.includes(panel.objective)&&new RegExp('step\\s*'+original.selected_state.step+'(?:\\s|$)').test(normalizeText(panel.heading)),
        'Outside movie heading does not identify the original selected attack step: '+row.id+'/'+panel.objective);
      check(normalizeText(panel.caption).startsWith(row.task==='tracking'?'Clean | PGD':'GT | Clean | PGD'),
        'Outside movie panel-order caption differs: '+row.id+'/'+panel.objective);
    }
    if(row.task==='tracking')check(section.foot?.includes('GT')&&section.foot.includes('Clean')&&section.foot.includes('PGD'),
      'Tracking trajectory color legend must be outside the plain movies: '+row.id);
    else check(section.foot?.includes('공통')&&section.foot.includes('inset')&&section.foot.includes('축 밖')
      &&section.notes?.includes('stride8')&&section.notes.includes('clamp'),
      'Shared reconstruction bounds/detail crop/global inset/fixed GT display support must be disclosed outside movies: '+row.id);
  }
  const fullscreenButtons=await page.locator('[data-video-fullscreen]').evaluateAll(nodes=>nodes.map(el=>({id:el.dataset.videoFullscreen,
    slide:el.closest('section.slide')?.id,text:el.textContent.trim()})));
  check(fullscreenButtons.length===64&&same(fullscreenButtons.map(row=>row.id).sort(),[...MEDIA_IDS].sort()),
    'All64 plain movies require an external, identity-bound fullscreen button');
  for(const row of fullscreenButtons)check(row.text.length>0&&row.slide===`slide-${GROUPS.find(group=>group.media_ids.includes(row.id)).slide_index+1}`,
    'Fullscreen button is mislabeled or assigned to another clip/task: '+row.id);
  const blocks=await page.locator('[data-task-values]').evaluateAll(nodes=>nodes.map(el=>({dataset:el.dataset.dataset,
    sequence:el.dataset.sequence,objective:el.dataset.objective,task:el.dataset.task,
    section:el.closest('section.slide').id,text:el.textContent,
    cells:Array.from(el.querySelectorAll('[data-key]')).map(cell=>({key:cell.dataset.key,digits:cell.dataset.digits,text:cell.textContent}))
  })));
  check(blocks.length===64&&new Set(blocks.map(row=>identity(row)+'/'+row.task)).size===64,
    'All64 independent clip/objective/task scalar blocks are required');
  for(const row of CASES) for(const task of TASKS) {
    const block=blocks.find(item=>identity(item)===identity(row)&&item.task===task),record=data.cases.find(item=>identity(item)===identity(row));
    const keys=[task+'_apd_drop_pp_clean',task+'_apd_drop_pp_attacked',task+'_epe_increase_m_clean',task+'_epe_increase_m_attacked'];
    const group=GROUPS.find(group=>group.prefix===row.prefix&&group.task===task);
    check(block?.section===`slide-${group.slide_index+1}`&&block.cells.length===4&&same(block.cells.map(cell=>cell.key).sort(),[...keys].sort()),
      'Original clean/attack APD and EPE must be displayed below their task video: '+identity(row)+'/'+task);
    for(const cell of block.cells) {
      const digits=cell.key.includes('_apd_')?2:3;
      check(cell.digits===String(digits)&&normalizeText(cell.text)===record[cell.key].toFixed(digits),
        'Displayed case/task scalar differs from the original saved result: '+identity(row)+'/'+cell.key);
    }
    check(/APD/.test(block.text)&&/EPE/.test(block.text)&&/%/.test(block.text)&&/m/.test(block.text),
      'Task values must identify APD percentage and EPE meter units: '+identity(row)+'/'+task);
  }
  const text=await page.locator('section.slide').evaluateAll(nodes=>nodes.map(el=>el.textContent+' '+(el.dataset.notes||'')+' '+
    Array.from(el.querySelectorAll('img,video')).map(item=>item.getAttribute('alt')||item.getAttribute('aria-label')||'').join(' ')).join('\n'));
  check(!/po_rgb|RGB\s*(?:차분|차이)|\|\s*δ\s*\|\s*[×*]\s*64|absolute\s*RGB\s*difference/i.test(text),
    'The deck still describes the removed absolute RGB difference visualization');
  check(!/tracking_mse|\bMSE\b|평가\s*정렬/.test(text),'Tracking MSE must be absent from public slides, notes and media labels');
  return {cases:32,comparison_groups:16,task_scalar_blocks:64,case_task_identity:true,exact_saved_clean_attack_scalars:true,
    absolute_rgb_difference_removed:true,tracking_mse_absent_from_public_presentation:true};
}

async function checkInlineImageProof(page,data) {
  const images=await page.locator('img').evaluateAll(async nodes=>{
    const digest=async url=>{
      const match=/^data:image\/png;base64,([A-Za-z0-9+/=\s]+)$/.exec(url||'');
      if(!match)throw new Error('Image is not an inline PNG');
      const binary=atob(match[1].replace(/\s+/g,'')),bytes=Uint8Array.from(binary,c=>c.charCodeAt(0));
      const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))).map(x=>x.toString(16).padStart(2,'0')).join('');
      return {sha256:hash,bytes:bytes.length};
    };
    return Promise.all(nodes.map(async el=>({slide:el.closest('section.slide')?.id,...await digest(el.getAttribute('src'))})));
  });
  const accepted=new Set([...Object.values(data.sources.selected_assets||{}),...Object.values(data.sources.generated_assets)]);
  check(images.every(row=>accepted.has(row.sha256)),'An inline PNG is not bound to the verified original plot or source-rendered poster');
  const cover=images.filter(row=>row.slide==='slide-1'),expected=data.media.po01_tracking_3d_reconstruction.source_render.posters['64'];
  check(cover.length===1&&cover[0].sha256===expected.sha256&&cover[0].bytes===expected.bytes,
    'Cover must show the actual frame64 Reconstruction result, without the removed RGB difference');
  const posters=await page.locator('video').evaluateAll(async nodes=>Promise.all(nodes.map(async el=>{
    const match=/^data:image\/png;base64,([A-Za-z0-9+/=\s]+)$/.exec(el.poster||'');
    if(!match)throw new Error('Video poster is not an inline PNG');
    const bytes=Uint8Array.from(atob(match[1].replace(/\s+/g,'')),c=>c.charCodeAt(0));
    const digest=await crypto.subtle.digest('SHA-256',bytes);
    return {id:el.dataset.media,sha256:Array.from(new Uint8Array(digest)).map(x=>x.toString(16).padStart(2,'0')).join(''),bytes:bytes.length};
  })));
  for(const row of posters) {
    const proof=data.media[row.id]?.source_render.posters['0'];
    check(proof&&row.sha256===proof.sha256&&row.bytes===proof.bytes,'Video poster does not match its own actual frame0: '+row.id);
  }
  return {count:images.length,source_hashes_verified:true,cover_reconstruction_frame:64,video_posters:posters,images};
}

async function checkNavigation(page) {
  const checks=[];
  await goTo(page,0);
  for(const [key,index] of [['ArrowRight',1],['ArrowLeft',0],['End',SLIDE_COUNT-1],['Home',0]]) {
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
  const items=page.locator('#overview [data-slide-index]');check(await items.count()===SLIDE_COUNT,'Overview must expose24 selectable slide indices');
  await page.locator('#overview [data-slide-index="5"]').click();await page.locator('#overview').waitFor({state:'hidden'});await slideState(page,5);
  checks.push({check:'overview_select_slide',slide_index:5,passed:true});
  const fullscreenApplicable=await page.evaluate(()=>Boolean(document.fullscreenEnabled&&document.documentElement.requestFullscreen));
  if(fullscreenApplicable) {
    await page.keyboard.press('f');await page.waitForFunction(()=>document.fullscreenElement!==null,{},{timeout:10000});
    await page.keyboard.press('f');await page.waitForFunction(()=>document.fullscreenElement===null,{},{timeout:10000});
    await page.locator('#fullscreen').click();await page.waitForFunction(()=>document.fullscreenElement!==null,{},{timeout:10000});
    await page.locator('#fullscreen').click();await page.waitForFunction(()=>document.fullscreenElement===null,{},{timeout:10000});
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
  check(metadata.controls===false&&!metadata.autoplay&&!metadata.error&&metadata.src.startsWith('blob:'),
    `Plain local Blob movie must hide native text/number controls and use the external controls: ${id}`);
  close(metadata.duration,reference.probe.duration,`Browser duration differs from frame-count proof: ${id}`,0.05);
  check(metadata.width===reference.probe.width&&metadata.height===reference.probe.height,`Browser video dimensions differ from source proof: ${id}`);
  await page.locator(`[data-video-fullscreen="${id}"]`).click();
  await page.waitForFunction(mediaId=>document.fullscreenElement===document.querySelector(`video[data-media="${mediaId}"]`),id,{timeout:10000});
  const fullscreen=await locator.evaluate(el=>({id:el.dataset.media,fullscreen:document.fullscreenElement===el,
    native_controls:el.controls,width:el.getBoundingClientRect().width,height:el.getBoundingClientRect().height,viewport_width:innerWidth,viewport_height:innerHeight}));
  check(fullscreen.fullscreen&&fullscreen.native_controls===false&&fullscreen.width>=fullscreen.viewport_width-2
    &&fullscreen.height>=fullscreen.viewport_height-2,'External movie fullscreen did not preserve a text-free fitted movie: '+id);
  await page.keyboard.press('Escape');await page.waitForFunction(()=>document.fullscreenElement===null,{},{timeout:10000});
  await locator.evaluate(async el=>{el.currentTime=0;let timer;try{await Promise.race([el.play(),new Promise((_,reject)=>{
    timer=setTimeout(()=>reject(new Error('video play() did not resolve')),20000);})]);}finally{clearTimeout(timer);}});
  await page.waitForFunction(mediaId=>document.querySelector(`video[data-media="${mediaId}"]`).currentTime>=0.3,id,{timeout:15000});
  const advanced=await locator.evaluate(el=>({current_time:el.currentTime,paused:el.paused,quality:el.getVideoPlaybackQuality?.()||null}));
  check(advanced.current_time>=0.3&&!advanced.paused,`Video did not actually advance: ${id}`);
  const lastTime=(reference.probe.decoded_frames-1)/reference.probe.fps;
  const playStart=(reference.probe.decoded_frames-2)/reference.probe.fps;
  const last=await locator.evaluate(async (el,{lastTime,playStart})=>{
    const started=performance.now();
    el.pause();el.__lastMediaTime=null;
    const play=async()=>{let timer;try{await Promise.race([el.play(),new Promise((_,reject)=>{
      timer=setTimeout(()=>reject(new Error('last-frame play() did not resolve')),20000);})]);}finally{clearTimeout(timer);}};
    const frame=new Promise((resolve,reject)=>{
      const timer=setTimeout(()=>reject(new Error('last decoded frame was not presented')),15000);
      const onFrame=(_now,metadata)=>{if(metadata.mediaTime>=lastTime-0.02){clearTimeout(timer);el.__lastMediaTime=metadata.mediaTime;resolve(metadata.mediaTime);}else el.requestVideoFrameCallback(onFrame);};
      if(typeof el.requestVideoFrameCallback!=='function'){clearTimeout(timer);return reject(new Error('requestVideoFrameCallback unavailable'));}
      el.requestVideoFrameCallback(onFrame);
    });
    const seek=new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(new Error('last-frame seek timeout')),15000);
      el.addEventListener('seeked',()=>{clearTimeout(timer);resolve();},{once:true});});
    // Chromium can report ended=true after seeking directly to the last frame,
    // and play() then rewinds to zero. Start at the penultimate source frame,
    // require a completed non-ended seek and verify play() did not rewind.
    el.currentTime=playStart;await seek;
    const seekTime=el.currentTime,seekEnded=el.ended;
    if(seekEnded||Math.abs(seekTime-playStart)>0.005)throw new Error('Penultimate-frame seek is ended or inaccurate; refusing full-sequence rewind');
    await play();const actualStart=el.currentTime;
    if(actualStart<playStart-0.005)throw new Error('Last-segment play() unexpectedly rewound to zero');
    await frame;
    await new Promise((resolve,reject)=>{if(el.ended)return resolve();const timer=setTimeout(()=>reject(new Error('video did not play to end')),15000);
      el.addEventListener('ended',()=>{clearTimeout(timer);resolve();},{once:true});});
    return {target_last_frame_time:lastTime,last_presented_media_time:el.__lastMediaTime,current_time:el.currentTime,ended:el.ended,
      seek_start_time:playStart,actual_seek_time:seekTime,seek_ended_before_play:seekEnded,actual_playback_start_time:actualStart,
      full_sequence_rewind:false,last_segment_wall_seconds:(performance.now()-started)/1000,
      quality:el.getVideoPlaybackQuality?{total_video_frames:el.getVideoPlaybackQuality().totalVideoFrames,dropped_video_frames:el.getVideoPlaybackQuality().droppedVideoFrames}:null};
  },{lastTime,playStart});
  check(last.ended&&last.last_presented_media_time>=lastTime-0.02,`Last frame/end playback failed: ${id}`);
  await locator.evaluate(async el=>{el.currentTime=0.2;let timer;try{await Promise.race([el.play(),new Promise((_,reject)=>{
    timer=setTimeout(()=>reject(new Error('video restart play() did not resolve')),20000);})]);}finally{clearTimeout(timer);}});
  await page.waitForFunction(mediaId=>!document.querySelector(`video[data-media="${mediaId}"]`).paused,id,{timeout:10000});
  await goTo(page,index===SLIDE_COUNT-1?0:SLIDE_COUNT-1);
  check(await locator.evaluate(el=>el.paused),`Leaving a slide did not pause its video: ${id}`);
  return {id,dataset:reference.dataset,sequence:reference.sequence,objective:reference.objective,task:reference.task,
    encoded_sha256:reference.sha256,source_render_sha256:reference.source_render.sha256,decoded_frame_proof:128,preview_fps:10,metadata,actual_playback:advanced,
    last_frame_playback:last,leaving_slide_pauses:true,external_fullscreen_button:fullscreen,fullscreen_escape_verified:true};
}

async function pairProof(page,row) {
  await goTo(page,row.slide_index);
  const ids=row.media_ids,section=page.locator(`#slide-${row.slide_index+1}`);
  await page.evaluate(ids=>Promise.all(ids.map(id=>window.presentation.loadVideo(id))),ids);
  await section.locator('[data-pause-pair]').click();
  const seek=section.locator('input[data-frame-seek]'),frames=[];
  // Starting at64 avoids treating the initial frame0/poster as proof that a
  // seek really worked. Browser readyState2 and !seeking prove a decoded current
  // frame; each individual movie's final-frame presentation is checked below.
  for(const frame of [64,127,0]) {
    await seek.evaluate((input,frame)=>{input.value=String(frame);input.dispatchEvent(new Event('input',{bubbles:true}));},frame);
    await page.waitForFunction(({ids,frame})=>ids.every(id=>{
      const video=document.querySelector(`video[data-media="${id}"]`);
      return video.paused&&!video.seeking&&video.readyState>=2&&Math.abs(video.currentTime-frame/10)<0.005;
    }),{ids,frame},{timeout:20000});
    const state=await section.locator('video').evaluateAll(nodes=>nodes.map(video=>({id:video.dataset.media,
      current_time:video.currentTime,ready_state:video.readyState,seeking:video.seeking,paused:video.paused,src:video.currentSrc})));
    check(await seek.inputValue()===String(frame)&&state.every(video=>video.src.startsWith('blob:')),
      'Paired source-frame slider or Blob decoding differs: '+identity(row));
    frames.push({source_frame:frame,target_time:frame/10,all_four_current_frames_decoded:true,videos:state});
  }
  // Exercise the actual range control's keyboard behavior independently from
  // its dispatched input path, and ensure it does not navigate the slide.
  await seek.focus();await seek.press('End');
  await page.waitForFunction(ids=>ids.every(id=>Math.abs(document.querySelector(`video[data-media="${id}"]`).currentTime-12.7)<0.005
    &&!document.querySelector(`video[data-media="${id}"]`).seeking),ids,{timeout:20000});
  check(await seek.inputValue()==='127'&&await page.evaluate(()=>window.presentation.currentIndex)===row.slide_index,
    'Range End key must seek all four objectives without changing the comparison slide');
  await seek.press('Home');
  await page.waitForFunction(ids=>ids.every(id=>document.querySelector(`video[data-media="${id}"]`).currentTime<0.005
    &&!document.querySelector(`video[data-media="${id}"]`).seeking),ids,{timeout:20000});
  check(await seek.inputValue()==='0','Range Home key did not select frame0');
  await section.locator('[data-play-pair]').click();
  await page.waitForFunction(ids=>ids.every(id=>{const video=document.querySelector(`video[data-media="${id}"]`);
    return !video.paused&&video.currentTime>=0.3;}),ids,{timeout:20000});
  const advancing=await section.locator('video').evaluateAll(nodes=>nodes.map(video=>({id:video.dataset.media,current_time:video.currentTime,paused:video.paused})));
  const drift=Math.max(...advancing.map(video=>video.current_time))-Math.min(...advancing.map(video=>video.current_time));
  check(drift<=0.11,'Together-play task videos drifted by more than approximately one preview frame: '+identity(row));
  await section.locator('[data-pause-pair]').click();
  const paused=await section.locator('video').evaluateAll(async nodes=>{
    const before=nodes.map(video=>({id:video.dataset.media,current_time:video.currentTime,paused:video.paused}));
    await new Promise(resolve=>setTimeout(resolve,200));
    return {before,after:nodes.map(video=>({id:video.dataset.media,current_time:video.currentTime,paused:video.paused}))};
  });
  check(paused.before.every(video=>video.paused)&&paused.after.every((video,index)=>video.paused&&Math.abs(video.current_time-paused.before[index].current_time)<0.005),
    'Quartet pause button failed to stop all four objective videos: '+row.id);
  await section.locator('[data-play-pair]').click();
  await page.waitForFunction(ids=>ids.every(id=>!document.querySelector(`video[data-media="${id}"]`).paused),ids,{timeout:20000});
  await goTo(page,0);
  check(await section.locator('video').evaluateAll(nodes=>nodes.every(video=>video.paused)),
    'Leaving a comparison slide did not pause all four objective videos: '+row.id);
  const cancellations=[];
  for(const action of ['pause','leave','seek','notes','overview']) {
    await goTo(page,row.slide_index);
    await page.evaluate(({id,action})=>{
      const section=document.querySelector(`section[data-comparison="${id}"]`);
      section.querySelector('[data-play-pair]').click();
      if(action==='pause')section.querySelector('[data-pause-pair]').click();
      if(action==='leave')window.presentation.goTo(0);
      if(action==='seek'){const seek=section.querySelector('[data-frame-seek]');seek.value='64';seek.dispatchEvent(new Event('input',{bubbles:true}));}
      if(action==='notes')document.getElementById('notes-button').click();
      if(action==='overview')document.getElementById('overview-button').click();
    },{id:row.id,action});
    // Observe after pending asynchronous play/metadata actions have had time
    // to resume; immediate paused state alone cannot prove race cancellation.
    await page.waitForTimeout(300);
    const state=await page.evaluate(ids=>({current_index:window.presentation.currentIndex,
      all_videos_paused:Array.from(document.querySelectorAll('video')).every(video=>video.paused),
      videos:ids.map(id=>{const video=document.querySelector(`video[data-media="${id}"]`);return {id,paused:video.paused,current_time:video.currentTime,seeking:video.seeking};})}),ids);
    check(state.all_videos_paused&&state.videos.every(video=>video.paused),'Late playback after rapid '+action+' cancellation in '+row.id);
    check(state.current_index===(action==='leave'?0:row.slide_index),'Rapid cancellation changed the wrong slide: '+row.id+'/'+action);
    if(action==='notes'||action==='overview'){const overlay=page.locator(action==='notes'?'#notes':'#overview');check(await overlay.isVisible(),'Expected cancellation overlay is absent');await page.keyboard.press('Escape');await overlay.waitFor({state:'hidden'});}
    if(action==='seek')await page.waitForFunction(ids=>ids.every(id=>{const v=document.querySelector(`video[data-media="${id}"]`);return !v.seeking&&Math.abs(v.currentTime-6.4)<0.005;}),ids,{timeout:20000});
    cancellations.push({action,passed:true,...state});
  }
  return {...row,ids,source_frame_seeks:frames,range_keyboard_seeks:true,actual_quartet_playback:advancing,
    playback_drift_seconds:drift,max_playback_drift_seconds:0.11,quartet_pause:paused,leaving_slide_pauses_all_four:true,
    rapid_cancellations:cancellations,all_five_rapid_cancellation_actions_verified:true};
}

async function optionalPrint(page,directory) {
  const evidence={optional:true,status:'not_generated'};
  try {
    await page.emulateMedia({media:'print'});
    const visible=await page.locator('section.slide').evaluateAll(nodes=>nodes.filter(el=>{
      const s=getComputedStyle(el);return s.display!=='none'&&s.visibility!=='hidden'&&Number(s.opacity)!==0;
    }).length);
    if(visible!==SLIDE_COUNT){evidence.reason='Print CSS does not expose all24 slides';return evidence;}
    const hiddenControls=await page.locator('[data-video-fullscreen],.pair-controls').evaluateAll(nodes=>nodes.every(el=>getComputedStyle(el).display==='none'));
    check(hiddenControls,'Print must hide external movie fullscreen buttons and quartet controls');evidence.external_video_controls_hidden=true;
    const target=path.join(directory,'presentation_print.pdf');
    const bytes=await page.pdf({path:target,printBackground:true,preferCSSPageSize:true});
    const pages=(bytes.toString('latin1').match(/\/Type\s*\/Page\b/g)||[]).length;
    check(pages===SLIDE_COUNT,`Print PDF has ${pages} pages instead of24`);
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
  const result={status:'running',qa_scope:'full_revision4_64_videos',file_path:htmlFile,file_url:expectedURL,started_at_utc:currentUtc(),events};
  try {
    await page.goto(expectedURL,{waitUntil:'load',timeout:60000});
    await page.waitForFunction(()=>window.presentation&&typeof window.presentation.goTo==='function'
      &&typeof window.presentation.loadVideo==='function',{},{timeout:30000});
    const data=await page.locator('#presentation-data').evaluate(el=>JSON.parse(el.textContent));
    const validated=validateData(data);
    const ids=await page.locator('section.slide').evaluateAll(nodes=>nodes.map(el=>el.id));
    check(JSON.stringify(ids)===JSON.stringify(Array.from({length:SLIDE_COUNT},(_,i)=>`slide-${i+1}`)),'Twenty-four numbered slides in order are required');
    result.scope=data.scope;result.source_scope=data.source_scope;result.sources=data.sources;result.visualization_revision=4;result.images=await waitImages(page);
    result.renderer_evidence={status:data.visualization.status,cpu_only:data.visualization.cpu_only,
      new_model_inference:data.visualization.new_model_inference,GPU_used:data.visualization.GPU_used,
      run_signature:data.visualization.run_signature,analysis_sha256:data.visualization.analysis_sha256,
      original_report_preservation:data.visualization.original_report_preservation,
      source_code_sha256:data.visualization.source_code_sha256,
      source_validation_policy:'Hash-bound renderer evidence and exact case/task/source/scalar consistency; browser QA does not reread experiment arrays',
      source_renders:MEDIA_IDS.map(id=>({id,record_sha256:sha(Buffer.from(JSON.stringify(canonical(data.media[id].source_render)))),
        movie_sha256:data.media[id].source_render.sha256,source_evidence:data.media[id].source_render.source_evidence,
        selected_state:data.media[id].source_render.selected_state,case_metrics:data.media[id].source_render.case_metrics}))};
    result.case_task_dom=await checkCaseDOM(page,data);result.inline_image_proof=await checkInlineImageProof(page,data);
    const videoStates=await page.locator('video').evaluateAll(nodes=>nodes.map(el=>({id:el.dataset.media,src:el.getAttribute('src'),controls:el.controls,
      task:el.dataset.task,autoplay:el.autoplay,slide:Number(el.closest('section.slide').id.slice(6))-1})));
    check(videoStates.length===64&&new Set(videoStates.map(v=>v.id)).size===64&&videoStates.every(v=>MEDIA_IDS.includes(v.id)&&!v.controls&&!v.autoplay),
      'Exactly64 text-free, non-autoplay movies with external controls are required');
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
    for(let i=0;i<SLIDE_COUNT;i++) {
      await goTo(page,i);await waitImages(page);const state=await layout(page);
      if(screenshots){const filename=`slide${String(i+1).padStart(3,'0')}.png`;await page.screenshot({path:path.join(directory,filename),animations:'disabled'});state.screenshot=filename;}
      result.desktop_slides.push(state);
    }
    result.viewport_checks=[];
    for(const viewport of [{width:1280,height:800},{width:390,height:844},{width:844,height:390}]) {
      await page.setViewportSize(viewport);
      const checks=[];
      for(let i=0;i<SLIDE_COUNT;i++){await goTo(page,i);checks.push(await layout(page));}
      result.viewport_checks.push({viewport,slides:checks});
      if(screenshots&&viewport.width===390){for(const i of [0,SLIDE_COUNT-1]){await goTo(page,i);await page.screenshot({path:path.join(directory,`mobile-slide-${i+1}.png`),animations:'disabled'});}}
    }
    await page.setViewportSize({width:1600,height:1000});result.task_quartets=[];
    result.videos=[];
    for(const row of GROUPS)result.task_quartets.push(await pairProof(page,row));
    for(const id of MEDIA_IDS) {
      const movie=await mediaProof(page,id,data.media[id]),group=result.task_quartets.find(row=>row.ids.includes(id));
      movie.source_frame_seeks=group.source_frame_seeks.map(row=>({source_frame:row.source_frame,target_time:row.target_time,
        decoded_current_frame:row.all_four_current_frames_decoded,video:row.videos.find(video=>video.id===id)}));
      result.videos.push(movie);
    }
    await goTo(page,0);
    if(print){result.print=await optionalPrint(page,directory);check(result.print.status==='passed'&&result.print.pages===24,
      'Original presentation did not produce a verified24-page print PDF: '+JSON.stringify(result.print));}
    check(events.external_requests.length===0&&events.file_asset_requests.length===0&&events.console_errors.length===0&&events.page_errors.length===0,
      `Offline/console failure: ${JSON.stringify(events)}`);
    Object.assign(result,{status:'passed',completed_at_utc:currentUtc(),all_slides_visible_and_fitted:true,
      actual_all64_video_playback:true,video_count:64,all16_task_quartets_verified:true,
      absolute_rgb_difference_removed:true,all_last_frame_playback:true,leaving_slide_pauses_videos:true,
      playback_checks_rerun:true,no_prior_playback_evidence_inherited:true,
      playback_policy:'All64 movies freshly advanced at least0.3s and presented source frame127(12.7s)/end12.8s; all16 four-objective groups freshly sought frames0/64/127 and passed play/pause/leave/five rapid-cancellation actions'});
    return result;
  } catch(error) {result.status='failed';result.error=String(error.stack||error);throw Object.assign(error,{location_evidence:result});}
  finally {await context.close();}
}

async function prepareSourceEvidence(args,inputBytes) {
  const receipts=[];
  async function read(file) {const bytes=await fs.readFile(file);receipts.push({path:path.resolve(file),sha256:sha(bytes),bytes:bytes.length});return bytes;}
  const match=inputBytes.toString('utf8').match(/<script\b(?=[^>]*\bid="presentation-data")[^>]*>([\s\S]*?)<\/script>/);
  check(match,'Presentation data JSON script is absent');const data=JSON.parse(match[1]);validateData(data);
  const reportBytes=await read(path.join(args['report-dir'],'data','report_data.json'));
  const manifestBytes=await read(path.join(args['report-dir'],'report_manifest.json'));
  const visualizationBytes=await read(path.join(args['visualization-dir'],'task_visualization_manifest.json'));
  const couplingBytes=await read(path.join(args['report-dir'],'data','task_coupling_analysis.json'));
  const report=JSON.parse(reportBytes),manifest=JSON.parse(manifestBytes),visualization=JSON.parse(visualizationBytes);
  const coupling=JSON.parse(couplingBytes);
  check(sha(reportBytes)===data.sources.report_data_sha256&&sha(manifestBytes)===data.sources.report_manifest_sha256,
    'Current original detailed report SHA differs from presentation provenance');
  check(sha(visualizationBytes)===data.sources.visualization_manifest_sha256&&same(visualization,data.visualization),
    'Fresh physical64-movie renderer manifest differs from embedded provenance');
  check(report.status==='complete'&&report.synthetic_only===false&&manifest.status==='complete'&&report.run_id===data.run_id,
    'Original detailed report is not a completed actual run');
  check(same(data.source_scope,report.scope),'Original48-condition scope metadata was altered');
  check(same(report.clips,CLIPS.map(({dataset,sequence})=>({dataset,sequence}))),
    'Presentation clips differ from the original eight-clip interleaved order');
  const reportRows=report.impact?.filter(row=>row.dataset==='all'&&OBJECTIVES.includes(row.objective));
  check(reportRows?.length===4,'Original report lacks four focused overall impact rows');
  for(const row of data.aggregates)check(same(row,reportRows.find(item=>item.objective===row.objective)),
    'Embedded aggregate differs from the original report: '+row.objective);
  for(const row of data.cases)check(same(row,report.conditions.find(item=>identity(item)===identity(row))),
    'Embedded case differs from the original saved scalar row: '+identity(row));
  const expectedPoints=report.scatter.points.filter(row=>OBJECTIVES.includes(row.objective));
  check(same(data.scatter.points,expectedPoints)&&same(data.scatter.connections,report.scatter.connections)
    &&same(data.scatter.points,coupling.scatter.points.filter(row=>OBJECTIVES.includes(row.objective)))
    &&same(data.scatter.connections,coupling.scatter.connections)&&data.scatter.source_data_sha256===sha(couplingBytes),
    'Focused32 scatter points/eight paired connections differ from the original source coupling JSON/report');
  check(manifest.outputs?.['data/task_coupling_analysis.json']?.sha256===sha(couplingBytes),
    'Original coupling source JSON hash differs from the preserved detailed report manifest');
  const accepted=new Set([...Object.values(data.sources.selected_assets||{}),...Object.values(data.sources.generated_assets||{})]);
  check(hashString(data.scatter.asset_sha256)&&accepted.has(data.scatter.asset_sha256)
    &&data.scatter.point_count===32&&data.scatter.connection_count===8&&same(data.scatter.objectives,OBJECTIVES),
    'Focused scatter image/source/count/objective provenance differs');
  return {schema_version:1,status:'passed',original_conditions:48,focused_conditions:40,focused_clean_conditions:8,
    focused_attack_conditions:32,clips:8,aggregate_rows:4,exact_original_case_rows:32,scatter_points:32,
    scatter_connections:8,current_report_data_sha256:sha(reportBytes),current_report_manifest_sha256:sha(manifestBytes),
    current_visualization_manifest_sha256:sha(visualizationBytes),current_scatter_source_sha256:sha(couplingBytes),no_prior_playback_evidence_inherited:true,
    receipts,policy:'Fresh source scalar/manifest SHA bindings only; this browser verifier does not reread experiment arrays or run a model'};
}

async function verifyReceipts(receipts) {
  for(const item of receipts) {const bytes=await fs.readFile(item.path);check(bytes.length===item.bytes&&sha(bytes)===item.sha256,'Numerical report or renderer evidence changed during full QA: '+item.path);}
}

async function main() {
  const args=argumentsFrom(process.argv.slice(2));
  if(args.help){process.stdout.write('Usage: node verify_component_presentation.cjs --presentation ABSHTML --qa-dir ABSSCRATCH --playwright-module MODULE --browser-executable EDGE --report-dir ABSREPORTDIR --visualization-dir ABSVISUALDIR\nFull revision4 QA:24 slides, four objectives, eight clips,64 fresh videos; no playback evidence is inherited.\n');return 0;}
  const source=await fs.realpath(args.presentation),qaDir=path.resolve(args['qa-dir']);
  check((await fs.stat(source)).isFile()&&/\.html?$/i.test(source),'Presentation must be one existing HTML file');
  const relative=path.relative(source,qaDir);check(relative!==''&&!qaDir.startsWith(source+path.sep),'QA scratch must not overwrite the presentation');
  await fs.mkdir(qaDir,{recursive:true});const qaPath=path.join(qaDir,'presentation_qa.json');
  try {await fs.access(qaPath);throw new Error('Existing QA evidence must be preserved; choose a fresh --qa-dir');}catch(error){if(error.code!=='ENOENT')throw error;}
  const started=currentUtc(),inputBytes=await fs.readFile(source),inputSha=sha(inputBytes);
  const output={schema_version:4,status:'running',qa_scope:'full_revision4_64_videos',started_at_utc:started,presentation:source,presentation_sha256:inputSha,
    verifier_sha256:sha(await fs.readFile(__filename)),command:process.argv,cpu_only:true,new_model_inference:false,experiment_modified:false,errors:[]};
  let browser;
  try {
    output.source_evidence=await prepareSourceEvidence(args,inputBytes);
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
    await verifyReceipts(output.source_evidence.receipts);output.source_evidence.evidence_unchanged_during_qa=true;
    Object.assign(output,{status:'passed',completed_at_utc:currentUtc(),isolated_single_file_copy_verified:true,external_network_requests:0,
      external_file_asset_requests:0,console_errors:0,actual_all64_video_playback:true,video_count:64,
      all16_task_quartets_verified:true,absolute_rgb_difference_removed:true,all_last_frame_playback:true,
      playback_checks_rerun:true,no_prior_playback_evidence_inherited:true,
      fresh_checks:['original48_vs_focused40_source_binding','four_objective_no_mse_public_scope','32cases_64task_scalars_32scatter_points',
        'actual_post_attack_apd_percent_and_common_0_100_bar_scale','all24_slide_layouts_at_four_viewports',
        'inline_images_and64_encoded_media_hashes','navigation_and_overlays','all64_actual_playback_and_presented_last_frames',
        'all16_quartet_same_frame_seek_play_pause_leave_and80_rapid_cancellations',
        'original_and_renamed_html_only_copy_offline','24_page_print']});
  } catch(error) {output.status='failed';output.errors.push(String(error.stack||error));if(error.location_evidence)output.failed_location=error.location_evidence;}
  finally {if(browser)await browser.close();}
  output.elapsed_seconds=(Date.parse(currentUtc())-Date.parse(started))/1000;
  await fs.writeFile(qaPath,JSON.stringify(output,null,2)+'\n',{encoding:'utf8',flag:'wx'});
  process.stdout.write(JSON.stringify({status:output.status,qa:qaPath,presentation_sha256:inputSha,errors:output.errors})+'\n');
  return output.status==='passed'?0:1;
}

if(require.main===module) main().then(code=>{process.exitCode=code;}).catch(error=>{process.stderr.write(String(error.stack||error)+'\n');process.exitCode=1;});
module.exports={argumentsFrom,validateData,validateLocation:verifyLocation,validateBarGeometry,prepareSourceEvidence,verifyMovie:mediaProof};
