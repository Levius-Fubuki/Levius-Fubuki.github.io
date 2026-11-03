/* Local-only teaching controls. No model execution, network calls, or TPS forecasts. */
(()=>{
  'use strict';
  const M=window.QwenHybridModels;
  const panels={chain:['请求与混合层','从请求准入到释放，观察全注意力、GDN和MoE如何共同工作。'],budget:['Token 上限与显存容量','普通无MTP调度的算术示意；权重均分估算与实际运行峰值分开。'],state:['接受前缀与状态位置','只展示线性草稿前缀。正式历史与KV、GDN、卷积状态必须对应同一位置。'],evidence:['历史调优证据','固定C16历史点。选择一个候选，查看假设、动作后的结果和证据边界。']};
  const initial=new URLSearchParams(location.search).get('panel');
  const originPanel=panels[initial]?initial:'chain';
  const state={panel:originPanel,model:'35b',step:0,limit:2048,decode:8,prefill:3072,tp:2,drafts:4,accepted:2,experiment:'selected'};
  const picker=document.getElementById('panel'),modelPicker=document.getElementById('model'),controls=document.getElementById('controls'),result=document.getElementById('result');
  const fmt=(x,n=2)=>Number(x).toFixed(n);
  const input=(id,label,min,max,step=1)=>`<label class="control" for="${id}">${label}<input id="${id}" type="number" min="${min}" max="${max}" step="${step}" value="${state[id]}"></label>`;
  const select=(id,label,values)=>`<label class="control" for="${id}">${label}<select id="${id}">${values.map(([v,t])=>`<option value="${v}"${String(state[id])===String(v)?' selected':''}>${t}</option>`).join('')}</select></label>`;
  const card=(label,value,text)=>`<div class="card"><h3>${label}</h3><p class="stat">${value}</p><p class="small muted">${text}</p></div>`;
  const stages=['队列与准入','Prefill预算','混合层主干','MoE与通信','首token与Decode','MTP验证','请求结束'];
  function mount(){
    picker.value=state.panel;modelPicker.value=state.model;
    document.getElementById('panel-title').textContent=panels[state.panel][0];document.getElementById('panel-description').textContent=panels[state.panel][1];
    if(state.panel==='chain')controls.innerHTML='<div class="actions"><button type="button" data-action="prev">上一步</button><button type="button" data-action="next">下一步</button><button type="button" data-action="reset">回到入口</button></div>';
    if(state.panel==='budget')controls.innerHTML=select('limit','本轮token上限',[[1024,'1024'],[2048,'2048'],[4096,'4096'],[6144,'6144']])+input('decode','本轮普通Decode token',0,64)+input('prefill','待处理Prefill token',0,8192)+select('tp','权重均分假设 TP',[[1,'TP1'],[2,'TP2'],[4,'TP4']]);
    if(state.panel==='state')controls.innerHTML=input('drafts','草稿候选数',1,6)+input('accepted','接受前缀长度',0,state.drafts);
    if(state.panel==='evidence')controls.innerHTML=select('experiment','选择历史候选',M.experiments[state.model].map(x=>[x.id,x.label]));
    render();
  }
  function render(){
    const m=M.models[state.model];
    if(state.panel==='chain'){
      const descriptions=[
        ['先检查容量','CPU调度器检查序列槽位、全注意力KV、GDN和卷积状态池。','模型权重常驻；请求拥有自己的历史状态。'],
        [`${m.input} token输入`,'一次调度只处理预算和策略允许的部分；启用分块Prefill时可以分轮推进。','输入长度、并发和本轮token数是不同变量。'],
        [`${m.layers}层：${m.gdn} GDN + ${m.full}全注意力`,'每四层为三个GDN和一个全注意力；每层都有自己的MoE。','点击下方层块可定位到混合层或MoE说明。'],
        [`Top-${m.top} / ${m.experts}路由专家`,'每个token在每层重新路由，另有共享专家。TP切分张量，EP改变专家所有权。',`当前实测拓扑TP${m.tp}；collective次数和融合要以厂商实际路径为准。`],
        ['输出与继续生成','Prefill末位置的logits用于产生首个输出。普通Decode每个活跃请求本轮通常输入一个新token。','全注意力KV增长，GDN矩阵与短卷积状态持续更新。'],
        ['提出、验证、提交','MTP辅助路径提出候选，目标路径验证。只有可接受前缀成为正式历史。','必须让KV有效长度、GDN与卷积状态对应同一已提交位置。'],
        ['结束与释放','达到停止条件后回收请求槽位和历史缓存，模型权重仍可常驻。','输出成功、资源释放、冷启动稳定性是不同的验证项。']
      ];
      const d=descriptions[state.step];
      result.innerHTML=`<div class="steps">${stages.map((x,i)=>`<div class="step${i===state.step?' active':''}">${i+1}. ${x}</div>`).join('')}</div><div class="card"><h2>${state.step+1} / 7 · ${d[0]}</h2><p>${d[1]}</p><p class="notice">${d[2]}</p></div><div class="card"><h3>层排列：G = GDN，A = 全注意力；每格另有 MoE</h3><div class="layers" aria-label="${m.layers}个模型层">${Array.from({length:m.layers},(_,i)=>`<button type="button" class="layer ${(i+1)%4?'gdn':'full'}" data-layer="${i+1}" aria-label="第${i+1}层，${(i+1)%4?'GDN':'全注意力'}与MoE">${String(i+1).padStart(2,'0')} ${(i+1)%4?'G':'A'}<span>MoE</span></button>`).join('')}</div><p class="legend">青绿 GDN：固定形状递归状态；蓝色全注意力：随历史长度增长的KV。这里只表示层次，不表示耗时比例。</p></div>`;
      controls.querySelector('[data-action="prev"]').disabled=state.step===0;controls.querySelector('[data-action="next"]').disabled=state.step===6;
    }
    if(state.panel==='budget'){
      const b=M.budget(state.limit,state.decode,state.prefill),mem=M.memory(state.model,state.tp),over=mem.weightGiB>56;
      result.innerHTML=`<div class="grid">${card('本轮处理token',`${b.total} / ${state.limit}`,`Decode ${b.decode}，Prefill ${b.prefill}；未用预算${b.unused}。`)}${card('Prefill尚余',b.remaining,'本例优先给普通Decode预留预算；真实调度还受序列数、对齐和优先级限制。')}</div><div class="card"><h3>本轮 token 分配</h3><div class="bar" role="img" aria-label="Decode ${b.decode}，Prefill ${b.prefill}，未用${b.unused}"><span class="decode" style="width:${b.decode/state.limit*100}%"></span><span class="prefill" style="width:${b.prefill/state.limit*100}%"></span><span class="space" style="width:${b.unused/state.limit*100}%"></span></div><p class="legend">蓝：Decode；青绿：Prefill；深色：未用预算。它不是整个请求输入加输出的上限。</p></div><div class="card"><h3>另一项独立估算：每 rank 主要权重</h3><p class="stat">${fmt(mem.weightGiB)} GiB</p><p class="small">${m.params}B × 2 bytes ÷ TP${state.tp}；忽略复制参数。单逻辑设备参考容量56 GiB。</p>${Array.from({length:state.tp},(_,i)=>`<div class="rank"><div class="track-label"><span>rank ${i}</span><span>${fmt(mem.weightGiB)} / 56 GiB</span></div><div class="bar" role="img" aria-label="rank${i}权重粗估${fmt(mem.weightGiB)}GiB，容量56GiB"><span class="weights" style="width:${Math.min(100,mem.weightGiB/56*100)}%"></span></div></div>`).join('')}<p class="notice${over?' warning':''}">${over?'仅权重粗估已超过参考容量。':'权重粗估低于参考容量，但不能据此承诺服务可启动。'} KV、GDN、卷积状态、工作区、Graph和通信缓冲均另计。</p><p class="small muted">4096位置时，全模型未分片的原始KV约${mem.kvMiB} MiB／请求；按FP32粗估GDN矩阵约${mem.gdnMiB} MiB／请求。两项不是本rank实际分配，TP4的KV尤其不能直接除以四。</p></div>`;
    }
    if(state.panel==='state'){
      const position=128+state.accepted;
      result.innerHTML=`<span class="badge">线性候选前缀示意 · 非真实状态数值</span><div class="card"><h3>已有128个正式位置，接着提出候选</h3><div class="candidates">${Array.from({length:state.drafts},(_,i)=>`<div class="candidate ${i<state.accepted?'committed':'discarded'}"><b>候选 ${i+1}</b><span>位置 ${129+i}</span><p class="small">${i<state.accepted?'接受并提交':'不提交'}</p></div>`).join('')}</div><p class="legend">实线青绿：接受前缀；虚线暖色：未提交候选。这里只计算草稿前缀，不计框架可能附加的目标token。</p></div><div class="card"><h3>下一轮应使用的正式历史</h3><div class="state-row"><span>已提交 token 位置</span><strong>${position}</strong></div><div class="state-row"><span>全注意力 KV 有效位置</span><strong>${position}</strong></div><div class="state-row"><span>GDN 对应历史位置</span><strong>${position}</strong></div><div class="state-row"><span>短卷积对应历史位置</span><strong>${position}</strong></div><p class="notice">正式状态必须对应同一前缀。具体如何恢复、选择或重算候选状态，由后端实现决定；本图不宣称厂商采用某一种回滚算法。</p></div>`;
    }
    if(state.panel==='evidence'){
      const points=M.experiments[state.model],e=points.find(x=>x.id===state.experiment),max=Math.max(...points.map(x=>x.tps??0));
      result.innerHTML=`<div class="card"><span class="badge">历史C16聚合输出吞吐 · tok/s</span>${points.map(x=>`<div class="evidence-row${x.id===e.id?' selected':''}"><div class="track-label"><span>${x.label}</span><span>${x.tps===null?'无有效结果':fmt(x.tps)}</span></div>${x.tps===null?'':`<div class="bar" role="img" aria-label="${x.label}，${x.tps} output tok/s"><span class="history-bar" style="width:${x.tps/max*100}%"></span></div>`}</div>`).join('')}<p class="legend">横条表示各历史配置的记录值；不是连续加速过程，不累乘各步增益。</p></div><div class="card"><h2>${e.label}</h2><p><strong>条件：</strong>${e.condition}</p><p><strong>假设：</strong>${e.hypothesis}</p><p><strong>决策：</strong>${e.decision}</p><p class="notice">${e.boundary}</p></div>`;
    }
    reportHeight();
  }
  picker.addEventListener('change',()=>{state.panel=picker.value;mount();});
  modelPicker.addEventListener('change',()=>{state.model=modelPicker.value;state.tp=M.models[state.model].tp;state.experiment='selected';mount();});
  controls.addEventListener('input',ev=>{const el=ev.target;if(!el.id||el.value===''||!el.checkValidity())return;state[el.id]=el.tagName==='SELECT'&&el.id==='experiment'?el.value:Number(el.value);if(el.id==='drafts'){state.accepted=Math.min(state.accepted,state.drafts);const a=document.getElementById('accepted');a.max=state.drafts;a.value=state.accepted;}render();});
  controls.addEventListener('click',ev=>{const action=ev.target.closest('[data-action]')?.dataset.action;if(!action)return;state.step=action==='reset'?0:Math.max(0,Math.min(6,state.step+(action==='next'?1:-1)));render();});
  result.addEventListener('click',ev=>{const layer=ev.target.closest('[data-layer]');if(layer){state.step=state.step===2?3:2;render();}});
  let lastHeight=0;
  function reportHeight(){requestAnimationFrame(()=>{const padding=parseFloat(getComputedStyle(document.body).paddingTop)*2;const height=Math.ceil(document.getElementById('lab').getBoundingClientRect().height+padding);if(Math.abs(lastHeight-height)<2)return;lastHeight=height;if(parent!==window)parent.postMessage({type:'transformer-viz-height',widget:`qwen35b80b-${originPanel}`,height},location.origin);});}
  new ResizeObserver(reportHeight).observe(document.getElementById('lab'));addEventListener('load',reportHeight);document.fonts.ready.then(reportHeight);mount();
})();
