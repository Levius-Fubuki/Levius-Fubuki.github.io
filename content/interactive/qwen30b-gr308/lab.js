/* Local teaching controls; historical evidence never changes with timing sliders. */
(() => {
  'use strict';
  const M=window.Qwen30BModels;
  const panels={chain:['一条请求与双芯容量','3072→1024 请求贯穿全注意力、专家、缓存和输出。容量仅计算BF16权重与原始KV，未计工作区、Graph和对齐。'],moe:['Top-8 路由与专家分布','固定种子的合成路由，非checkpoint激活。TP切专家内部通道，EP分专家所有权；切换放置不改变模型Top-8规则。'],scheduling:['Graph 桶与异步调度','普通无MTP Decode的向上匹配示意。下方时间为可调教学输入，不是本机实测，也不预测正式吞吐。'],bottleneck:['瓶颈假设、实验与下一步','历史记录固定展示；下方计算/访存/通信模型独立标为假设。实际瓶颈需要同一服务的profiler证据。']};
  const query=new URLSearchParams(location.search).get('panel'),initial=panels[query]?query:'chain',widget=`qwen30b-${initial}`;
  const state={panel:initial,step:0,requests:16,context:4096,tokens:16,skew:'uniform',placement:'tp',active:13,buckets:'tiered',cpu:2,gpu:10,experiment:'graph',flops:60,bytes:32,compute:20,bandwidth:600,comm:3,gap:2};
  const controls=document.getElementById('controls'),result=document.getElementById('result'),picker=document.getElementById('panel');
  const n=(id,label,min,max,step=1)=>`<label class="control" for="${id}">${label}<input id="${id}" type="number" min="${min}" max="${max}" step="${step}" value="${state[id]}"></label>`;
  const select=(id,label,items)=>`<label class="control" for="${id}">${label}<select id="${id}">${items.map(([v,t])=>`<option value="${v}"${state[id]===v?' selected':''}>${t}</option>`).join('')}</select></label>`;
  const fmt=(x,d=2)=>Number(x).toFixed(d);
  const card=(title,value,text)=>`<div class="card"><h3>${title}</h3><p class="stat">${value}</p><p class="small muted">${text}</p></div>`;
  const stages=[['队列与容量准入','请求尚未进入主干。检查序列槽位、缓存容量和请求所有权。','CPU调度器决定本轮执行哪些请求。'],['Prefill预算','输入3072个token，按本轮token预算分块或整体执行。','本步T可以远大于Decode的活跃请求数。'],['GQA全注意力','48层均使用全注意力：32个Q头、4个KV头，Q/K Norm与RoPE后读取因果历史。','TP2每rank通常负责16个Q头、2个KV头；没有GDN状态。'],['Top-8 MoE','每层Router从128个专家选8个。分组后执行各自SwiGLU，再加权合并。','每层独立选择；不同token可以触及不同专家。'],['TP归约','attention和专家的分片输出需要按实现合并。','常见TP结构需要归约；次数和融合方式以厂商trace为准。'],['首token与Decode','Prefill末位置产生首个输出；之后每个活跃请求通常每步处理一个新token。','Graph可回放匹配路径；KV继续增长，权重不改变。'],['结束与释放','达到停止条件后释放请求占用的KV块和槽位。','模型常驻，不因一个请求结束释放整套权重。']];
  function mount(){
    picker.value=state.panel;document.getElementById('panel-title').textContent=panels[state.panel][0];document.getElementById('panel-description').textContent=panels[state.panel][1];
    if(state.panel==='chain')controls.innerHTML='<div class="actions"><button data-action="prev">上一步</button><button data-action="next">下一步</button><button data-action="reset">回到入口</button></div>'+n('requests','容量估算：请求数',1,64)+n('context','每请求KV位置数',256,32768,256);
    if(state.panel==='moe')controls.innerHTML=n('tokens','本步token数 T',1,64)+select('skew','合成路由分布',[['uniform','均匀随机分数'],['hot','前16专家为热点']])+select('placement','同两rank的放置方式',[['tp','TP2：各专家切通道'],['ep','EP2：每rank 64个完整专家']]);
    if(state.panel==='scheduling')controls.innerHTML=n('active','Graph本步真实槽位',1,80)+select('buckets','Graph捕获集合',[['single','只有 [32]'],['tiered','[4,8,16,32]'],['wide','[4,8,16,32,64]']])+n('cpu','假设CPU准备 / ms',0,20,0.5)+n('gpu','假设GPU执行 / ms',1,50,0.5);
    if(state.panel==='bottleneck')controls.innerHTML=select('experiment','查看历史实验',M.experiments.map(x=>[x.id,x.title]))+n('flops','假设算子运算量 / GFLOP',1,500)+n('bytes','假设实际访存 / GB',0.1,100,0.1)+n('compute','假设有效算力 / TFLOPS',1,100)+n('bandwidth','假设有效带宽 / GB/s',100,1008)+n('comm','未隐藏通信 / ms',0,50,0.5)+n('gap','执行间隙 / ms',0,50,0.5);
    render();
  }
  function render(){
    if(state.panel==='chain'){
      const s=stages[state.step],m=M.memory(state.requests,state.context);
      result.innerHTML=`<div class="steps">${stages.map((x,i)=>`<div class="step ${i===state.step?'active':''}">${i+1}. ${x[0]}</div>`).join('')}</div><div class="card"><h2>${state.step+1} / ${stages.length} · ${s[0]}</h2><p>${s[1]}</p><p class="notice">${s[2]}</p></div><div class="grid">${card('每rank主要权重理想份额',`${fmt(m.weightGiB)} GiB`,'30.5B × 2 bytes ÷ TP2；复制参数与运行时另计。')}${card('每rank原始KV',`${fmt(m.kvGiB,3)} GiB`,`${state.requests}请求 × ${state.context}位置 × 48 KiB；不是实际分配峰值。`)}</div><p class="notice">本机每逻辑设备报告56 GiB。上述两项不是完整显存占用，不能据此承诺真实服务一定装得下。</p>`;
      controls.querySelector('[data-action="prev"]').disabled=state.step===0;controls.querySelector('[data-action="next"]').disabled=state.step===stages.length-1;
    }
    if(state.panel==='moe'){
      const d=M.route(state.tokens,state.skew==='hot');
      result.innerHTML=`<span class="badge">合成路由 · 非真实权重</span><div class="grid">${card('Token-expert assignment',d.total,`${state.tokens}个token，每个选8个不同专家。`)}${card('本例触及的不同专家',d.used,state.skew==='uniform'?`均匀独立路由的理论期望为${fmt(d.uniformExpected)}，单次合成样本可以不同。`:'热点模型不适用均匀期望公式。')}</div><div class="card"><h3>每格：专家编号 / 收到的assignment数</h3><div class="experts" role="img" aria-label="128个专家，本例${d.used}个被使用，合计${d.total}个assignment">${d.counts.map((c,i)=>`<span class="expert ${c?'used':''} ${state.placement==='tp'?'both':i<64?'rank0':'rank1'}" title="专家${i}：${c}个assignment">${i}<br>${c}</span>`).join('')}</div><p class="legend">${state.placement==='tp'?'每个专家两边分别持有384个中间通道；金/绿边表示两个rank共同参与。':'金边：rank0持有专家0–63；绿边：rank1持有专家64–127。'}</p></div><div class="card"><h3>放置改变什么</h3><p>${state.placement==='tp'?`两边都参与被选专家的部分矩阵计算。不是各自只执行一半专家。`:`本例rank0接收${d.rank0}、rank1接收${d.rank1}个assignment。差异显示分布偏斜，不等于真实时间比。`}</p><p class="formula">第一个token的专家：${d.assignments[0].map(x=>x.expert).join(' / ')}</p><p class="small muted">每个选中专家有独立权重；八个输出按归一化路由权重求和。热度、通信和kernel性能需真实采样，本图不预测EP速度。</p></div>`;
    }
    if(state.panel==='scheduling'){
      const sets={single:[32],tiered:[4,8,16,32],wide:[4,8,16,32,64]},g=M.chooseGraph(state.active,sets[state.buckets]),sync=state.cpu+state.gpu,overlap=Math.max(state.cpu,state.gpu);
      result.innerHTML=`<div class="grid">${card('向上匹配的执行规格',g.bucket===null?'Eager':`Graph ${g.bucket}`,'没有兼容桶时，本例回退；真实厂商规则需看日志。')}${card('填充槽位',g.padding,g.bucket===null?'本例不创建超出捕获集合的Graph。':'直接填充并屏蔽，不等待请求凑满。')}</div><div class="card"><h3>真实与填充位置</h3><div class="tokens" role="img" aria-label="${state.active}个真实位置，${g.padding}个填充位置">${Array.from({length:g.bucket??state.active},(_,i)=>`<span class="token ${i>=state.active?'pad':''}"></span>`).join('')}</div><p class="legend">青绿：真实位置；棕色：padding。槽位不是历史上下文长度。</p></div><div class="card"><h3>另一组独立教学输入：调度重叠</h3><div class="grid">${card('串行一步',`${fmt(sync,1)} ms`,'CPU准备 + GPU执行。')}${card('理想稳态一步下界',`${fmt(overlap,1)} ms`,'max(CPU准备, GPU执行)，忽略额外依赖与同步。')}</div><div class="bar" role="img" aria-label="串行CPU${state.cpu}毫秒，GPU${state.gpu}毫秒"><span class="weights" style="width:${state.cpu/sync*100}%"></span><span class="useful" style="width:${state.gpu/sync*100}%"></span></div><p class="legend">金：CPU准备；绿：GPU执行。async尽量重叠可提前完成的工作，不能提前知道下一枚token。</p><p class="notice">这是理想下界，不保证真实async达到该值。它不是API async/await，也不证明GPU多流或算子融合。Graph与async收益不能简单相乘。</p></div>`;
    }
    if(state.panel==='bottleneck'){
      const e=M.experiments.find(x=>x.id===state.experiment),t=M.timing(state.flops,state.bytes,state.compute,state.bandwidth,state.comm,state.gap);
      result.innerHTML=`<div class="card"><span class="badge">历史记录 · 不随滑块改变</span><h2>${e.title}</h2><p><strong>假设：</strong>${e.hypothesis}</p><p><strong>动作：</strong>${e.action}</p><p><strong>结果：</strong>${e.result}</p><p class="notice">${e.boundary}</p><p><strong>下一步：</strong>${e.next}</p></div><div class="card"><span class="badge">独立教学模型 · 非GR308峰值</span><h3>计算与访存的理想下界</h3><div class="grid">${card('计算下界',`${fmt(t.computeMs)} ms`,'GFLOP ÷ 假设有效TFLOPS')}${card('访存下界',`${fmt(t.memoryMs)} ms`,'GB ÷ 假设有效GB/s × 1000')}</div><p class="formula">算子下界 max(${fmt(t.computeMs)}, ${fmt(t.memoryMs)}) = ${fmt(t.kernelMs)} ms</p><div class="bar" role="img" aria-label="算子下界${fmt(t.kernelMs)}毫秒、未隐藏通信${state.comm}毫秒、间隙${state.gap}毫秒"><span class="useful" style="width:${100*t.kernelMs/t.stepMs}%"></span><span class="comm" style="width:${100*state.comm/t.stepMs}%"></span><span class="gap" style="width:${100*state.gap/t.stepMs}%"></span></div><p class="legend">绿：算子下界；蓝：未隐藏通信；紫：执行间隙。</p><p>本例一步约 <strong>${fmt(t.stepMs)} ms</strong>，仅按下界加未隐藏部分计算。</p><p class="notice">本机BF16峰值未获核验。假设有效算力不是硬件规格；实际访存不是激活参数量。没有trace和DRAM计数器，不能将这张图当硬件瓶颈鉴定。</p></div>`;
    }
    reportHeight();
  }
  picker.addEventListener('change',()=>{state.panel=picker.value;mount();});
  controls.addEventListener('input',ev=>{const e=ev.target;if(!e.id||!e.checkValidity()||e.value==='')return;state[e.id]=e.tagName==='SELECT'?e.value:Number(e.value);render();});
  controls.addEventListener('click',ev=>{const a=ev.target.closest('[data-action]')?.dataset.action;if(!a)return;state.step=a==='reset'?0:Math.max(0,Math.min(stages.length-1,state.step+(a==='next'?1:-1)));render();});
  let lastHeight=0;
  function reportHeight(){requestAnimationFrame(()=>{const height=Math.ceil(document.getElementById('lab').getBoundingClientRect().height+parseFloat(getComputedStyle(document.body).paddingTop)*2);if(Math.abs(height-lastHeight)<2)return;lastHeight=height;if(window.parent!==window)window.parent.postMessage({type:'transformer-viz-height',widget,height},location.origin);});}
  new ResizeObserver(reportHeight).observe(document.getElementById('lab'));window.addEventListener('load',reportHeight);document.fonts.ready.then(reportHeight);mount();
})();
