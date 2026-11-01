/* Self-contained teaching UI. No requests, telemetry, or GPU timing estimates. */
(() => {
  'use strict';
  const M = window.Qwen9BModels;
  const names = {
    chain:['一条请求怎样经过模型','固定 3072→1024 的文本请求。按步骤查看本轮工作与缓存，所有阶段说明均为机制演示。'],
    gdn:['从局部卷积到状态纠错','上半部显示单通道四位置卷积；下半部是独立的二维 GDN 教学例子。两者不冒充真实权重或同一层的实测激活。'],
    scheduler:['一轮究竟能处理多少 token','假设启用分块 prefill，已有 decode 数固定且每请求每轮一个位置；忽略对齐、抢占和其他调度约束。'],
    graph:['为真实 batch 选择 Graph 桶','只演示支持向上 padding 的普通单 token decode。MTP 验证还受 query length、请求数和后端能力约束。'],
    memory:['把缓存容量与副本数分开算','总请求数固定。估算 8 层 BF16 KV、24 层 FP32 GDN 状态和约 9B 权重；未计卷积、Graph、MTP、工作区和分页对齐。'],
  };
  const initial = new URLSearchParams(location.search).get('panel');
  const initialPanel = names[initial] ? initial : 'chain';
  const widget = `qwen9b-${initialPanel}`;
  const picker = document.getElementById('panel'), controls = document.getElementById('controls'), result = document.getElementById('result');
  const state = {panel:initialPanel,step:0,alpha:0.9,beta:0.5,conv:4,prompt:3072,budget:2048,decodes:0,active:5,buckets:'tiered',requests:16,context:4096,replicas:2,allowed:36,nonCache:24};
  picker.value = state.panel;
  const fmt = (n,d=4) => Number(n).toFixed(d);
  const number = (id,label,value,min,max,step=1) => `<label class="control" for="${id}">${label}<input id="${id}" type="number" value="${value}" min="${min}" max="${max}" step="${step}"></label>`;
  const slider = (id,label,value,min,max,step) => `<label class="control" for="${id}">${label} <output id="${id}-value">${value}</output><input id="${id}" type="range" value="${value}" min="${min}" max="${max}" step="${step}"></label>`;
  const matrix = (title,values,positive=false) => `<div class="card"><h3>${title}</h3><div class="matrix ${positive?'positive':''}">${values.flat().map(x=>`<span>${fmt(x)}</span>`).join('')}</div></div>`;
  const vector = v => `[${v.map(x=>fmt(x)).join(', ')}]`;
  function mount() {
    const p=state.panel;document.getElementById('panel-title').textContent=names[p][0];document.getElementById('panel-description').textContent=names[p][1];
    if(p==='chain') controls.innerHTML='<div class="actions"><button type="button" data-action="prev">上一步</button><button type="button" data-action="next">下一步</button><button type="button" data-action="reset">回到请求入口</button></div>';
    if(p==='gdn') controls.innerHTML=slider('conv','当前通道投影值',state.conv,-4,8,0.5)+slider('alpha','旧状态保留 α',state.alpha,0,1,0.05)+slider('beta','纠错写入 β',state.beta,0,1,0.05);
    if(p==='scheduler') controls.innerHTML=number('prompt','新请求 prompt 长度',state.prompt,256,8192,256)+'<label class="control" for="budget">Token 预算<select id="budget"><option value="512">512</option><option value="2048">2048</option><option value="4096">4096</option></select></label>'+number('decodes','已有普通 decode 请求',state.decodes,0,32);
    if(p==='graph') controls.innerHTML=number('active','本轮实际 token 位置',state.active,1,80)+'<label class="control" for="buckets">捕获集合<select id="buckets"><option value="single">仅 [32]</option><option value="tiered">[4,8,16,32]</option><option value="wide">[4,8,16,32,64]</option></select></label>';
    if(p==='memory') controls.innerHTML=number('requests','服务总请求数',state.requests,1,64)+number('context','每请求缓存位置数',state.context,256,32768,256)+'<label class="control" for="replicas">独立权重副本数<select id="replicas"><option value="1">1 份</option><option value="2">2 份</option></select></label>'+number('allowed','假设引擎显存预算 / GiB',state.allowed,16,64)+number('nonCache','假设非缓存占用 / GiB',state.nonCache,8,48);
    for(const id of ['budget','buckets','replicas']) {const e=document.getElementById(id);if(e)e.value=state[id];}
    render();
  }
  const stages=[
    ['队列与准入','3072 个输入 token 尚未送入主干。先检查序列槽位、缓存池与设备状态。','读取：服务队列、空闲块和槽位；更新：请求与槽位映射。'],
    ['分块 prefill','调度 token 预算决定本轮处理多少 prompt 位置。2048 预算下可拆块，4096 有机会一次容纳3072。','读取：本轮输入、已有部分状态；更新：prefill进度。'],
    ['24 层 GDN','每层投影 QKVZ/AB，短卷积混合近邻，按门控纠错状态，再由Q读取。','读取/更新：该请求、该层的conv state与32个矩阵状态。'],
    ['8 层完整注意力','执行带门控的GQA，16个Q头对应4个KV头；每层后仍有稠密FFN。','读取/更新：各完整注意力层自己的历史KV。'],
    ['首个输出 token','主干最终归一化后由LM Head计算词表logits。通常从prefill末位置采样首个输出。','TTFT还包含排队、调度和返回开销，不只是某个kernel时间。'],
    ['Decode / MTP','普通decode每请求通常处理一个新位置；MTP提出多个候选，再验证和提交接受的前缀。','持续更新KV、conv与递推状态；被拒绝候选的状态不能当正式历史。'],
    ['结束与释放','达到停止条件后，释放请求占用的缓存块与状态槽位，供新请求复用。','清除请求所有权并正确初始化复用槽位；无需释放整个模型权重。'],
  ];
  function render() {
    const p=state.panel;
    if(p==='chain'){
      const s=stages[state.step];
      result.innerHTML=`<div class="steps">${stages.map((x,i)=>`<div class="step ${i===state.step?'active':''}">${i+1}. ${x[0]}</div>`).join('')}</div><div class="card"><h2>${state.step+1} / ${stages.length} · ${s[0]}</h2><p>${s[1]}</p><p class="notice">${s[2]}</p></div><div class="card"><h3>主干排列：3个GDN＋1个完整注意力，重复8组</h3><div class="layers">${Array.from({length:32},(_,i)=>`<span class="layer ${i%4===3?'attn':''}">${i+1}<br>${i%4===3?'ATTN':'GDN'}</span>`).join('')}</div><p class="small muted">图按层号展示结构，不表示各层同时执行；每一层都有自己的FFN。</p></div>`;
      controls.querySelector('[data-action="prev"]').disabled=state.step===0;controls.querySelector('[data-action="next"]').disabled=state.step===stages.length-1;
    }
    if(p==='gdn'){
      const c=M.causalConv([1,2,3,state.conv],[0.1,0.2,0.3,0.4]);
      const old=[[0.6,0.1],[0.2,0.4]],q=[1/Math.sqrt(2),0],k=[0.6,0.8],v=[1,-0.5];
      const d=M.deltaStep(old,k,v,q,state.alpha,state.beta);
      result.innerHTML=`<div class="card"><h3>单通道因果卷积：输入按过去→当前排列</h3><div class="window">${[1,2,3,state.conv].map((x,i)=>`<span>t${i===3?'':i-3}<br>${fmt(x,1)} × ${[0.1,0.2,0.3,0.4][i]}</span>`).join('')}</div><p class="formula">加权和 = ${fmt(c.weighted)}；SiLU 后 = ${fmt(c.silu)}</p><p class="small muted">固定教学卷积核，只读取过去与当前；真实模型对各QKV通道使用各自训练权重。</p></div><p class="vector">独立二维例：k=[0.6,0.8]，v=[1,−0.5]，q=[1,0]/√2</p><div class="grid">${matrix('① 旧状态 S',old)}${matrix('② 衰减 αS',d.decayed)}${matrix('③ 外积修正 βk(v−kᵀαS)ᵀ',d.correction)}${matrix('④ 新状态',d.state,true)}</div><div class="card"><p class="formula">预测 v̂ = ${vector(d.prediction)}<br>误差 v−v̂ = ${vector(d.error)}<br>读出 qᵀS = ${vector(d.output)}</p><p class="small muted">α 控制保留，β 控制纠错。这里只展示状态核心，未执行输出Z门、头拼接或真实模型。</p></div>`;
    }
    if(p==='scheduler'){
      const d=M.scheduleRounds(state.prompt,state.budget,state.decodes);
      result.innerHTML=`<div class="grid"><div class="card"><h3>每轮剩余 prefill 预算</h3><p class="stat">${d.available}</p><p class="formula">${state.budget} − ${state.decodes} = ${d.available} token</p></div><div class="card"><h3>本例需要的 prefill 轮数</h3><p class="stat">${d.chunks.length}</p><p class="small muted">每轮已有decode数不变；轮数不等于毫秒延迟。</p></div></div><div class="card">${d.chunks.map((n,i)=>`<div class="round"><h3>第${i+1}轮 · prefill ${n} ＋ decode ${state.decodes}</h3><div class="bar" role="img" aria-label="本轮处理${n}个prefill和${state.decodes}个decode token，预算${state.budget}"><span class="useful" style="width:${100*n/state.budget}%"></span><span class="weights" style="width:${100*state.decodes/state.budget}%"></span></div></div>`).join('')}<p class="small muted">青绿：prefill；金色：decode；未填满部分为空闲预算。未检查额外序列槽位或缓存准入。</p></div><p class="notice">更大预算可以减少分块，却也可能增加单轮执行时间。此图不据轮数推算TTFT、吞吐或真实调度次序。</p>`;
    }
    if(p==='graph'){
      const sets={single:[32],tiered:[4,8,16,32],wide:[4,8,16,32,64]},d=M.graphChoice(state.active,sets[state.buckets]);
      result.innerHTML=`<div class="grid"><div class="card"><h3>选择的执行路径</h3><p class="stat">${d.bucket===null?'Eager':`Graph ${d.bucket}`}</p><p class="small muted">捕获集合：${sets[state.buckets].join(' / ')}</p></div><div class="card"><h3>额外 padding 位置</h3><p class="stat">${d.padding}</p><p class="small muted">${d.bucket===null?'没有兼容桶，本例回退eager，不人为补空位。':`有效位置占本桶 ${fmt(d.fill*100,1)}%`}</p></div></div><div class="card"><h3>实际位置与补齐槽位</h3><div class="tokens" role="img" aria-label="${d.useful}个真实位置，${d.padding}个补齐位置">${Array.from({length:d.bucket??d.useful},(_,i)=>`<span class="token ${i>=d.useful?'pad':''}"></span>`).join('')}</div><div class="legend"><span><i class="useful"></i>实际位置 ${d.useful}</span><span><i class="padding"></i>padding ${d.padding}</span></div></div><p class="notice">桶规模不是历史上下文长度。增加64桶只扩展覆盖，不改变已命中8桶的执行；更多桶还需要捕获和管理。真实MTP分派需要另看日志。</p>`;
    }
    if(p==='memory'){
      const d=M.memoryEstimate(state.requests,state.context,state.replicas),total=d.weightGiB+d.kvGiB+d.stateGiB,c=M.cacheBudget(state.allowed,state.nonCache,8);
      result.innerHTML=`<div class="grid"><div class="card"><h3>BF16 KV · 随上下文增长</h3><p class="stat">${fmt(d.kvGiB,3)} GiB</p><p class="formula">总请求数 × 缓存位置 × 32 KiB</p></div><div class="card"><h3>FP32 GDN 状态 · 固定每请求</h3><p class="stat">${fmt(d.stateGiB,3)} GiB</p><p class="formula">总请求数 × 48 MiB</p></div></div><div class="card"><h3>已建模部分合计 ${fmt(total,2)} GiB</h3><div class="bar" role="img" aria-label="权重${fmt(d.weightGiB,2)}GiB，KV${fmt(d.kvGiB,3)}GiB，GDN状态${fmt(d.stateGiB,3)}GiB"><span class="weights" style="width:${100*d.weightGiB/total}%"></span><span class="kv" style="width:${100*d.kvGiB/total}%"></span><span class="ssm" style="width:${100*d.stateGiB/total}%"></span></div><p class="small">金：${state.replicas}份权重约 ${fmt(d.weightGiB,2)} GiB；绿：KV；蓝：GDN状态。</p><p class="small muted">总请求数不因副本数翻倍，因此KV与状态的全服务估算不变。这里只改变独立权重份数，不模拟TP分片、共享分配或DP负载倾斜。</p></div><div class="card"><h3>另一组假设：每worker固定与自动KV预算</h3><p class="formula">允许 ${state.allowed} − 非缓存 ${state.nonCache} = ${fmt(c.automatic,1)} GiB 可规划空间</p><div class="grid"><p>固定：<strong>8 GiB</strong></p><p>自动：约 <strong>${fmt(c.automatic,1)} GiB</strong></p></div><p class="small muted">${c.headroom<0?'固定8GiB已超过本例剩余空间，存在容量冲突。':'固定模式下仍有部分余量。'}自动是启动容量推导，不是运行中无限扩容。</p></div><p class="notice">两组估算分别用于解释总服务状态和单worker预算，不相互套用。未计额外内存，不能据此宣称真实GR308一定装得下。</p>`;
    }
    for(const id of ['conv','alpha','beta']){const e=document.getElementById(`${id}-value`);if(e)e.textContent=state[id];}
    reportHeight();
  }
  picker.addEventListener('change',()=>{state.panel=picker.value;mount();});
  controls.addEventListener('input',event=>{
    const e=event.target;if(!e.id)return;
    if(e.type==='number'&&(e.value===''||!e.checkValidity()))return;
    state[e.id]=e.tagName==='SELECT'&&e.id==='buckets'?e.value:Number(e.value);
    render();
  });
  controls.addEventListener('click',event=>{const action=event.target.closest('[data-action]')?.dataset.action;if(!action)return;state.step=action==='reset'?0:Math.max(0,Math.min(stages.length-1,state.step+(action==='next'?1:-1)));render();});
  let lastHeight=0;
  function reportHeight(){
    requestAnimationFrame(()=>{const height=Math.ceil(document.getElementById('lab').getBoundingClientRect().height+parseFloat(getComputedStyle(document.body).paddingTop)*2);if(Math.abs(height-lastHeight)<2)return;lastHeight=height;if(window.parent!==window)window.parent.postMessage({type:'transformer-viz-height',widget,height},location.origin);});
  }
  new ResizeObserver(reportHeight).observe(document.getElementById('lab'));
  window.addEventListener('load',reportHeight);
  document.fonts.ready.then(reportHeight);
  mount();
})();
