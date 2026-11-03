/* Public structure, labeled teaching arithmetic, and independent historical points. */
(function(root){
  'use strict';
  const models={
    '35b':{name:'Qwen3.6-35B-A3B',params:35,layers:40,gdn:30,full:10,experts:256,top:8,tp:2,input:3072},
    '80b':{name:'Qwen3-Next-80B-A3B',params:80,layers:48,gdn:36,full:12,experts:512,top:10,tp:4,input:1024}
  };
  const experiments={
    '35b':[
      {id:'mtp3',label:'MTP3 / budget2048',tps:164.38,condition:'TP2、C16；MTP3、预算2048，历史短测对照。',hypothesis:'比较增加一个候选后的有效提交量与验证成本。',decision:'继续比较MTP4。',boundary:'与MTP4对照仍需核对同卡、启动顺序和重复波动。'},
      {id:'mtp4',label:'MTP4 / budget2048',tps:175.57,condition:'TP2、C16；MTP4、预算2048。',hypothesis:'增加候选的收益超过额外执行成本。',decision:'进入预算4096筛选。',boundary:'性能方向有记录；没有逐项计时将收益分配到每一种机制。'},
      {id:'selected',label:'MTP4 / budget4096',tps:180.64,condition:'TP2、C16；Graph4–64、自动KV、同步；32请求、0失败。',hypothesis:'改变调度预算后减少部分切分或等待。',decision:'保留最佳短测候选。',boundary:'预算2048与4096筛选放在不同物理卡，不是严格同卡单变量因果结论。'},
      {id:'grouped',label:'Grouped GEMM + fusion',tps:167.69,condition:'35B后期C16候选；相应C32为188.63 tok/s。',hypothesis:'专家分组与融合能否改善窄GEMM效率。',decision:'未转化为端到端收益。',boundary:'组合候选不等于单一kernel独立消融。'},
      {id:'ep',label:'EP2 linear',tps:167.81,condition:'C16；对应C32为185.50 tok/s。',hypothesis:'专家所有权切分能否抵消派发和汇总成本。',decision:'关闭EP。',boundary:'端到端回归无法唯一归因于通信带宽。'},
      {id:'c64',label:'C64 layout failure',tps:null,condition:'SSM kernel状态索引非连续布局报错。',hypothesis:'更高并发是否扩大服务吞吐。',decision:'先修兼容性。',boundary:'没有有效性能结果；不将失败画成0 tok/s。'}
    ],
    '80b':[
      {id:'no-mtp',label:'Eager / no MTP',tps:177.71,condition:'TP4、C16；1024→1024、BF16、预算2048。',hypothesis:'建立不含投机路径的起点。',decision:'对照MTP1。',boundary:'与35B输入长度不同，不能直接按柱高比较模型优劣。'},
      {id:'mtp1',label:'Eager / MTP1',tps:231.30,condition:'TP4、C16；早期MTP1短测。',hypothesis:'一轮验证提交更多token能否覆盖draft成本。',decision:'继续比较Graph。',boundary:'接受率高本身不保证端到端加速。'},
      {id:'graph',label:'MTP1 / Graph64',tps:231.98,condition:'TP4、C16；捕获集合[4,8,16,32,64]。',hypothesis:'固定执行形状能否降低重复发射开销。',decision:'保留Graph64。',boundary:'Graph桶定义需要日志确认，不能用并发×候选数替代。'},
      {id:'mtp2',label:'Graph / MTP2',tps:251.49,condition:'后期TP4、C16短测候选。',hypothesis:'继续提高每轮有效提交量。',decision:'进一步比较MTP3。',boundary:'后期实验与最初只覆盖MTP0/1的脚本分开记录。'},
      {id:'selected',label:'Graph / MTP3',tps:255.93,condition:'TP4、C16；预算2048、8GiB KV、同步；32请求、0失败。',hypothesis:'MTP3在吞吐与C1之间取得更好组合。',decision:'保留最佳短测候选。',boundary:'相对MTP2的C16差异不足3%；原始独立C16入口尚未确认，需正式配对复测。'},
      {id:'ep',label:'TP4 + EP4',tps:233.93,condition:'后期C16专家并行候选。',hypothesis:'完整专家计算能否抵消跨rank派发成本。',decision:'不保留EP4。',boundary:'分项损失还需profile；不能断言所有EP实现均无效。'}
    ]
  };
  function budget(limit,decode,prefill){
    const d=Math.min(decode,limit),p=Math.min(prefill,Math.max(0,limit-d));
    return {decode:d,prefill:p,total:d+p,remaining:Math.max(0,prefill-p),unused:limit-d-p};
  }
  function memory(model,tp){
    const m=models[model];return {weightGiB:m.params*1e9*2/2**30/tp,kvMiB:m.full*4096*2048/2**20,gdnMiB:m.gdn*32*128*128*4/2**20};
  }
  const api={models,experiments,budget,memory};root.QwenHybridModels=api;
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
})(typeof window==='undefined'?globalThis:window);
