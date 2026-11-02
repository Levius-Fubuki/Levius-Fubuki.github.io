/* Teaching arithmetic is separated from historical measurements. No GPU model is run. */
((root, factory) => {
  const models = factory();
  if (typeof module === 'object' && module.exports) module.exports = models;
  if (root) root.Qwen30BModels = models;
})(typeof window === 'undefined' ? null : window, () => {
  'use strict';
  const chooseGraph = (active, buckets) => {
    const bucket = [...buckets].sort((a,b)=>a-b).find(n=>n>=active) ?? null;
    return {active, bucket, padding:bucket===null?0:bucket-active};
  };
  const memory = (requests, context, tp=2) => ({
    kvGiB: requests*context*48*2*4*128*2/tp/2**30,
    weightGiB: 30.5e9*2/tp/2**30,
    kvKiBPerToken: 48*2*4*128*2/tp/1024,
  });
  function route(tokens, hot=false) {
    let seed=30830;
    const rand=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/2**32;};
    const counts=Array(128).fill(0), assignments=[];
    for(let t=0;t<tokens;t++) {
      const chosen=Array.from({length:128},(_,e)=>({e,score:rand()+(hot&&e<16?0.85:0)})).sort((a,b)=>b.score-a.score).slice(0,8);
      const sum=chosen.reduce((s,x)=>s+Math.exp(x.score),0);
      assignments.push(chosen.map(x=>({expert:x.e,weight:Math.exp(x.score)/sum})));
      for(const x of chosen)counts[x.e]++;
    }
    return {counts,assignments,used:counts.filter(n=>n>0).length,total:tokens*8,rank0:counts.slice(0,64).reduce((a,b)=>a+b,0),rank1:counts.slice(64).reduce((a,b)=>a+b,0),uniformExpected:128*(1-(15/16)**tokens)};
  }
  const timing = (flopsGF,bytesGB,computeTF,bandwidthGBs,commMs,gapMs) => {
    const computeMs=flopsGF/computeTF,memoryMs=bytesGB/bandwidthGBs*1000;
    return {computeMs,memoryMs,kernelMs:Math.max(computeMs,memoryMs),stepMs:Math.max(computeMs,memoryMs)+commMs+gapMs};
  };
  const experiments = [
    {id:'capacity',title:'先解决容量：同卡双芯 TP2',hypothesis:'约56.8 GiB的BF16权重，单56 GiB逻辑设备还要留KV与工作区。',action:'用两个逻辑设备分片主要权重，核对运行时并发容量。',result:'主要权重理想份额约28.4 GiB/rank；这不是完整显存峰值。',boundary:'容量推导支持部署选择，没有测得TP1与TP2的同负载速度倍率。',next:'在可运行基线上比较小批次与高并发执行。'},
    {id:'graph',title:'Graph：补齐小批次桶',hypothesis:'只捕获32桶可能让小批次的执行规格不匹配。',action:'[32]改为[4,8,16,32]。',result:'C1：10.88→24.39 tok/s；C16：215.33→216.77 tok/s。',boundary:'主要改善C1，不能把差额全部归于CPU launch。',next:'查看C16的Prefill/Decode组织与token预算。'},
    {id:'batch',title:'Token预算：2048→4096',hypothesis:'3072-token prompt的Prefill切块与混合调度可能有改进空间。',action:'提高max-num-batched-tokens，其他主要配置保持。',result:'C16：216.77→218.80 tok/s。6144吞吐微增，但TTFT明显恶化。',boundary:'不是将C16纯Decode变成4096-token矩阵；历史短测不证明细小差异的统计显著性。',next:'保留4096，再检查能否隐藏调度工作。'},
    {id:'async',title:'Async：减少调度间隙',hypothesis:'部分CPU准备可以与GPU执行重叠。',action:'同步切为--async-scheduling。',result:'C16：218.80→221.52 tok/s，约+1.24%。',boundary:'不是实现新调度器，不是API异步，也不证明CPU为主瓶颈。',next:'检查容量、专家内核与并行选择的边际收益。'},
    {id:'kv',title:'KV：自动预算并未稳定更快',hypothesis:'固定8 GiB KV可能限制容量或驻留。',action:'比较固定预算和自动KV，另测block800/1088。',result:'自动KV：C16约+0.9%，C32约−0.35%；block变化基本持平。',boundary:'扩容无收益不等于显存带宽不是瓶颈。',next:'保留固定预算，转向专家计算和通信。'},
    {id:'gemm',title:'专家内核：Triton与auto持平',hypothesis:'小专家GEMM可以通过分组执行改善。',action:'显式切换Triton MoE后端，与auto比较。',result:'C32：320.90→321.35 tok/s，约+0.14%。',boundary:'auto不是无优化；没有实际调用路径不能断言两者内核完全不同。',next:'保留auto。需要router/GEMM/combine分阶段剖析。'},
    {id:'ep',title:'EP2：端到端回归',hypothesis:'完整专家本地执行可能比专家内部TP更高效。',action:'同两个rank启用EP2，attention仍可采用TP2。',result:'EP2 C16为185.86，C32为230.52 tok/s；报告对照下约−16%/−28%。',boundary:'回归不等于已经证明物理带宽不足；分发、等待和负载偏斜尚未分解。',next:'关闭EP，固定候选并正式复测。'},
    {id:'final',title:'最终复测：记录产能与延迟',hypothesis:'筛选出的配置应在完整输出窗口保留收益。',action:'TP2/no EP/auto/Graph4–32/b4096/block800/8GiB/async。',result:'C16正式225.67 tok/s，64请求、0失败；C32短测320.90 tok/s。',boundary:'正式窗口不是长期SLA，C32延迟高于C16；不能将多项提升率累乘。',next:'同服务采时间线、专家直方图、collective与可信内存流量，再定位剩余瓶颈。'},
  ];
  return {chooseGraph,memory,route,timing,experiments};
});
