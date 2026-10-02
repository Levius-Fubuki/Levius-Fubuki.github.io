// Bound each request (including its body) and retry independently. A stalled
// optional layer must never leave the entire card waiting on an unbounded Image.
export async function fetchCardResource(path, {timeout = 12000, attempts = 3} = {}) {
  let cause;
  for (let attempt = 0; attempt < attempts; attempt++) {
    const url = new URL(path, location.href);
    if (attempt) url.searchParams.set('card_retry', `${Date.now()}-${attempt}`);
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    try {
      const response = await fetch(url, {signal: controller.signal, cache: attempt ? 'reload' : 'default'});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.blob();
    } catch (error) {
      cause = error;
      if (attempt + 1 < attempts) await new Promise(resolve => setTimeout(resolve, 250 * (attempt + 1)));
    } finally {
      clearTimeout(timer);
    }
  }
  const name = new URL(path, location.href).pathname.split('/').pop();
  throw new Error(`${name} 加载失败，请点击重试`, {cause});
}

export async function prepareCardAssets(config, progress = () => {}) {
  const groups = [config.assets, config.back?.assets].filter(Boolean);
  const paths = [...new Set(groups.flatMap(group => Object.values(group)))];
  let loaded = 0;
  const blobs = await Promise.all(paths.map(async path => {
    const blob = await fetchCardResource(path);
    progress(++loaded, paths.length);
    return [path, blob];
  }));
  const urls = new Map(blobs.map(([path, blob]) => [path, URL.createObjectURL(blob)]));
  for (const group of groups) for (const key of Object.keys(group)) group[key] = urls.get(group[key]);
  return () => { for (const url of urls.values()) URL.revokeObjectURL(url); };
}
