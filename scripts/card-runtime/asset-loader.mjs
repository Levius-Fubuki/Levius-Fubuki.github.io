// Hedge slow custom-domain requests against the same public repository bytes.
// Complete bodies race, not just headers; cancel the losing transfer.
export async function fetchCardResource(path, {timeout = 4000, hedgeDelay = 500, onSource = () => {}} = {}) {
  const local = new URL(path, location.href);
  const mirror = local.origin === location.origin && local.pathname.startsWith('/card/')
    ? new URL(local.pathname.slice(1), 'https://raw.githubusercontent.com/Levius-Fubuki/Levius-Fubuki.github.io/main/')
    : null;
  async function read(url, controller, limit) {
    const timer = setTimeout(() => controller.abort(), limit);
    try {
      const response = await fetch(url, {signal: controller.signal});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return {blob: await response.blob(), source: url.origin};
    } finally { clearTimeout(timer); }
  }
  const controllers = [new AbortController(), new AbortController()];
  let hedgeTimer, cause;
  try {
    const requests = [read(local, controllers[0], timeout)];
    if (mirror) requests.push(new Promise((resolve, reject) => {
      hedgeTimer = setTimeout(() => read(mirror, controllers[1], 10000).then(resolve, reject), hedgeDelay);
    }));
    const result = await Promise.any(requests);
    onSource(result.source);
    return result.blob;
  } catch (error) { cause = error; }
  finally {
    clearTimeout(hedgeTimer);
    controllers.forEach(controller => controller.abort());
  }
  if (mirror) {
    // One final uncached retry if both delivery paths failed.
    mirror.searchParams.set('card_retry', Date.now());
    const retry = new AbortController();
    try {
      const result = await read(mirror, retry, 10000);
      onSource(result.source);
      return result.blob;
    } catch (error) { cause = error; }
    finally { retry.abort(); }
  }
  throw new Error(`${local.pathname.split('/').pop()} could not load. Please try again.`, {cause});
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
