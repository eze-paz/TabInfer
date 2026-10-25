// Usage: python demo/serve.py &  then  node tests/e2e-cpu-bonsai.mjs [url]
// Needs: npm i -D playwright (Chromium). A persistent profile dir (.pw-profile) keeps the 1.16 GB weight cache and a normal storage quota.
// Headless check of the CPU ternary engine path (nested workers: not runnable in the in-app browser pane).
import { chromium } from 'playwright';
const url = process.argv[2] || 'http://127.0.0.1:8765/demo/index.html';
const browser = await chromium.launchPersistentContext(process.env.PW_PROFILE || '.pw-profile', { headless: true, args: ['--enable-features=SharedArrayBuffer'] });
const page = browser.pages()[0] || await browser.newPage();
const lines = [];
page.on('console', m => { const t = m.text(); if (/bonsai|perf|cpukern|error|Error/i.test(t)) { lines.push(t); console.log('[console]', t.slice(0, 200)); } });
page.on('pageerror', e => console.log('[pageerror]', e.message));
await page.goto(url);
console.log('coi', await page.evaluate(() => self.crossOriginIsolated));
await page.selectOption('#model', 'bonsai-1.7b-cpu');
await page.fill('#prompt', 'Name three uses of liquid nitrogen, one sentence each.');
await page.click('#run');
const t0 = Date.now();
await page.waitForFunction(() => !document.getElementById('run').disabled, null, { timeout: 270000 }).catch(() => console.log('TIMEOUT waiting for run to finish'));
const r = await page.evaluate(() => ({ status: document.getElementById('status').textContent, stats: document.getElementById('stats').textContent, answer: document.getElementById('answer').textContent.slice(0, 600) }));
console.log('elapsed s', ((Date.now() - t0) / 1000).toFixed(0)); console.log(JSON.stringify(r, null, 1));
await browser.close();
