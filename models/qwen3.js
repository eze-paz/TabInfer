// sandpie/modules/webgpu-qwen3.js — Qwen3 model layer for the hand-written
// WebGPU engine. Depends on window.SandpieWebGPU (the generic core in
// webgpu-engine.js) for GPU init, buffers, pipeline cache and dispatch.
//
// This file owns the MODEL-SPECIFIC kernels and (later) the forward graph,
// weight loader and tokenizer. See [[project_sandpie-webgpu-engine]].
//
// STATUS: kernel set + per-kernel CPU-reference self-tests. NO forward graph /
// loader / tokenizer yet — those assemble once every kernel here is green.
// Everything is f32 for first-correctness; f16/subgroup fusion comes after we
// produce a correct token.
//
// Qwen3-0.6B (verified from config.json + safetensors header):
//   28 layers · hidden 1024 · 16 q-heads / 8 kv-heads · head_dim 128 (explicit:
//   16*128=2048≠1024) · intermediate 3072 · vocab 151936 · tied embeds ·
//   RoPE θ=1e6 (full rotary) · RMSNorm eps 1e-6 · SwiGLU(silu) · no bias ·
//   per-head q_norm/k_norm (RMSNorm over head_dim, applied BEFORE RoPE).
//   Linear weights stored [out,in] row-major → y = x · Wᵀ.

const SandpieQwen3 = (function () {
  'use strict';

  const E = (typeof window !== 'undefined') ? window.SandpieWebGPU : null;

  const CONFIG = {
    numLayers: 28, hidden: 1024,
    nHeads: 16, nKvHeads: 8, headDim: 128,
    intermediate: 3072, vocab: 151936,
    ropeTheta: 1000000, rmsEps: 1e-6,
    tieEmbeddings: true,
  };

  const U = GPUBufferUsage;
  const ST = () => (U.STORAGE | U.COPY_DST | U.COPY_SRC);

  // Small uniform buffer from a Uint32Array/Float32Array (padded to 16 bytes).
  function uniform(arr) {
    const bytes = Math.max(16, Math.ceil(arr.byteLength / 16) * 16);
    const buf = E.createBuffer(bytes, U.UNIFORM | U.COPY_DST, 'u');
    E.device().queue.writeBuffer(buf, 0, arr.buffer, arr.byteOffset, arr.byteLength);
    return buf;
  }

  // ============================================================
  // Kernel 1 — RMSNorm.  y[T,H] = x / sqrt(mean(x²)+eps) * w[H]
  // One workgroup per row; WG_H threads cooperatively reduce over H.
  // ============================================================
  const WG_H = 256;
  const RMSNORM_WGSL = `
struct P { T:u32, H:u32, eps:f32, _p:u32 };
@group(0) @binding(0) var<storage, read>       x : array<f32>;
@group(0) @binding(1) var<storage, read>       w : array<f32>;
@group(0) @binding(2) var<storage, read_write> y : array<f32>;
@group(0) @binding(3) var<uniform>             p : P;
var<workgroup> red : array<f32, ${WG_H}>;
@compute @workgroup_size(${WG_H},1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>) {
  let row = wg.x; let H = p.H; let base = row*H;
  var s : f32 = 0.0;
  var i = lid.x;
  loop { if (i >= H) { break; } let v = x[base+i]; s = s + v*v; i = i + ${WG_H}u; }
  red[lid.x] = s; workgroupBarrier();
  var stride = ${WG_H}u/2u;
  loop { if (stride==0u){break;} if (lid.x<stride){red[lid.x]=red[lid.x]+red[lid.x+stride];} workgroupBarrier(); stride=stride/2u; }
  let inv = inverseSqrt(red[0]/f32(H) + p.eps);
  i = lid.x;
  loop { if (i >= H) { break; } y[base+i] = x[base+i]*inv*w[i]; i = i + ${WG_H}u; }
}`;
  function rmsnorm(xBuf, wBuf, yBuf, T, H, eps) {
    const pipe = E.getPipeline('q3.rmsnorm', RMSNORM_WGSL);
    const p = E.createBuffer(16, U.UNIFORM | U.COPY_DST, 'u');
    // pack T,H (u32) + eps (f32) into the 16-byte uniform
    const u = new ArrayBuffer(16); const du = new DataView(u);
    du.setUint32(0, T, true); du.setUint32(4, H, true); du.setFloat32(8, eps, true);
    E.device().queue.writeBuffer(p, 0, u);
    return E.dispatch(pipe, [xBuf, wBuf, yBuf, p], [T, 1, 1], { await: true });
  }

  // ============================================================
  // Kernel 2 — Linear (transposed weight).  y[T,N] = x[T,K] · W[N,K]ᵀ
  // W is stored [out=N, in=K] row-major (safetensors nn.Linear layout), so
  // y[t,n] = Σ_k x[t,k]·W[n,k]. Tiled 16×16; reads W transposed in-place (no
  // explicit transpose of the big weight matrices).
  // ============================================================
  const TILE = 16;
  const LINEAR_WGSL = `
struct D { T:u32, N:u32, K:u32, _p:u32 };
@group(0) @binding(0) var<storage, read>       x : array<f32>;
@group(0) @binding(1) var<storage, read>       W : array<f32>;
@group(0) @binding(2) var<storage, read_write> y : array<f32>;
@group(0) @binding(3) var<uniform>             d : D;
var<workgroup> tX : array<array<f32, ${TILE}>, ${TILE}>;
var<workgroup> tW : array<array<f32, ${TILE}>, ${TILE}>;
@compute @workgroup_size(${TILE},${TILE},1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>){
  let row=gid.y; let col=gid.x; let T=d.T; let N=d.N; let K=d.K;
  var acc:f32=0.0;
  let nT=(K+${TILE}u-1u)/${TILE}u;
  for (var t:u32=0u; t<nT; t=t+1u){
    let xCol=t*${TILE}u+lid.x;
    let wCol=t*${TILE}u+lid.y;   // index into K dim of W[col, K]
    tX[lid.y][lid.x]=select(0.0, x[row*K+xCol], (row<T)&&(xCol<K));
    tW[lid.y][lid.x]=select(0.0, W[col*K+wCol], (col<N)&&(wCol<K));
    workgroupBarrier();
    for (var k:u32=0u;k<${TILE}u;k=k+1u){ acc=acc+tX[lid.y][k]*tW[k][lid.x]; }
    workgroupBarrier();
  }
  if((row<T)&&(col<N)){ y[row*N+col]=acc; }
}`;
  function linearT(xBuf, wBuf, yBuf, T, N, K) {
    const pipe = E.getPipeline('q3.linearT', LINEAR_WGSL);
    const d = uniform(new Uint32Array([T, N, K, 0]));
    return E.dispatch(pipe, [xBuf, wBuf, yBuf, d], [Math.ceil(N/TILE), Math.ceil(T/TILE), 1], { await: true });
  }

  // ============================================================
  // Kernel 3 — Embedding gather.  y[T,H] = embed[ids[t], :]
  // ============================================================
  const EMBED_WGSL = `
struct P { T:u32, H:u32, _a:u32, _b:u32 };
@group(0) @binding(0) var<storage, read>       ids   : array<u32>;
@group(0) @binding(1) var<storage, read>       embed : array<f32>;
@group(0) @binding(2) var<storage, read_write> y     : array<f32>;
@group(0) @binding(3) var<uniform>             p     : P;
@compute @workgroup_size(64,1,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x; let total=p.T*p.H; if(idx>=total){return;}
  let t=idx/p.H; let h=idx%p.H;
  y[idx]=embed[ids[t]*p.H + h];
}`;
  function embedGather(idsBuf, embedBuf, yBuf, T, H) {
    const pipe = E.getPipeline('q3.embed', EMBED_WGSL);
    const p = uniform(new Uint32Array([T, H, 0, 0]));
    return E.dispatch(pipe, [idsBuf, embedBuf, yBuf, p], [Math.ceil((T*H)/64), 1, 1], { await: true });
  }

  // ============================================================
  // Kernel 4 — RoPE + per-head QK-norm (fused), for one tensor (q or k).
  // in[T, nH*hd] → out[T, nH*hd]. Per (t,head): RMSNorm over hd * normW[hd],
  // then RoPE (full rotary, rotate_half) at absolute position posBase+t.
  // One workgroup per (t,head); hd threads.
  // ============================================================
  const ROPEQK_WGSL = `
struct P { T:u32, nH:u32, hd:u32, posBase:u32, theta:f32, eps:f32, _a:u32, _b:u32 };
@group(0) @binding(0) var<storage, read>       inp  : array<f32>;
@group(0) @binding(1) var<storage, read>       normW: array<f32>;
@group(0) @binding(2) var<storage, read_write> out  : array<f32>;
@group(0) @binding(3) var<uniform>             p    : P;
var<workgroup> red : array<f32, 128>;
var<workgroup> nrm : array<f32, 128>;
@compute @workgroup_size(128,1,1)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>){
  let hd=p.hd; let j=lid.x;
  let unit=wg.x;            // 0 .. T*nH-1
  let t=unit/p.nH; let head=unit%p.nH;
  let base=t*(p.nH*hd) + head*hd;
  // RMSNorm over hd
  var v:f32=0.0; if(j<hd){ v=inp[base+j]; }
  red[j]=select(0.0, v*v, j<hd); workgroupBarrier();
  var stride=64u;
  loop{ if(stride==0u){break;} if(j<stride){red[j]=red[j]+red[j+stride];} workgroupBarrier(); stride=stride/2u; }
  let inv=inverseSqrt(red[0]/f32(hd)+p.eps);
  if(j<hd){ nrm[j]=v*inv*normW[j]; }
  workgroupBarrier();
  if(j>=hd){ return; }
  // RoPE rotate_half: pair j with j±hd/2
  let half=hd/2u;
  let pos=f32(p.posBase + t);
  let freqIdx = select(j-half, j, j<half);              // index into [0,half)
  let invFreq = pow(p.theta, -2.0*f32(freqIdx)/f32(hd));
  let ang = pos*invFreq;
  let c=cos(ang); let s=sin(ang);
  let xj=nrm[j];
  let partner = select(nrm[j-half], nrm[j+half], j<half);
  // j<half: out = x*cos - partner*sin ; j>=half: out = x*cos + partner*sin
  let rot = select(partner, -partner, j<half);
  out[base+j] = xj*c + rot*s;
}`;
  function ropeQK(inBuf, normWBuf, outBuf, T, nH, hd, posBase, theta, eps) {
    const pipe = E.getPipeline('q3.ropeqk', ROPEQK_WGSL);
    const u = new ArrayBuffer(32); const dv = new DataView(u);
    dv.setUint32(0,T,true); dv.setUint32(4,nH,true); dv.setUint32(8,hd,true);
    dv.setUint32(12,posBase,true); dv.setFloat32(16,theta,true); dv.setFloat32(20,eps,true);
    const p = E.createBuffer(32, U.UNIFORM|U.COPY_DST, 'u'); E.device().queue.writeBuffer(p,0,u);
    return E.dispatch(pipe, [inBuf, normWBuf, outBuf, p], [T*nH, 1, 1], { await: true });
  }

  // ============================================================
  // Kernel 5 — GQA causal attention.
  // Q[T, nHq*hd], K[S, nKv*hd], V[S, nKv*hd] → O[T, nHq*hd].
  // q-head h uses kv-head h/(nHq/nKv). Causal: key s attends iff s <= (S-T)+t
  // (so decode with T=1,S=cacheLen attends all; prefill T=S is lower-triangular).
  // One workgroup per (t, qhead); single-thread softmax over S (correctness-first).
  // ============================================================
  const ATTN_WGSL = `
struct P { T:u32, S:u32, nHq:u32, nKv:u32, hd:u32, _a:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage, read>       Q : array<f32>;
@group(0) @binding(1) var<storage, read>       K : array<f32>;
@group(0) @binding(2) var<storage, read>       V : array<f32>;
@group(0) @binding(3) var<storage, read_write> O : array<f32>;
@group(0) @binding(4) var<uniform>             p : P;
@compute @workgroup_size(64,1,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let unit=gid.x; if(unit>=p.T*p.nHq){return;}
  let t=unit/p.nHq; let hq=unit%p.nHq;
  let hd=p.hd; let grp=p.nHq/p.nKv; let hk=hq/grp;
  let qb=t*(p.nHq*hd)+hq*hd;
  let scale=1.0/sqrt(f32(hd));
  let last=(p.S-p.T)+t;
  var m:f32=-3.0e38;
  for(var s:u32=0u; s<=last; s=s+1u){
    let kb=s*(p.nKv*hd)+hk*hd;
    var dot:f32=0.0; for(var i:u32=0u;i<hd;i=i+1u){ dot=dot+Q[qb+i]*K[kb+i]; }
    dot=dot*scale; if(dot>m){m=dot;}
  }
  var denom:f32=0.0;
  for(var i:u32=0u;i<hd;i=i+1u){ O[qb+i]=0.0; }
  for(var s:u32=0u; s<=last; s=s+1u){
    let kb=s*(p.nKv*hd)+hk*hd; let vb=s*(p.nKv*hd)+hk*hd;
    var dot:f32=0.0; for(var i:u32=0u;i<hd;i=i+1u){ dot=dot+Q[qb+i]*K[kb+i]; }
    let w=exp(dot*scale - m); denom=denom+w;
    for(var i:u32=0u;i<hd;i=i+1u){ O[qb+i]=O[qb+i]+w*V[vb+i]; }
  }
  let invd=1.0/denom;
  for(var i:u32=0u;i<hd;i=i+1u){ O[qb+i]=O[qb+i]*invd; }
}`;
  function attention(qBuf, kBuf, vBuf, oBuf, T, S, nHq, nKv, hd) {
    const pipe = E.getPipeline('q3.attn', ATTN_WGSL);
    const p = uniform(new Uint32Array([T, S, nHq, nKv, hd, 0, 0, 0]));
    return E.dispatch(pipe, [qBuf, kBuf, vBuf, oBuf, p], [Math.ceil((T*nHq)/64), 1, 1], { await: true });
  }

  // ============================================================
  // Kernel 6 — SwiGLU.  y[T,I] = silu(gate[T,I]) * up[T,I]
  // ============================================================
  const SWIGLU_WGSL = `
struct P { n:u32, _a:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage, read>       gate : array<f32>;
@group(0) @binding(1) var<storage, read>       up   : array<f32>;
@group(0) @binding(2) var<storage, read_write> y    : array<f32>;
@group(0) @binding(3) var<uniform>             p    : P;
@compute @workgroup_size(64,1,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let i=gid.x; if(i>=p.n){return;}
  let g=gate[i]; let silu=g/(1.0+exp(-g)); y[i]=silu*up[i];
}`;
  function swiglu(gateBuf, upBuf, yBuf, n) {
    const pipe = E.getPipeline('q3.swiglu', SWIGLU_WGSL);
    const p = uniform(new Uint32Array([n, 0, 0, 0]));
    return E.dispatch(pipe, [gateBuf, upBuf, yBuf, p], [Math.ceil(n/64), 1, 1], { await: true });
  }

  // ============================================================
  // Kernel 7 — Residual add (in place).  a[n] += b[n]
  // ============================================================
  const ADD_WGSL = `
struct P { n:u32, _a:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage, read_write> a : array<f32>;
@group(0) @binding(1) var<storage, read>       b : array<f32>;
@group(0) @binding(2) var<uniform>             p : P;
@compute @workgroup_size(64,1,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){ let i=gid.x; if(i>=p.n){return;} a[i]=a[i]+b[i]; }`;
  function addInPlace(aBuf, bBuf, n) {
    const pipe = E.getPipeline('q3.add', ADD_WGSL);
    const p = uniform(new Uint32Array([n, 0, 0, 0]));
    return E.dispatch(pipe, [aBuf, bBuf, p], [Math.ceil(n/64), 1, 1], { await: true });
  }

  // ============================================================
  // Self-tests — each kernel vs a CPU reference. Returns {name, ok, err}[].
  // ============================================================
  function f32buf(arr) { return E.uploadF32(arr instanceof Float32Array ? arr : new Float32Array(arr), ST()); }
  function u32buf(arr) { const a = arr instanceof Uint32Array ? arr : new Uint32Array(arr); const b = E.createBuffer(a.byteLength, ST(), 'ids'); E.device().queue.writeBuffer(b,0,a.buffer,a.byteOffset,a.byteLength); return b; }
  const maxAbs = (a, b) => { let m = 0; for (let i=0;i<a.length;i++) m=Math.max(m, Math.abs(a[i]-b[i])); return m; };

  async function selfTestKernels() {
    await E.init();
    const out = [];
    const check = (name, err, tol) => out.push({ name, ok: err < (tol||1e-3), err });

    // --- rmsnorm ---
    {
      const T=3, H=512, eps=1e-6;
      const x=new Float32Array(T*H), w=new Float32Array(H);
      for(let i=0;i<x.length;i++)x[i]=Math.sin(i*0.07);
      for(let i=0;i<H;i++)w[i]=0.5+0.5*Math.cos(i*0.03);
      const y=new Float32Array(T*H);
      const xb=f32buf(x), wb=f32buf(w), yb=E.createBuffer(T*H*4, ST(),'y');
      await rmsnorm(xb,wb,yb,T,H,eps);
      const got=await E.readF32(yb,T*H);
      for(let t=0;t<T;t++){let ss=0;for(let i=0;i<H;i++)ss+=x[t*H+i]**2;const inv=1/Math.sqrt(ss/H+eps);for(let i=0;i<H;i++)y[t*H+i]=x[t*H+i]*inv*w[i];}
      check('rmsnorm', maxAbs(got,y));
      [xb,wb,yb].forEach(b=>b.destroy());
    }
    // --- linearT ---
    {
      const T=5,N=7,K=64;
      const x=new Float32Array(T*K),W=new Float32Array(N*K);
      for(let i=0;i<x.length;i++)x[i]=Math.sin(i*0.11);
      for(let i=0;i<W.length;i++)W[i]=Math.cos(i*0.05);
      const xb=f32buf(x),wb=f32buf(W),yb=E.createBuffer(T*N*4,ST(),'y');
      await linearT(xb,wb,yb,T,N,K);
      const got=await E.readF32(yb,T*N);
      const y=new Float32Array(T*N);
      for(let t=0;t<T;t++)for(let n=0;n<N;n++){let a=0;for(let k=0;k<K;k++)a+=x[t*K+k]*W[n*K+k];y[t*N+n]=a;}
      check('linearT', maxAbs(got,y));
      [xb,wb,yb].forEach(b=>b.destroy());
    }
    // --- embedGather ---
    {
      const T=4,H=32,V=20;
      const embed=new Float32Array(V*H); for(let i=0;i<embed.length;i++)embed[i]=i*0.01;
      const ids=new Uint32Array([3,0,19,7]);
      const ib=u32buf(ids),eb=f32buf(embed),yb=E.createBuffer(T*H*4,ST(),'y');
      await embedGather(ib,eb,yb,T,H);
      const got=await E.readF32(yb,T*H);
      const y=new Float32Array(T*H);
      for(let t=0;t<T;t++)for(let h=0;h<H;h++)y[t*H+h]=embed[ids[t]*H+h];
      check('embedGather', maxAbs(got,y));
      [ib,eb,yb].forEach(b=>b.destroy());
    }
    // --- ropeQK (rope + qk-norm) ---
    {
      const T=2,nH=2,hd=8,theta=10000,eps=1e-6,posBase=1;
      const inp=new Float32Array(T*nH*hd),nw=new Float32Array(hd);
      for(let i=0;i<inp.length;i++)inp[i]=Math.sin(i*0.3);
      for(let i=0;i<hd;i++)nw[i]=0.7+0.1*i;
      const ib=f32buf(inp),nb=f32buf(nw),ob=E.createBuffer(inp.length*4,ST(),'o');
      await ropeQK(ib,nb,ob,T,nH,hd,posBase,theta,eps);
      const got=await E.readF32(ob,inp.length);
      // CPU ref
      const y=new Float32Array(inp.length); const half=hd/2;
      for(let t=0;t<T;t++)for(let h=0;h<nH;h++){
        const base=t*(nH*hd)+h*hd; let ss=0; for(let j=0;j<hd;j++)ss+=inp[base+j]**2; const inv=1/Math.sqrt(ss/hd+eps);
        const nrm=new Float32Array(hd); for(let j=0;j<hd;j++)nrm[j]=inp[base+j]*inv*nw[j];
        const pos=posBase+t;
        for(let j=0;j<hd;j++){const fi=j<half?j:j-half;const ang=pos*Math.pow(theta,-2*fi/hd);const c=Math.cos(ang),s=Math.sin(ang);
          const partner=j<half?nrm[j+half]:nrm[j-half];const rot=j<half?-partner:partner;y[base+j]=nrm[j]*c+rot*s;}
      }
      check('ropeQK', maxAbs(got,y), 2e-3);
      [ib,nb,ob].forEach(b=>b.destroy());
    }
    // --- attention (GQA causal) ---
    {
      const T=3,S=3,nHq=4,nKv=2,hd=8;
      const Q=new Float32Array(T*nHq*hd),Kk=new Float32Array(S*nKv*hd),Vv=new Float32Array(S*nKv*hd);
      for(let i=0;i<Q.length;i++)Q[i]=Math.sin(i*0.2);
      for(let i=0;i<Kk.length;i++)Kk[i]=Math.cos(i*0.15);
      for(let i=0;i<Vv.length;i++)Vv[i]=Math.sin(i*0.09+1);
      const qb=f32buf(Q),kb=f32buf(Kk),vb=f32buf(Vv),ob=E.createBuffer(Q.length*4,ST(),'o');
      await attention(qb,kb,vb,ob,T,S,nHq,nKv,hd);
      const got=await E.readF32(ob,Q.length);
      const y=new Float32Array(Q.length); const grp=nHq/nKv; const scale=1/Math.sqrt(hd);
      for(let t=0;t<T;t++)for(let hq=0;hq<nHq;hq++){const hk=Math.floor(hq/grp);const qb2=t*(nHq*hd)+hq*hd;const last=(S-T)+t;
        let m=-1e38;for(let s=0;s<=last;s++){const kb2=s*(nKv*hd)+hk*hd;let d=0;for(let i=0;i<hd;i++)d+=Q[qb2+i]*Kk[kb2+i];d*=scale;if(d>m)m=d;}
        let den=0;const acc=new Float32Array(hd);for(let s=0;s<=last;s++){const kb2=s*(nKv*hd)+hk*hd;const vb2=s*(nKv*hd)+hk*hd;let d=0;for(let i=0;i<hd;i++)d+=Q[qb2+i]*Kk[kb2+i];const w=Math.exp(d*scale-m);den+=w;for(let i=0;i<hd;i++)acc[i]+=w*Vv[vb2+i];}
        for(let i=0;i<hd;i++)y[qb2+i]=acc[i]/den;}
      check('attention', maxAbs(got,y), 2e-3);
      [qb,kb,vb,ob].forEach(b=>b.destroy());
    }
    // --- swiglu ---
    {
      const n=300;const g=new Float32Array(n),u=new Float32Array(n);
      for(let i=0;i<n;i++){g[i]=Math.sin(i*0.1)*2;u[i]=Math.cos(i*0.07);}
      const gb=f32buf(g),ub=f32buf(u),yb=E.createBuffer(n*4,ST(),'y');
      await swiglu(gb,ub,yb,n);
      const got=await E.readF32(yb,n);const y=new Float32Array(n);
      for(let i=0;i<n;i++){const s=g[i]/(1+Math.exp(-g[i]));y[i]=s*u[i];}
      check('swiglu', maxAbs(got,y));
      [gb,ub,yb].forEach(b=>b.destroy());
    }
    // --- addInPlace ---
    {
      const n=257;const a=new Float32Array(n),b=new Float32Array(n);
      for(let i=0;i<n;i++){a[i]=i*0.5;b[i]=-i*0.2;}
      const ab=f32buf(a),bb=f32buf(b);
      await addInPlace(ab,bb,n);
      const got=await E.readF32(ab,n);const y=new Float32Array(n);for(let i=0;i<n;i++)y[i]=a[i]+b[i];
      check('addInPlace', maxAbs(got,y));
      [ab,bb].forEach(x=>x.destroy());
    }
    return out;
  }

  return {
    CONFIG,
    rmsnorm, linearT, embedGather, ropeQK, attention, swiglu, addInPlace,
    selfTestKernels,
  };
})();

if (typeof window !== 'undefined') window.SandpieQwen3 = SandpieQwen3;
