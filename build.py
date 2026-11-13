"""Build index.html from content.html + code/*.py"""
import html
import pathlib

ROOT = pathlib.Path(__file__).parent
codes = sorted((ROOT / "code").glob("*.py"))

TITLES = {
    "01": "MLP + softmax/CE + grad check (numpy)",
    "02": "LayerNorm / RMSNorm / BatchNorm backward (numpy)",
    "03": "Multi-head attention + masks (numpy)",
    "04": "Mini GPT: RoPE, KV cache, sampling, AdamW, warmup/cosine (torch)",
    "05": "SGD / Adam / AdamW from scratch (numpy, vs torch)",
    "06": "Debug drill: buggy vs fixed training loop (torch)",
    "07": "Classic ML: logreg, k-means, PCA, kNN, conv2d im2col (numpy)",
    "08": "RL losses: REINFORCE, GAE, PPO, GRPO, k3 KL (torch)",
    "09": "System-design snippets: worker pool, UCB, bootstrap, cache, halving",
    "10": "LeetCode patterns: 16 patterns, ~35 canonical problems with tests",
    "11": "MHA + KV cache + GQA/MQA from scratch (torch)",
    "13": "LLM problems in LeetCode format (PyTorch): 21 problems, hidden tests vs torch reference",
    "14": "Online entropy / logsumexp / cross-entropy over vocab chunks, shard merging, streaming empirical entropy (torch)",
    "15": "Streaming / chunked problems: online KL, fused linear-CE, FlashAttention fwd, Welford, Gumbel, reservoir, grad accum, packing, spec decoding, linear attn",
    "12": "Perplexity: from logits, masking, corpus aggregation, sliding window, n-gram, bits/byte",
}

code_index = "".join(
    f'<li><a href="#c{p.name[:2]}">{p.name}</a> — {TITLES.get(p.name[:2], "")}</li>' for p in codes
)
code_blocks = "".join(
    f'''<div class="codeblock" id="c{p.name[:2]}">
<div class="codehead"><span>{p.name} — {TITLES.get(p.name[:2], "")}</span>
<span><a href="code/{p.name}" download>download</a> · <button onclick="copyCode(this)">copy</button></span></div>
<pre><code class="language-python">{html.escape(p.read_text())}</code></pre></div>'''
    for p in codes
)

page = f"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<title>ML Coding & System Design Prep</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
<style>
:root {{ --fg:#1f2328; --muted:#656d76; --bg:#fff; --card:#f6f8fa; --accent:#0969da; --border:#d0d7de; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Noto Sans SC",sans-serif; color:var(--fg); background:var(--bg); }}
nav {{ position:fixed; top:0; left:0; bottom:0; width:230px; padding:24px 18px; border-right:1px solid var(--border); overflow-y:auto; background:#fafbfc; }}
nav h1 {{ font-size:16px; margin:0 0 16px; }}
nav a {{ display:block; color:var(--fg); text-decoration:none; padding:4px 0; font-size:14px; }}
nav a:hover {{ color:var(--accent); }}
nav .sub {{ padding-left:12px; font-size:13px; color:var(--muted); }}
main {{ margin-left:230px; padding:32px 48px 80px; max-width:1100px; }}
h2 {{ border-bottom:1px solid var(--border); padding-bottom:6px; margin-top:48px; }}
h3 {{ margin-top:28px; }}
a {{ color:var(--accent); }}
table {{ border-collapse:collapse; width:100%; margin:12px 0; font-size:14px; }}
th,td {{ border:1px solid var(--border); padding:6px 10px; vertical-align:top; text-align:left; }}
th {{ background:var(--card); }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
.card {{ background:var(--card); border:1px solid var(--border); border-radius:8px; padding:4px 18px; }}
.callout {{ border-left:4px solid var(--accent); background:#ddf4ff; padding:10px 14px; border-radius:4px; margin:14px 0; }}
.muted {{ color:var(--muted); }}
.cols2 {{ columns:2; column-gap:32px; }}
details {{ border:1px solid var(--border); border-radius:8px; padding:8px 14px; margin:10px 0; }}
summary {{ cursor:pointer; }}
code {{ background:var(--card); padding:1px 5px; border-radius:4px; font-size:13px; }}
.codeblock {{ border:1px solid var(--border); border-radius:8px; margin:22px 0; overflow:hidden; }}
.codehead {{ display:flex; justify-content:space-between; background:var(--card); padding:8px 14px; font-weight:600; font-size:14px; border-bottom:1px solid var(--border); }}
.codehead button {{ border:1px solid var(--border); background:#fff; border-radius:4px; cursor:pointer; font-size:12px; }}
pre {{ margin:0; max-height:640px; overflow:auto; }}
pre code {{ background:none; padding:14px 16px !important; font-size:12.5px; line-height:1.5; }}
@media (max-width:900px) {{ nav {{ display:none; }} main {{ margin-left:0; padding:20px; }} .grid2 {{ grid-template-columns:1fr; }} .cols2 {{ columns:1; }} }}
</style>
</head>
<body>
<nav>
<h1>ML Interview Prep</h1>
<a href="#overview">0. 面试形式</a>
<a href="#plan">1. 准备计划</a>
<a href="#mlcoding">2. ML Coding 题库</a>
<a class="sub" href="#onlineentropy">2.3c Online entropy</a>
<a class="sub" href="#leetcode">2.8 LeetCode 风格</a>
<a class="sub" href="#mlleetcode">2.9 LLM × LeetCode 格式</a>
<a class="sub" href="#streaming">2.10 Streaming / 分块题</a>
<a href="#sysdesign">3. System Design</a>
<a href="#numbers">4. 要背熟的数字</a>
<a href="#code">5. 代码</a>
{''.join(f'<a class="sub" href="#c{p.name[:2]}">{p.name}</a>' for p in codes)}
</nav>
<main>
<h1 style="margin-top:0">ML Coding &amp; System Design 面试准备</h1>
<p class="muted">所有代码都实际跑过（numpy + PyTorch CPU），每个脚本末尾有 assert 自检，跑到 <code>OK</code> 就说明正确。</p>
{(ROOT / "content.html").read_text()}
<section id="code">
<h2>5. 代码</h2>
<p>在本地或 Colab 运行：<code>pip install numpy torch</code>，然后 <code>python 01_mlp_numpy.py</code>。建议先自己从空白写一遍，再来对照。</p>
<ul>{code_index}</ul>
{code_blocks}
</section>
</main>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script>
hljs.highlightAll();
function copyCode(btn) {{
  const code = btn.closest('.codeblock').querySelector('code').innerText;
  navigator.clipboard.writeText(code).then(() => {{ btn.textContent = 'copied'; setTimeout(() => btn.textContent = 'copy', 1200); }});
}}
</script>
</body>
</html>
"""
(ROOT / "index.html").write_text(page)
print("wrote index.html", len(page), "bytes,", len(codes), "code files")
