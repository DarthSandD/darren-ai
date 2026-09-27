import shutil, os

V = 'web/vendor'

# Use the full ESM transformers.web.js (self-contained webpack bundle)
shutil.copy('node_modules/@huggingface/transformers/dist/transformers.web.js', f'{V}/transformers.web.js')

# Remove the UMD/minified copy (won't work with import())
os.remove(f'{V}/transformers.web.min.js')
os.remove(f'{V}/ort-wasm-simd-threaded.jsep.mjs')
os.remove(f'{V}/ort-wasm-simd-threaded.jsep.wasm')

print('')
print('=== vendor contents ===')
total = 0
for root, dirs, files in os.walk(V):
    for f in sorted(files):
        sz = os.path.getsize(os.path.join(root, f))
        rel = os.path.relpath(os.path.join(root, f), V).replace('\\', '/')
        print(f'  {rel}: {sz/1e6:.2f} MB')
        total += sz
print('')
print(f'total: {total/1e6:.2f} MB')
