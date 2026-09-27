# Publish checklist

I built and verified the code. These steps need **your** accounts/credentials —
that's the boundary I can't cross for you. Do them in order.

## 0. Name (do this first — it blocks everything else)
You said you'd type the exact name. Tell me and I'll do a global rename:
brand in `web/index.html`, `package.json`, `capacitor.config.ts`,
`src-tauri/tauri.conf.json`, and the app title.
**Check availability before committing:** domain registrar + Google Play +
trademark search.

## 1. Backend live (the API everything talks to)
Pick one:
- **Free demo (HF ZeroGPU):** create a Space, push `deploy/hf-space/`, add the
  `backend/` app. Limit: 3.5 GPU-min/day, sleeps when idle. Fine for a demo.
- **Production (RunPod/Vast/Lambda):** build `deploy/runpod/Dockerfile`, run on a
  16 GB GPU (~$0.30–0.60/hr). This is what real traffic needs.
- **CPU-only:** deploy `deploy/Dockerfile.backend` with
  `DARREN_DETECTOR_ENGINE=stub` — honest but weak; use only to ship the shell.

Note the public URL, e.g. `https://api.yourdomain.com`.

## 2. Website
- The FastAPI app already serves the UI at `/`. Point your domain at it, or host
  `web/` on Netlify/Vercel/Cloudflare Pages and set the API base via
  `window.DARREN_API` (or `?api=`).
- Add HTTPS (most hosts do this automatically).

## 3. Desktop app
```bash
npm install
npm run desktop:build
```
- Produces installers in `src-tauri/target/release/bundle/`.
- **macOS:** needs an Apple Developer account ($99/yr) to sign + notarize, or
  users get a Gatekeeper warning.
- **Windows:** unsigned is fine for testing; code-signing cert needed to avoid
  SmartScreen warnings.

## 4. Android app
```bash
npm install
npm run android:add && npm run android:sync && npm run android:open
```
- In Android Studio: set `applicationId`, app icon, version, then
  **Build → Generate Signed Bundle (.aab)**.
- **Google Play:** $25 one-time developer account, then upload the `.aab`.
  First review ~1–7 days. You'll need: privacy policy URL, screenshots,
  feature graphic, content rating questionnaire.

## 5. Store assets you must supply
- App icon (512×512 + adaptive), feature graphic (1024×500)
- 2–8 screenshots per device type
- Short + full description
- Privacy policy page (required if you collect any data — say clearly what you
  send to the backend)

## 6. Compliance / honesty (important for a detector app)
- Google Play and Apple both scrutinise "AI detection" claims. **Do not claim
  it proves authorship.** Keep the disclaimer visible.
- If you market the humanizer as "bypass Turnitin", you risk store rejection and
  institutional-policy problems. Frame it as a *writing-style editor*.
- Detectors discriminate against non-native English writers (documented in the
  literature). State that you don't recommend high-stakes decisions from it.

## 7. CI
`.github/workflows/ci.yml` runs the test suite on every push — free on GitHub.
