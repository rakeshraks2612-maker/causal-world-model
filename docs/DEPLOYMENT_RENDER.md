# 🚀 Deploying PRISM to Render (render.com)

This guide provides the complete, step-by-step instructions to deploy PRISM to [Render](https://render.com).

PRISM includes two deployable services configured in [`render.yaml`](../render.yaml):
1. **`prism-web-dashboard` (Static Site):** The lightweight, responsive Web SPA dashboard. Completely free, instant build, zero memory overhead, and 100% uptime on Render's global CDN.
2. **`prism-operator-console` (Python Web Service):** The interactive Streamlit operator console supporting live scenario exploration, Pearl Level-3 counterfactual simulations, and uncertainty safety gating.

---

## Quick Deploy: 1-Click Render Blueprint (Recommended)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/rakeshraks2612-maker/causal-world-model)

1. Click the **Deploy to Render** button above (or navigate to [dashboard.render.com](https://dashboard.render.com)).
2. Log into your Render account.
3. Click **New +** $\to$ **Blueprint**.
4. Connect your GitHub repository:
   ```text
   https://github.com/rakeshraks2612-maker/causal-world-model
   ```
5. Render automatically discovers `render.yaml` and stages the two services:
   - `prism-web-dashboard` (Static site)
   - `prism-operator-console` (Web service)
6. Click **Apply**. Render will automatically build and assign live public HTTPS URLs to both services!

---

## Manual Deployment Options

If you prefer to configure services individually via the Render dashboard:

### Method 1: Deploy Web SPA Dashboard (Static Site)
* **Service Type:** `Static Site`
* **Name:** `prism-web-dashboard`
* **Repository:** `https://github.com/rakeshraks2612-maker/causal-world-model`
* **Branch:** `main`
* **Build Command:** `echo "PRISM Web Dashboard Ready"`
* **Publish Directory:** `prism/dashboard/web`
* **Rewrites / Redirects:**
  * Rewrite `/*` $\to$ `/index.html`

### Method 2: Deploy Streamlit Operator Console (Web Service)
* **Service Type:** `Web Service`
* **Name:** `prism-operator-console`
* **Repository:** `https://github.com/rakeshraks2612-maker/causal-world-model`
* **Branch:** `main`
* **Environment:** `Python 3`
* **Plan:** `Free`
* **Build Command:** `./render-build.sh`
* **Start Command:** `streamlit run scripts/run_dashboard.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
* **Health Check Path:** `/_stcore/health`
* **Environment Variables:**
  * `PYTHON_VERSION` = `3.11.9`
  * `STREAMLIT_SERVER_HEADLESS` = `true`
  * `STREAMLIT_SERVER_ENABLE_CORS` = `false`

---

## Verification After Deployment

Once deployment finishes, Render provides public URLs:
* **Web SPA:** `https://prism-web-dashboard.onrender.com`
* **Operator Console:** `https://prism-operator-console.onrender.com`

Verify the deployment:
1. Navigate to the Web SPA URL and test scenario transitions (`● S1` through `● S6`).
2. Verify that clicking **Run Autonomous Tour** launches the step-by-step presentation tour.
3. Verify that the **Inspect 421 Cryptographic Invariant Proofs** modal displays the SHA-256 evidence ledger.
