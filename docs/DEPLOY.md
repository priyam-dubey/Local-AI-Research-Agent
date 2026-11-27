# Deploying on a free tier

> Free-tier terms change often. The limits below were checked in **November 2026** — confirm them on each provider's page before you commit.

## Read this first

1. **Set `ACCESS_PASSWORD` for any deployment others can reach.** Without it, anyone with the URL can use *your* Gemini key, read your documents and delete them. With it, the whole API (and the key/settings endpoints) needs a sign-in; the API docs are hidden. Use a long random password (e.g. `openssl rand -base64 24`). It is a **single-owner** gate, not multi-user accounts.
2. **Use HTTPS.** Cloud Run and Cloudflare Tunnel give you HTTPS automatically. Don't expose the app on plain `http://` over the internet: the password would travel unencrypted.
3. **Memory:** the local embedding model (PyTorch + MiniLM) needs roughly 1.5–2 GB RAM in practice (my estimate; not measured). Hosts with 512 MB will not work. This is true even if you only use Gemini, because embeddings always run locally.
4. **Ollama is not practical on free hosting** (no GPU, little RAM). Use **Gemini** mode when deployed.

## What does *not* work as a free host (checked)

| Host | Why not |
|---|---|
| Hugging Face Spaces (Docker) | HF's docs now say creating a Docker/Gradio Space on compute requires a paid plan; only Static Spaces are free. |
| Render free web service | Reported 512 MB RAM and sleeps after 15 min — too small for the embedding model. |
| Vercel / Netlify / GitHub Pages | Static or short serverless functions only; this app needs a long-running Python process. |

## Option A — your own computer + a free HTTPS tunnel (zero cost, simplest)

Your data stays on your machine; the link works only while your computer and the app are running.

1. Follow the README quick start, and in `backend/.env` set `ACCESS_PASSWORD=...` (and `GEMINI_API_KEY=...` if you like).
2. Start the app: `cd backend && uvicorn app.main:app --port 8000`.
3. Install [`cloudflared`](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/) and run, in a second terminal:
   ```
   cloudflared tunnel --url http://localhost:8000
   ```
   It prints a random `https://….trycloudflare.com` address (no account needed; it changes on each start). Share it only with people who should have the password.

## Option B — Oracle Cloud "Always Free" VM (persistent, best specs)

* Needs a **credit or debit card** (not prepaid/virtual) and a phone number for sign-up; you are not charged unless you upgrade.
* The free Arm (Ampere A1) allowance was **reported to be cut in mid-2026 to 2 OCPUs / 12 GB** (older docs still say 4 / 24). That is still plenty for this app. "Out of capacity" errors when creating the VM are common — retry later or try another region.
* You get a real disk, so uploaded documents survive restarts.

Steps (Ubuntu 22.04/24.04 image, shape `VM.Standard.A1.Flex`, 2 OCPU / 12 GB):

```bash
# on the VM (ssh in)
curl -fsSL https://get.docker.com | sudo sh && sudo usermod -aG docker $USER && newgrp docker
git clone --branch feature/hybrid-platform https://github.com/priyam-dubey/Local-AI-Research-Agent.git
cd Local-AI-Research-Agent
docker build -t research-agent .          # 10–25 min the first time (downloads PyTorch + the embedding model)

cat > ~/ra.env <<'ENV'
DEFAULT_PROVIDER=gemini
ACCESS_PASSWORD=change-me-to-a-long-random-password
GEMINI_API_KEY=your-gemini-key
ENV
chmod 600 ~/ra.env

docker run -d --name research-agent --restart unless-stopped \
  -p 127.0.0.1:8080:8080 -v ra-data:/data --env-file ~/ra.env research-agent
```
Publish it over HTTPS **without opening any firewall port**, using a Cloudflare tunnel (install the `arm64` package of `cloudflared` from the link in Option A):
```bash
cloudflared tunnel --url http://localhost:8080      # prints a https://….trycloudflare.com URL
```
For a permanent address you need a (free) Cloudflare account and a domain you control — see Cloudflare's "named tunnel" guide. Run `cloudflared` as a service (`sudo cloudflared service install`) so it survives reboots.

## Option C — Google Cloud Run (serverless; needs a billing account)

* Needs a Google Cloud **billing account** (card) even though there is an always-free monthly quota (new accounts also get trial credit).
* Free quota (us-central1, "CPU always allocated"): **240,000 vCPU-seconds and 450,000 GiB-seconds per month**. With `--cpu 1 --memory 2Gi` that is about **60 hours of running time per month**; the instance scales to zero when idle. Other regions have different (sometimes smaller) quotas — check the pricing page.
* **No persistent disk.** Uploaded documents, indexes, the SQLite database and research history live in the instance's in-memory filesystem (which also uses your RAM) and **disappear when the instance scales to zero or restarts**. Treat it as a demo: re-upload after a cold start. Keep `GEMINI_API_KEY` in the environment, because a key pasted in the UI is also lost.
* `--max-instances 1` keeps all state in one instance; `--no-cpu-throttling` is required because uploads are processed in the background after the HTTP response.

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

# secrets file — it is git-ignored and docker-ignored; never commit it
cat > env.yaml <<'YAML'
DEFAULT_PROVIDER: "gemini"
ACCESS_PASSWORD: "change-me-to-a-long-random-password"
GEMINI_API_KEY: "your-gemini-key"
YAML

gcloud run deploy research-agent --source . --region us-central1 \
  --memory 2Gi --cpu 1 --no-cpu-throttling --min-instances 0 --max-instances 1 \
  --timeout 900 --allow-unauthenticated --env-vars-file env.yaml
```
(`--allow-unauthenticated` is intended: the app has its own password.) It prints an `https://….run.app` URL. Set a **budget alert** in the Cloud console so a mistake can't cost you money silently.

## After deploying — checklist

* Open the URL → you should see the **sign-in page**, and `https://YOUR-URL/docs` should be 404.
* Sign in, check the sidebar shows **Gemini (Online)** selectable with a *Valid API Key* badge (or paste the key once if you didn't set it in the environment).
* Upload a small document, wait for **Processed**, ask a question, confirm a cited answer.
* Never put real secrets in a public repository; rotate any key that was ever pasted into a chat or committed.

## Limits of this setup

Single shared password and shared data (no per-user accounts); Gemini receives your questions and retrieved excerpts; web research sends search queries to DuckDuckGo; free tiers can change or throttle without notice. The Dockerfile was built from verified pieces (frontend build, clean-venv backend install, simulated container layout and environment) but **I could not run `docker build` in my environment**, so expect to fix small things on the first build and tell me if you do.
