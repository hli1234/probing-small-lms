# Publishing this blog to GitHub Pages

Your blog ("Probing small LMs", 3 posts) is a Quarto website. Below are
the exact steps to put it online. You only do the **first-time setup**
once; after that, re-publishing edits is a single command.

Open Terminal and run these from the project folder.

---

## First-time setup

```bash
cd ~/Desktop/eigenation/AI

# Start a clean git repo (removes the partial one and re-inits cleanly)
rm -rf .git
git init
git add -A
git commit -m "Initial commit: Quarto blog source"
git branch -M main
```

### Create the GitHub repo + push

**If you have the GitHub CLI (`gh`) installed and logged in** — easiest:

```bash
gh repo create probing-small-lms --public --source=. --remote=origin --push
```

**Otherwise, do it manually:**

1. Go to https://github.com/new
2. Name the repo `probing-small-lms`, set it **Public**, and **do not**
   add a README/.gitignore (you already have them). Click *Create*.
3. Back in Terminal (replace `YOURNAME` with your GitHub username):

```bash
git remote add origin https://github.com/YOURNAME/probing-small-lms.git
git push -u origin main
```

### Publish the rendered site

```bash
quarto publish gh-pages
```

Answer **yes** when it asks to publish. Quarto renders the site, pushes
it to a `gh-pages` branch, and configures GitHub Pages for you.

Your site goes live at:

```
https://YOURNAME.github.io/probing-small-lms/
```

(Give it 1–2 minutes the first time.)

---

## Re-publishing after you edit posts

Just two commands whenever you change content:

```bash
git add -A && git commit -m "Update posts" && git push   # save source
quarto publish gh-pages                                   # rebuild + deploy
```

---

## Notes

- `_site/` and `.quarto/` are git-ignored on purpose — `quarto publish`
  rebuilds them and pushes the output to the separate `gh-pages` branch.
- If GitHub Pages isn't showing, check repo **Settings → Pages**: source
  should be the `gh-pages` branch, root folder. Quarto usually sets this
  automatically.
- Want a custom domain (e.g. yourdomain.com)? Tell me and I'll walk you
  through the CNAME setup.
