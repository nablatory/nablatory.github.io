# Publishing Nablatory

Nothing in this repository creates an organisation, pushes code or deploys a
site on its own. Each step below is taken by a maintainer, by hand.

## 1. Check the name again

The name was checked on 2026-10-14 (see `docs/ARCHITECTURE.md`, section 1).
Check again immediately before creating anything:

```sh
gh api users/nablatory            # expect 404 (name free)
npm view nablatory                # expect E404
curl -s -o /dev/null -w '%{http_code}\n' https://pypi.org/pypi/nablatory/json   # expect 404
```

## 2. Create the organisation and repository

1. Create the GitHub organisation `nablatory` (github.com/account/organizations/new).
2. Create the public repository `nablatory/nablatory.github.io` for an
   organisation site served at `https://nablatory.github.io/`, or
   `nablatory/nablatory` for a project site at `https://nablatory.github.io/nablatory/`.
3. Push:

   ```sh
   git remote add origin git@github.com:nablatory/<repo>.git
   git push -u origin main
   ```

## 3. Configure Pages

* Settings → Pages → Source: **GitHub Actions**. "Deploy from a branch" would
  publish the unbuilt source, which does not run.
* The deploy workflow takes the base path from GitHub Pages
  (`actions/configure-pages`): `/` for `<owner>.github.io` or a custom domain,
  `/<repo>/` for a project site. To override it, add the repository variable
  `NABLATORY_BASE` (Settings → Secrets and variables → Actions → Variables).

## 4. Deploy

Actions → **Deploy to GitHub Pages** → *Run workflow*. The workflow runs the
tests, builds, writes an `index.html` into every route directory
(`scripts/emit-routes.ts`) so direct links return HTTP 200, adds `404.html` and
`.nojekyll`, and publishes `dist/`.

The deploy workflow has only a manual trigger on purpose. Change it to run on
pushes to `main` only if the maintainers want every merge published.

## 5. Before each release

```sh
npm ci
npm run check              # format, lint, typecheck, tests
npm run build
npx vite preview           # open http://localhost:4173 and click through
```

* Run the gradient check of all models from the About page in Chrome, Firefox
  and Safari.
* Open every model page once; each must open, step and train without errors.
* Confirm `public/data/LICENSES.md` lists every shipped dataset.
* Confirm that every checkpoint in `public/weights/` has a script in
  `scripts/train/` that reproduces it and that its recorded metric is
  reproduced by its test.

## 6. Size budget

The site must load over a slow connection. Keep the initial JavaScript under
~400 kB gzipped, each checkpoint under 2 MB, and the whole `dist/` under 30 MB.
`npm run build` prints the bundle sizes.
