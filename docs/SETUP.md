# Setup: from zip → live website (no installs needed)

This walkthrough gets you from "downloaded zip" to "working website at
https://thekeeks.github.io/countme.in/" using only a browser.

## 1. Unzip the project

- **iPad / iPhone**: tap the downloaded zip in the Files app → it expands
  in place.
- **Mac**: double-click the zip in Finder.

You should now have a folder called `countme.in` with files inside.

## 2. Create the GitHub repo

1. Go to https://github.com/new (sign in if needed)
2. Repository name: `countme.in` (or whatever you want; the URL will reflect it)
3. Visibility: **Public** (required for free GitHub Pages hosting on personal
   accounts; private works too but needs a paid plan for Pages)
4. **Do not** check "Add a README file" — we already have one
5. Click **Create repository**

## 3. Upload the files

1. On the empty repo's page, click the link that says
   **"uploading an existing file"** (it's in the gray box of suggested
   next steps, or under Add file → Upload files)
2. Drag the **contents** of the `countme.in` folder into the upload area.
   That means: select everything inside the folder (README.md, tooling/, web/,
   etc.) and drag them in — not the folder itself
3. Scroll down, type a commit message like "initial commit", click
   **Commit changes**

GitHub will show you the populated repo within a few seconds.

## 4. Turn on GitHub Pages

1. In your repo, click **Settings** (top nav, not your profile settings)
2. Left sidebar → **Pages**
3. Under **Build and deployment** → **Source**, select **GitHub Actions**
4. Done. No "save" button on that page.

## 5. Wait for the first deploy

1. Click the **Actions** tab in the repo
2. You should see a workflow run named "Deploy to GitHub Pages" — it
   may show as still running, or already complete (~30 seconds)
3. When the green check appears, your site is live at:
   **`https://thekeeks.github.io/countme.in/`**

(Replace `thekeeks` with your GitHub username if different, and
`countme.in` with whatever you named the repo.)

## 6. Test it on your phone/iPad

1. Open the URL above in Safari (must be Safari for PWA install on iOS)
2. The home screen should show **Peggy O** in the setlist
3. Tap **Share** (square + arrow icon) → **Add to Home Screen**
4. The app icon appears on your home screen. Tap it: it opens full-screen
   like a native app
5. Tap Peggy O → tap the top of the screen → tap the red record button →
   allow microphone → watch the lyrics scroll

(The scrolling right now is on a timer, not actually listening to you yet —
that's Phase 3, which we'll build next in Claude Code on the web.)

## 7. Connect Claude Code on the web

Now you can iterate on the project without writing any code yourself:

1. Go to https://claude.ai/code
2. Sign in. If it's your first time, it'll ask to connect your GitHub
3. Pick your `countme.in` repo
4. From here, ask Claude to build features, fix bugs, add songs, etc.
   Each request becomes a pull request you review and merge from
   github.com

## Troubleshooting

**The Actions workflow failed with "Pages site not yet created"**
Run it once, then enable Pages (step 4), then re-run the workflow from
the Actions tab.

**The site loads but the song list is empty**
Hard-refresh (Cmd+Shift+R on desktop, or pull-to-refresh in mobile Safari).
Service workers cache aggressively; if you uploaded an empty repo first
and re-uploaded, the old empty version may be cached.

**Microphone won't start on iPhone**
Must be HTTPS for microphone access. GitHub Pages provides HTTPS by default,
so this only happens if you accidentally opened the page over http://.

**"Add to Home Screen" doesn't appear**
Only works in Safari on iOS. Chrome on iOS doesn't support PWA install
(Apple restriction). On Mac/desktop, use any modern browser.
