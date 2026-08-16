# SignLink Prototype

A browser-based hackathon prototype for **gesture → text → speech**, with a live camera feed,
real-time hand-skeleton tracking overlay, a broadcast-style caption bar, and a sentence
builder that chains individual gestures into full spoken sentences.

## What this prototype does

1. Opens your webcam and streams it live in the browser (no upload — everything runs on-device).
2. Uses Google's MediaPipe Gesture Recognizer, running in real time on every video frame,
   for 7 built-in hand shapes.
3. Adds 5 more gestures on top of that by classifying the raw 21-point hand landmarks itself
   (finger-extension + pinch geometry), so the vocabulary isn't capped by MediaPipe's stock model.
4. Draws the live 21-point hand skeleton on top of the video feed as it tracks your hand
   (tracks up to two hands at once).
5. Recognizes gestures and maps them to a demo vocabulary of 12 words.
6. Shows the translated word, lights up an equalizer-style confidence meter, and speaks the
   phrase aloud using the browser's Speech Synthesis API.
7. **Sentence Builder** — flip the "Sentence Mode" toggle and each held gesture adds its word
   to a growing sentence strip instead of being spoken immediately. Speak the full sentence,
   undo the last word, or clear it. A handful of common two-word combinations (e.g. HELLO →
   HELP) are auto-recognized and offered as a natural full sentence you can speak instead of
   the raw word list.
8. Keeps a scrolling transcript of recent single-word translations.

## Demo vocabulary

| Detected gesture | Demo output | Source |
|---|---|---|
| Open Palm | HELLO | MediaPipe built-in |
| Thumb Up | YES | MediaPipe built-in |
| Thumb Down | NO | MediaPipe built-in |
| Closed Fist | HELP | MediaPipe built-in |
| Victory (✌️) | THANK YOU | MediaPipe built-in |
| I Love You (🤟) | I LOVE YOU | MediaPipe built-in |
| Pointing Up (☝️) | WAIT | MediaPipe built-in |
| OK sign (👌) | OKAY | Custom landmark classifier |
| Three fingers (🖖 index+middle+ring) | PLEASE | Custom landmark classifier |
| Four fingers (🖐️ no thumb) | SORRY | Custom landmark classifier |
| Rock on (🤘 index+pinky) | STOP | Custom landmark classifier |
| Call me (🤙 thumb+pinky) | CALL ME | Custom landmark classifier |

**Important:** These mappings are intentionally a prototype/demo vocabulary. They are NOT a
claim that these gestures are the formal signs for those words in ISL/ASL. For a serious
sign-language product, train and validate a custom sign-language classifier with the target
language's dataset.

### How the 5 custom gestures work

MediaPipe's canned `GestureRecognizer` only ships 7 categories (`None` + the 6 original ones
+ `Pointing_Up`). `app.js` extends this by reading the 21 hand landmarks directly whenever the
canned classifier doesn't return a confident match, and classifying finger state itself:

- A finger counts as "extended" when its fingertip sits farther from the wrist than its own
  mid-joint — this stays roughly correct regardless of how the hand is rotated in frame, which
  is more forgiving for a live demo than comparing raw y-coordinates.
- OK sign is detected by measuring the pixel distance between the thumb tip and index tip.

These thresholds (`* 1.15` finger-extension margin, `0.06` pinch distance) are tuned for a
hand held roughly upright, filling a reasonable portion of the frame, at arm's length from a
laptop webcam. If a gesture isn't triggering reliably in your demo room's lighting/camera,
nudge those two constants in `classifyCustomGesture()` in `app.js`.

### Sentence phrase shortcuts

`PHRASES` in `app.js` maps a few natural two-word tails (last two words added to the sentence
strip) to a full sentence, e.g. `HELLO → HELP` suggests *"Hello, I need help please."* Add more
entries to that object to grow the phrase book.

## Run it

Browsers only grant camera access on secure origins — `https://`, or `http://localhost`.
A plain double-clicked `file://` page will not be allowed to open the camera, so serve the
folder locally.

### Option A — VS Code Live Server

1. Open this folder in VS Code.
2. Install the "Live Server" extension.
3. Right-click `index.html`.
4. Choose **Open with Live Server**.
5. Allow camera permission when the browser prompts you.

### Option B — Python

From this folder run:

```bash
python -m http.server 5500
```

Then open:

http://localhost:5500

### Camera troubleshooting

- **Permission denied**: check your browser's site settings (padlock icon in the address bar)
  and re-allow camera access, then reload.
- **No picture / black frame**: make sure no other app (Zoom, Teams, another browser tab) is
  already using the camera.
- **"AI failed to load"**: the model and inference engine load from a CDN on first run, so you
  need an internet connection the first time; after that MediaPipe's own caching keeps startup fast.
- Works best in Chrome, Edge, or Firefox on desktop. Mobile Safari support for `getUserMedia`
  can be inconsistent depending on iOS version.
- **Custom gestures (OK / Three / Four / Rock On / Call Me) not triggering**: hold your hand
  flatter to the camera and give it a second — these rely on landmark geometry, not the
  canned classifier, so they benefit from a clearer, closer view of the fingers.
- **OpenCV edge trace panel blank**: `opencv.js` is a large WASM download loaded from a CDN on
  page load and initializes asynchronously — give it a few seconds after the page loads before
  flipping the toggle on. If your network blocks `docs.opencv.org`, the toggle disables itself.
- **Suggested-sentence box showing empty**: fixed — it used to stay visible with no text because
  an author CSS rule (`.suggestion { display: flex }`) was overriding the browser's default
  `[hidden] { display: none }` at equal specificity. `style.css` now has an explicit
  `.suggestion[hidden] { display: none; }` rule so it only appears when there's a real
  suggestion to show. The suggestion itself only fires when the *last two* words added match
  one of the pairs in `PHRASES` — it's expected to stay hidden otherwise.

## Hackathon demo flow

1. Open SignLink, click **Start Camera**.
2. Hold your hand clearly in the camera — watch the live skeleton track your fingers.
3. Cycle through single-word gestures: Open Palm → HELLO, Thumb Up → YES, Closed Fist → HELP,
   then show off a custom one — OK sign → OKAY, or Rock On → STOP — to demonstrate the
   landmark-based classifier isn't limited to MediaPipe's stock shapes.
4. Flip **Sentence Mode** on. Hold HELLO, then HELP — the sentence strip fills with both words
   and a suggested full sentence ("Hello, I need help please.") appears. Tap **Speak this
   instead** to hear it, or **Speak sentence** for the raw word-by-word version.
5. Explain that the current prototype proves the real-time CV → language → speech pipeline,
   including composing gestures into sentences, not just isolated words.
6. Explain that the next version replaces both the canned and custom gesture classifiers with
   a trained ISL/ASL model, and extends the sentence builder with grammar-aware phrasing.

## Tech stack

SignLink is a layered, on-device computer-vision pipeline, not a single library wrapped in a
UI. Everything below runs client-side in vanilla JS — no build step, no framework, no backend.

**Capture & tracking**
- `getUserMedia` (WebRTC) for the live 1280×720 webcam feed
- MediaPipe Tasks Vision `GestureRecognizer`, loaded as a WASM module and running in
  `VIDEO` mode, GPU-delegated with automatic fallback to CPU delegation if the device/browser
  doesn't support GPU inference
- Tracks up to 2 hands simultaneously, 21 3D landmarks per hand at ~30 fps

**Gesture classification (two independent stages)**
1. MediaPipe's canned classifier for 7 built-in shapes (`Open_Palm`, `Thumb_Up`, `Thumb_Down`,
   `Closed_Fist`, `Victory`, `ILoveYou`, `Pointing_Up`)
2. A custom geometric classifier (`classifyCustomGesture` in `app.js`) written against the raw
   landmark set — wrist-relative distance ratios for finger extension, Euclidean thumb–index
   pinch distance for the OK sign — extending the vocabulary to 12 gestures without training a
   new model

**Second CV pipeline — OpenCV.js**
- `cv.Mat`, `cv.cvtColor`, `cv.GaussianBlur`, and `cv.Canny` run live on every frame, cropped to
  the hand's bounding box (derived from the MediaPipe landmarks) and rendered to an inset
  canvas as a real-time edge trace — a second, independent vision pipeline layered on top of
  the landmark model rather than a decorative filter
- Toggleable so it only spends CPU when a judge wants to see it

**Rendering & UI**
- `Canvas2D` (`DrawingUtils` for the hand skeleton overlay, a dedicated canvas for the OpenCV
  edge trace) composited over the mirrored `<video>` element
- `requestAnimationFrame` drives the render/inference loop, gated on `video.currentTime`
  changes so it never re-processes a duplicate frame
- Hand-rolled state machine for gesture stability (N consecutive matching frames before a
  word "fires") and speech/word debouncing to stop rapid-fire repeats

**Language & speech**
- A small phrase-composition engine (`PHRASES` in `app.js`) that watches the last two words
  added to the sentence builder and offers a natural full-sentence upgrade
- Browser `SpeechSynthesis` API for text-to-speech output, no external TTS service

**Design system**
- Hand-authored CSS custom properties for the whole HUD/viewfinder visual language — no CSS
  framework — with `prefers-reduced-motion` support and visible keyboard focus states

Everything above — video, landmarks, both classifiers, the OpenCV pipeline, and speech — runs
entirely on-device in the browser; no camera frame or gesture data is ever sent to a server.

## Next upgrade for the real project

For a stronger hackathon version, collect/obtain a validated dataset for your target sign
language, extract MediaPipe hand landmarks, train a classifier (for example Random Forest,
SVM or a small neural network) instead of the geometric heuristics used here, then export the
model for browser inference. Add temporal sequence recognition for signs that depend on
movement rather than a single hand pose, and give the sentence builder real grammar (word
order, tense) instead of a fixed lookup table of two-word phrases.

Official MediaPipe web documentation:
https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer/web_js
