import {
  GestureRecognizer,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.mjs";

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const overlayCtx = overlay.getContext("2d");
const startBtn = document.getElementById("startBtn");
const statusPill = document.getElementById("statusPill");
const statusText = document.getElementById("statusText");
const handStatus = document.getElementById("handStatus");
const confidence = document.getElementById("confidence");
const eq = document.getElementById("eq");
const gestureText = document.getElementById("gestureText");
const speechText = document.getElementById("speechText");
const resultIcon = document.getElementById("resultIcon");
const speakBtn = document.getElementById("speakBtn");
const history = document.getElementById("history");
const clearBtn = document.getElementById("clearBtn");
const recBadge = document.getElementById("recBadge");
const captionBar = document.getElementById("captionBar");
const captionText = document.getElementById("captionText");
const vocabEls = Array.from(document.querySelectorAll(".vocab"));

// sentence builder elements
const sentenceToggle = document.getElementById("sentenceToggle");
const toggleLabel = document.getElementById("toggleLabel");
const sentenceStrip = document.getElementById("sentenceStrip");
const sentenceEmpty = document.getElementById("sentenceEmpty");
const suggestion = document.getElementById("suggestion");
const suggestionText = document.getElementById("suggestionText");
const suggestionSpeak = document.getElementById("suggestionSpeak");
const speakSentenceBtn = document.getElementById("speakSentenceBtn");
const undoWordBtn = document.getElementById("undoWordBtn");
const clearSentenceBtn = document.getElementById("clearSentenceBtn");

// OpenCV.js edge-trace panel
const cvInset = document.getElementById("cvInset");
const cvToggle = document.getElementById("cvToggle");
const cvCanvas = document.getElementById("cvCanvas");

let recognizer;
let stream;
let running = false;
let lastVideoTime = -1;
let lastCandidate = "";
let stableFrames = 0;
let lastSpoken = "";
let lastSpokenAt = 0;

// ---------- sentence builder state ----------
let sentenceMode = false;
let sentenceWords = []; // array of mapped vocab entries {text, icon, sentence}
let lastSentenceWord = "";
let lastSentenceWordAt = 0;

// ---------- OpenCV.js state ----------
let cvReady = false;
let cvOverlayOn = false;
const cvSourceCanvas = document.createElement("canvas");
const cvSourceCtx = cvSourceCanvas.getContext("2d");

// ---------- vocabulary ----------
// Gestures recognized directly by MediaPipe's built-in classifier.
const MAP = {
  Open_Palm: { text: "HELLO", icon: "✋", sentence: "Hello" },
  Thumb_Up: { text: "YES", icon: "👍", sentence: "Yes" },
  Thumb_Down: { text: "NO", icon: "👎", sentence: "No" },
  Closed_Fist: { text: "HELP", icon: "✊", sentence: "Help" },
  Victory: { text: "THANK YOU", icon: "✌️", sentence: "Thank you" },
  ILoveYou: { text: "I LOVE YOU", icon: "🤟", sentence: "I love you" },
  Pointing_Up: { text: "WAIT", icon: "☝️", sentence: "Wait" }
};

// Extra gestures classified ourselves from the 21 hand landmarks, since
// MediaPipe's canned classifier only ships the 7 shapes above. These give
// the demo a bigger vocabulary without needing a custom-trained model.
const CUSTOM_MAP = {
  OK_Sign: { text: "OKAY", icon: "👌", sentence: "Okay" },
  Three: { text: "PLEASE", icon: "🖖", sentence: "Please" },
  Four: { text: "SORRY", icon: "🖐️", sentence: "Sorry" },
  Rock_On: { text: "STOP", icon: "🤘", sentence: "Stop" },
  Call_Me: { text: "CALL ME", icon: "🤙", sentence: "Call me" }
};

// Combined lookup used anywhere both maps are treated the same way.
const ALL_MAP = { ...MAP, ...CUSTOM_MAP };
const CUSTOM_KEYS = new Set(Object.keys(CUSTOM_MAP));

// Two-word tails that get upgraded into a full suggested sentence while
// building in Sentence Mode. Key = "WORD1,WORD2" (last two words added).
const PHRASES = {
  "HELLO,HELP": "Hello, I need help please.",
  "HELLO,YES": "Hello, yes.",
  "HELLO,NO": "Hello, no.",
  "HELLO,THANK YOU": "Hello, thank you.",
  "HELLO,WAIT": "Hello, please wait a moment.",
  "HELLO,CALL ME": "Hello, please call me.",
  "HELP,STOP": "Please stop, I need help.",
  "HELP,OKAY": "Help is okay now, thank you.",
  "HELP,YES": "Yes, I need help.",
  "HELP,PLEASE": "Please, I need help.",
  "THANK YOU,OKAY": "Thank you, that's okay.",
  "THANK YOU,YES": "Thank you, yes.",
  "THANK YOU,SORRY": "Thank you, and sorry.",
  "SORRY,NO": "Sorry, no.",
  "SORRY,HELP": "Sorry, I need help.",
  "SORRY,STOP": "Sorry, please stop.",
  "PLEASE,YES": "Yes, please.",
  "PLEASE,NO": "No, please.",
  "PLEASE,STOP": "Please stop.",
  "PLEASE,WAIT": "Please wait.",
  "PLEASE,HELP": "Please help me.",
  "CALL ME,HELLO": "Hello, please call me.",
  "CALL ME,THANK YOU": "Please call me, thank you.",
  "I LOVE YOU,THANK YOU": "I love you, thank you.",
  "I LOVE YOU,OKAY": "I love you, okay.",
  "WAIT,OKAY": "Wait, okay.",
  "WAIT,HELP": "Wait, I need help.",
  "NO,STOP": "No, stop.",
  "NO,THANK YOU": "No, thank you.",
  "YES,THANK YOU": "Yes, thank you."
};

// ---------- equalizer meter ----------
const EQ_BARS = 28;
for (let i = 0; i < EQ_BARS; i++) {
  const bar = document.createElement("div");
  bar.className = "bar";
  eq.appendChild(bar);
}
const eqBars = Array.from(eq.children);

function setEq(score) {
  const litCount = Math.round(score * EQ_BARS);
  eqBars.forEach((bar, i) => {
    const active = i < litCount;
    bar.classList.toggle("active", active);
    // slight organic variance so it doesn't look like a flat block
    const wobble = active ? 55 + Math.sin(i * 1.7 + performance.now() / 120) * 20 : 12;
    bar.style.height = `${Math.max(8, Math.min(100, wobble))}%`;
  });
}

function setStatus(text, ready = false) {
  statusText.textContent = text;
  statusPill.classList.toggle("ready", ready);
}

async function initRecognizer() {
  try {
    setStatus("Loading AI model…");
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
    );

    recognizer = await GestureRecognizer.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/gesture_recognizer.task",
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.55,
      minHandPresenceConfidence: 0.55,
      minTrackingConfidence: 0.55,
      cannedGesturesClassifierOptions: {
        maxResults: 1,
        scoreThreshold: 0.55
      }
    });

    setStatus("AI ready — camera not started", true);
    startBtn.disabled = false;
  } catch (err) {
    console.error(err);
    // GPU delegate can fail on some browsers/devices — fall back to CPU automatically.
    try {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
      );
      recognizer = await GestureRecognizer.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/gesture_recognizer.task",
          delegate: "CPU"
        },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.55,
        minHandPresenceConfidence: 0.55,
        minTrackingConfidence: 0.55,
        cannedGesturesClassifierOptions: { maxResults: 1, scoreThreshold: 0.55 }
      });
      setStatus("AI ready (CPU mode)", true);
      startBtn.disabled = false;
    } catch (err2) {
      console.error(err2);
      setStatus("AI failed to load");
      handStatus.textContent = "Could not load MediaPipe. Check your internet connection.";
    }
  }
}

async function startCamera() {
  if (!recognizer) return;

  if (!window.isSecureContext) {
    handStatus.textContent = "Camera requires HTTPS (or localhost). Serve this page over a secure origin.";
    setStatus("Insecure origin");
    return;
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: "user"
      },
      audio: false
    });

    video.srcObject = stream;
    await video.play();

    // Size the overlay canvas to the real video resolution so landmark
    // coordinates (0..1 normalized) map to exact pixels.
    const sizeCanvas = () => {
      overlay.width = video.videoWidth || 1280;
      overlay.height = video.videoHeight || 720;
    };
    if (video.readyState >= 1) sizeCanvas();
    video.addEventListener("loadedmetadata", sizeCanvas);

    running = true;
    startBtn.textContent = "Camera On";
    startBtn.disabled = true;
    handStatus.textContent = "Looking for a hand…";
    setStatus("Camera active — tracking", true);
    recBadge.classList.add("on");

    requestAnimationFrame(loop);
  } catch (err) {
    console.error(err);
    if (err && err.name === "NotAllowedError") {
      handStatus.textContent = "Camera permission denied. Allow camera access in your browser and try again.";
    } else if (err && err.name === "NotFoundError") {
      handStatus.textContent = "No camera found on this device.";
    } else {
      handStatus.textContent = "Camera unavailable. Allow camera access and try again.";
    }
    setStatus("Camera unavailable");
  }
}

function loop() {
  if (!running) return;

  if (video.currentTime !== lastVideoTime) {
    const now = performance.now();
    const result = recognizer.recognizeForVideo(video, now);
    drawLandmarks(result);
    processResult(result);
    lastVideoTime = video.currentTime;
  }

  requestAnimationFrame(loop);
}

function drawLandmarks(result) {
  overlayCtx.save();
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

  if (result.landmarks && result.landmarks.length) {
    const drawer = new DrawingUtils(overlayCtx);
    for (const lm of result.landmarks) {
      drawer.drawConnectors(lm, GestureRecognizer.HAND_CONNECTIONS, {
        color: "rgba(70, 224, 193, 0.65)",
        lineWidth: 3
      });
      drawer.drawLandmarks(lm, {
        color: "#ffb454",
        fillColor: "#ffb454",
        radius: (data) => {
          // fingertips (4,8,12,16,20) drawn slightly larger
          const tip = [4, 8, 12, 16, 20].includes(data.index);
          return tip ? 5 : 3;
        }
      });
    }
  }
  overlayCtx.restore();
}

// ---------- custom landmark-based gesture classifier ----------
// Extends MediaPipe's 7 canned shapes with a few more, detected from raw
// finger geometry so the demo vocabulary isn't capped by the stock model.
function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0));
}

function getFingerStates(lm) {
  const wrist = lm[0];
  // A finger counts as "extended" when its tip sits farther from the wrist
  // than its own mid joint — this stays roughly true no matter how the
  // hand is rotated in frame, which is more robust for a live demo than
  // comparing raw y-coordinates.
  return {
    thumb: dist(wrist, lm[4]) > dist(wrist, lm[2]) * 1.15,
    index: dist(wrist, lm[8]) > dist(wrist, lm[6]) * 1.15,
    middle: dist(wrist, lm[12]) > dist(wrist, lm[10]) * 1.15,
    ring: dist(wrist, lm[16]) > dist(wrist, lm[14]) * 1.15,
    pinky: dist(wrist, lm[20]) > dist(wrist, lm[18]) * 1.15
  };
}

function classifyCustomGesture(lm) {
  const f = getFingerStates(lm);
  const thumbIndexPinch = dist(lm[4], lm[8]);

  // 👌 thumb + index touching, other three fingers open
  if (thumbIndexPinch < 0.06 && f.middle && f.ring && f.pinky) {
    return "OK_Sign";
  }

  // index + middle + ring up, thumb & pinky tucked
  if (f.index && f.middle && f.ring && !f.pinky && !f.thumb) {
    return "Three";
  }

  // all four fingers up, thumb tucked across the palm
  if (f.index && f.middle && f.ring && f.pinky && !f.thumb) {
    return "Four";
  }

  // 🤘 index + pinky up, middle & ring & thumb tucked
  if (f.index && f.pinky && !f.middle && !f.ring && !f.thumb) {
    return "Rock_On";
  }

  // 🤙 thumb + pinky out, others tucked
  if (f.thumb && f.pinky && !f.index && !f.middle && !f.ring) {
    return "Call_Me";
  }

  return null;
}

// ---------- OpenCV.js edge-trace pipeline ----------
// A second, independent CV pipeline running alongside MediaPipe: crops the
// live video to the hand's bounding box each frame and runs real OpenCV.js
// (grayscale -> Gaussian blur -> Canny) to draw a live edge trace. This is
// purely a visual/technical layer — gesture classification still comes
// from MediaPipe + the custom landmark classifier above.
function loadOpenCV() {
  if (window.cv && window.cv.Mat) {
    cvReady = true;
    return;
  }
  const script = document.createElement("script");
  script.src = "https://docs.opencv.org/4.x/opencv.js";
  script.async = true;
  script.onload = () => {
    if (window.cv.getBuildInformation) {
      cvReady = true;
    } else {
      window.cv["onRuntimeInitialized"] = () => {
        cvReady = true;
      };
    }
  };
  script.onerror = () => {
    console.error("OpenCV.js failed to load — edge trace will stay disabled.");
    cvToggle.disabled = true;
  };
  document.head.appendChild(script);
}

function handBoundingBox(lm, width, height, pad = 0.2) {
  let minX = 1, minY = 1, maxX = 0, maxY = 0;
  for (const p of lm) {
    if (p.x < minX) minX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.x > maxX) maxX = p.x;
    if (p.y > maxY) maxY = p.y;
  }
  const w = maxX - minX, h = maxY - minY;
  minX = Math.max(0, minX - w * pad);
  minY = Math.max(0, minY - h * pad);
  maxX = Math.min(1, maxX + w * pad);
  maxY = Math.min(1, maxY + h * pad);
  return {
    x: minX * width,
    y: minY * height,
    w: Math.max(1, (maxX - minX) * width),
    h: Math.max(1, (maxY - minY) * height)
  };
}

function runEdgeOverlay(lm) {
  if (!video.videoWidth) return;
  const box = handBoundingBox(lm, video.videoWidth, video.videoHeight);
  if (box.w < 10 || box.h < 10) return;

  cvSourceCanvas.width = cvCanvas.width;
  cvSourceCanvas.height = cvCanvas.height;
  cvSourceCtx.drawImage(
    video,
    box.x, box.y, box.w, box.h,
    0, 0, cvSourceCanvas.width, cvSourceCanvas.height
  );

  let src, gray, edges;
  try {
    src = cv.imread(cvSourceCanvas);
    gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    cv.GaussianBlur(gray, gray, new cv.Size(5, 5), 0);
    edges = new cv.Mat();
    cv.Canny(gray, edges, 50, 120);
    cv.imshow(cvCanvas, edges);
  } catch (e) {
    // OpenCV can throw on the first frame or two while its runtime is
    // still finishing initialization — safe to ignore and retry next frame.
  } finally {
    if (src) src.delete();
    if (gray) gray.delete();
    if (edges) edges.delete();
  }
}

function clearCvCanvas() {
  const ctx = cvCanvas.getContext("2d");
  ctx.clearRect(0, 0, cvCanvas.width, cvCanvas.height);
}

cvToggle.addEventListener("change", () => {
  cvOverlayOn = cvToggle.checked;
  cvInset.classList.toggle("active", cvOverlayOn);
  if (!cvOverlayOn) clearCvCanvas();
});

function processResult(result) {
  if (!result.landmarks || !result.landmarks.length) {
    clearCvCanvas();
    handStatus.textContent = "No hand detected";
    confidence.textContent = "—";
    setEq(0);
    lastCandidate = "";
    stableFrames = 0;
    clearVocabMatch();
    return;
  }

  const lm = result.landmarks[0];

  if (cvOverlayOn && cvReady) {
    runEdgeOverlay(lm);
  }

  const top = result.gestures?.[0]?.[0];
  let name = top && top.categoryName !== "None" ? top.categoryName : null;
  let score = top?.score || 0;
  let isCustom = false;

  // Fall back to our own landmark classifier when the canned model has no
  // confident match, or matched something outside our demo vocabulary.
  if (!name || !MAP[name]) {
    const custom = classifyCustomGesture(lm);
    if (custom) {
      name = custom;
      score = 0.9;
      isCustom = true;
    } else {
      name = null;
    }
  }

  const mapped = name ? ALL_MAP[name] : null;

  if (!mapped) {
    handStatus.textContent = "No matching gesture";
    confidence.textContent = "—";
    setEq(0);
    lastCandidate = "";
    stableFrames = 0;
    clearVocabMatch();
    return;
  }

  confidence.textContent = `${Math.round(score * 100)}%`;
  setEq(score);
  handStatus.textContent = `Detected: ${mapped.text}${isCustom ? " (shape match)" : ""}`;
  highlightVocabMatch(name);

  if (name === lastCandidate) {
    stableFrames++;
  } else {
    lastCandidate = name;
    stableFrames = 1;
  }

  // Require several consecutive frames so the app doesn't spam speech.
  if (stableFrames >= 8) {
    if (sentenceMode) {
      addWordToSentence(mapped);
    } else {
      showTranslation(mapped, score);

      const now = Date.now();
      if (mapped.sentence !== lastSpoken || now - lastSpokenAt > 3500) {
        speak(mapped.sentence);
        lastSpoken = mapped.sentence;
        lastSpokenAt = now;
        addHistory(mapped);
      }
    }
    stableFrames = 0;
  }
}

function highlightVocabMatch(name) {
  vocabEls.forEach((el) => {
    el.classList.toggle("match", el.dataset.key === name);
  });
}

function clearVocabMatch() {
  vocabEls.forEach((el) => el.classList.remove("match"));
}

function showTranslation(mapped, score) {
  resultIcon.textContent = mapped.icon;
  resultIcon.classList.remove("pop");
  void resultIcon.offsetWidth; // restart animation
  resultIcon.classList.add("pop");

  gestureText.textContent = mapped.text;
  speechText.textContent = `"${mapped.sentence}"`;
  speakBtn.disabled = false;
  speakBtn.dataset.text = mapped.sentence;

  captionText.textContent = mapped.sentence.toUpperCase();
  captionText.classList.remove("live");
  void captionText.offsetWidth;
  captionText.classList.add("live");
}

function speak(text) {
  if (!("speechSynthesis" in window)) return;

  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "en-US";
  utterance.rate = 0.95;
  utterance.pitch = 1;
  window.speechSynthesis.speak(utterance);
}

function addHistory(mapped) {
  const empty = history.querySelector(".empty");
  if (empty) empty.remove();

  const item = document.createElement("div");
  item.className = "history-item";
  item.innerHTML = `${mapped.icon} &nbsp; <b>${mapped.text}</b><span>${new Date().toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"})}</span>`;
  history.prepend(item);

  while (history.children.length > 8) {
    history.lastElementChild.remove();
  }
}

// ---------- sentence builder ----------
function addWordToSentence(mapped) {
  const now = Date.now();
  // Debounce so holding one gesture doesn't spam the same word repeatedly.
  if (mapped.text === lastSentenceWord && now - lastSentenceWordAt < 3500) return;
  lastSentenceWord = mapped.text;
  lastSentenceWordAt = now;

  sentenceWords.push(mapped);
  showTranslation(mapped, 1);
  renderSentence();
  updateSuggestion();
}

function renderSentence() {
  sentenceStrip.innerHTML = "";
  if (!sentenceWords.length) {
    sentenceStrip.appendChild(sentenceEmpty);
    speakSentenceBtn.disabled = true;
    undoWordBtn.disabled = true;
    clearSentenceBtn.disabled = true;
    return;
  }

  sentenceWords.forEach((w) => {
    const chip = document.createElement("span");
    chip.className = "word-chip";
    chip.textContent = `${w.icon} ${w.text}`;
    sentenceStrip.appendChild(chip);
  });

  speakSentenceBtn.disabled = false;
  undoWordBtn.disabled = false;
  clearSentenceBtn.disabled = false;
}

function updateSuggestion() {
  if (sentenceWords.length < 2) {
    suggestion.hidden = true;
    return;
  }
  const [a, b] = sentenceWords.slice(-2);
  const key = `${a.text},${b.text}`;
  const phrase = PHRASES[key];
  if (phrase) {
    suggestionText.textContent = phrase;
    suggestion.hidden = false;
    suggestion.dataset.text = phrase;
  } else {
    suggestion.hidden = true;
  }
}

function currentSentenceText() {
  return sentenceWords.map((w) => w.sentence).join(", ");
}

sentenceToggle.addEventListener("change", () => {
  sentenceMode = sentenceToggle.checked;
  toggleLabel.textContent = sentenceMode ? "On" : "Off";
  document.body.classList.toggle("sentence-active", sentenceMode);
});

speakSentenceBtn.addEventListener("click", () => {
  if (sentenceWords.length) speak(currentSentenceText());
});

undoWordBtn.addEventListener("click", () => {
  sentenceWords.pop();
  renderSentence();
  updateSuggestion();
});

clearSentenceBtn.addEventListener("click", () => {
  sentenceWords = [];
  lastSentenceWord = "";
  renderSentence();
  updateSuggestion();
});

suggestionSpeak.addEventListener("click", () => {
  if (suggestion.dataset.text) speak(suggestion.dataset.text);
});

startBtn.disabled = true;
startBtn.addEventListener("click", startCamera);

speakBtn.addEventListener("click", () => {
  const text = speakBtn.dataset.text;
  if (text) speak(text);
});

clearBtn.addEventListener("click", () => {
  history.innerHTML = '<div class="empty">No translations yet</div>';
});

setEq(0);
renderSentence();
initRecognizer();
loadOpenCV();
